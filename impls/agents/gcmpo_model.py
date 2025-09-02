import copy
from typing import Any, Dict, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.core import FrozenDict
from tensorflow_probability.substrates import jax as tfp

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCValue

# Use tensorflow probability for distributions
dist = tfp.distributions

# --- New Network Definition for the Dynamics Model ---

class GCEnsembleModel(nn.Module):
    """Goal-Conditioned Ensemble of Probabilistic Dynamics Models."""
    hidden_dims: Tuple[int, ...]
    ensemble_size: int = 7
    layer_norm: bool = False
    gc_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray, actions: jnp.ndarray) -> dist.MultivariateNormalDiag:
        
        # Vmap over the ensemble dimension
        vmap_fn = nn.vmap(
            self.build_single_model,
            in_axes=None,
            out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True},
        )
        # Pass dummy obs/actions to initialize correctly
        next_state_dist, reward_dist = vmap_fn(name='Ensemble')(observations, goals, actions)
        return next_state_dist, reward_dist

    def build_single_model(self, observations: jnp.ndarray, goals: jnp.ndarray, actions: jnp.ndarray):
        # This function defines a single model in the ensemble
        state_dim = observations.shape[-1]
        
        # Encoder for observations and goals
        x = self.gc_encoder(observations, goals)
        # Concatenate actions
        x = jnp.concatenate([x, actions], axis=-1)

        net = []
        for h in self.hidden_dims:
            net.append(nn.Dense(h))
            if self.layer_norm:
                net.append(nn.LayerNorm())
            net.append(nn.relu)
        
        mlp = nn.Sequential(net)
        features = mlp(x)

        # Predict mean and log_std for next state
        next_state_mean = nn.Dense(state_dim, name='next_state_mean')(features)
        next_state_log_std = nn.Dense(state_dim, name='next_state_log_std')(features)
        
        # Predict mean and log_std for reward
        reward_mean = nn.Dense(1, name='reward_mean')(features)
        reward_log_std = nn.Dense(1, name='reward_log_std')(features)

        # Clip log_std for stability
        log_std_min, log_std_max = -10.0, 2.0
        next_state_log_std = jnp.clip(next_state_log_std, log_std_min, log_std_max)
        reward_log_std = jnp.clip(reward_log_std, log_std_min, log_std_max)

        # The model predicts the *change* in state (delta)
        next_state_mean += observations

        next_state_dist = dist.MultivariateNormalDiag(loc=next_state_mean, scale_diag=jnp.exp(next_state_log_std))
        reward_dist = dist.MultivariateNormalDiag(loc=jnp.squeeze(reward_mean, -1), scale_diag=jnp.exp(jnp.squeeze(reward_log_std, -1)))

        return next_state_dist, reward_dist


class GCMBPOAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Model-Based MPO."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    # --- Model Loss ---
    def model_loss(self, batch, grad_params):
        next_state_dist, reward_dist = self.network.select('model')(
            batch['observations'], batch['goals'], batch['actions'], params=grad_params
        )
        # Maximize the log probability of the true next_state and reward
        log_prob_next_state = next_state_dist.log_prob(batch['next_observations'])
        log_prob_reward = reward_dist.log_prob(batch['rewards'])
        
        # Loss is the negative log-likelihood, averaged over the ensemble and batch
        loss = -(log_prob_next_state.mean() + log_prob_reward.mean())
        
        return loss, {
            'model_loss': loss,
            'model_log_prob_state': log_prob_next_state.mean(),
            'model_log_prob_reward': log_prob_reward.mean(),
        }
    
    # --- Actor-Critic Losses (trained on mixed data) ---
    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['goals'], params=grad_params)
        loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return loss, {'value_loss': loss, 'v_mean': v.mean()}

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select('value')(batch['next_observations'], batch['goals'])
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['goals'], batch['actions'], params=grad_params
        )
        loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return loss, {'critic_loss': loss, 'q_mean': ((q1 + q2) / 2).mean()}

    def actor_loss(self, batch, grad_params):
        v = self.network.select('value')(batch['observations'], batch['actor_goals'])
        q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v
        weights = jnp.exp(adv * self.config['mpo_alpha'])
        weights = jax.lax.stop_gradient(jnp.minimum(weights, self.config['mpo_clip']))
        
        dist_actor = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist_actor.log_prob(batch['actions'])
        loss = -(weights * log_prob).mean()
        
        return loss, {'actor_loss': loss, 'adv_mean': adv.mean()}

    # --- Main Update Function ---
    @jax.jit
    def update(self, real_batch: Dict):
        new_rng, model_rng, rollout_rng = jax.random.split(self.rng, 3)

        # 1. Update the dynamics model on the real batch
        def model_loss_fn(p): return self.model_loss(real_batch, p)
        new_network, model_info = self.network.apply_loss_fn(loss_fn=model_loss_fn, module_name='model')

        # 2. Generate imaginary data using the updated model
        # Select a random model from the ensemble for each trajectory
        model_indices = jax.random.randint(rollout_rng, (self.config['imaginary_batch_size'],), 0, self.config['model_ensemble_size'])
        
        def rollout_step(carry, _):
            obs, goals, rng_step = carry
            rng_action, rng_model = jax.random.split(rng_step)
            actions = self.sample_actions(obs, goals, seed=rng_action, temperature=1.0)
            
            # Predict with the ensemble
            next_state_dist, reward_dist = new_network.select('model')(obs, goals, actions)
            
            # Select predictions from the chosen models for each item in the batch
            def select_from_ensemble(dist):
                return dist.__class__(
                    loc=jax.vmap(lambda x, i: x[i])(dist.loc, model_indices),
                    scale_diag=jax.vmap(lambda x, i: x[i])(dist.scale_diag, model_indices)
                )

            next_obs = select_from_ensemble(next_state_dist).sample(seed=rng_model)
            rewards = select_from_ensemble(reward_dist).sample(seed=rng_model)
            
            # Create a transition and prepare for next step
            transition = {
                'observations': obs, 'goals': goals, 'actions': actions,
                'next_observations': next_obs, 'rewards': rewards,
                'masks': jnp.ones_like(rewards), 'actor_goals': goals # Keep goals consistent
            }
            return (next_obs, goals, rng_model), transition
        
        # Start rollouts from states in the real batch
        start_indices = jax.random.choice(model_rng, real_batch['observations'].shape[0], (self.config['imaginary_batch_size'],), replace=True)
        initial_obs = real_batch['observations'][start_indices]
        initial_goals = real_batch['goals'][start_indices]

        # Generate rollouts
        _, imaginary_transitions = jax.lax.scan(
            rollout_step, (initial_obs, initial_goals, rollout_rng), None, length=self.config['rollout_horizon']
        )
        # Reshape from (horizon, batch, ...) to (horizon * batch, ...)
        imaginary_batch = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), imaginary_transitions)
        
        # 3. Combine real and imaginary data and update actor/critic
        def mix_batches(b1, b2):
            return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), b1, b2)

        mixed_batch = mix_batches(real_batch, imaginary_batch)

        def ac_loss_fn(grad_params):
            val_loss, val_info = self.value_loss(mixed_batch, grad_params)
            cri_loss, cri_info = self.critic_loss(mixed_batch, grad_params)
            act_loss, act_info = self.actor_loss(mixed_batch, grad_params)
            total_loss = val_loss + cri_loss + act_loss
            return total_loss, {**val_info, **cri_info, **act_info}

        new_network, ac_info = new_network.apply_loss_fn(loss_fn=ac_loss_fn, module_name=['value', 'critic', 'actor'])

        # 4. Target network updates
        new_network = self.target_update(new_network, 'critic')

        # Combine all info dicts
        info = {**model_info, **ac_info}
        return self.replace(network=new_network, rng=new_rng), info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        params = network.params.unfreeze()
        params[f'modules_target_{module_name}'] = new_target_params
        return network.replace(params=FrozenDict(params))

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        
        # Define encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['model'] = GCEncoder(concat_encoder=encoder_module())
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
        
        # Define networks
        model_def = GCEnsembleModel(
            hidden_dims=config['model_hidden_dims'],
            ensemble_size=config['model_ensemble_size'],
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('model')
        )
        
        value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], gc_encoder=encoders.get('value'))
        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=True, gc_encoder=encoders.get('critic'))
        actor_def = GCActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim, gc_encoder=encoders.get('actor'))
        
        network_info = {
            'model': (model_def, (ex_observations, ex_goals, ex_actions)),
            'value': (value_def, (ex_observations, ex_goals)),
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        # Create separate optimizers for model and actor-critic
        network_txs = {
            'model': optax.adam(learning_rate=config['lr']),
            'value': optax.adam(learning_rate=config['lr']),
            'critic': optax.adam(learning_rate=config['lr']),
            'actor': optax.adam(learning_rate=config['lr']),
        }

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, txs=network_txs)
        
        params = network.params.unfreeze()
        params['modules_target_critic'] = params['modules_critic']
        network = network.replace(params=FrozenDict(params))

        return cls(rng, network=network, config=FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='gcmbpo',
        lr=3e-4,
        batch_size=256, # Real batch size
        actor_hidden_dims=(256, 256),
        value_hidden_dims=(256, 256),
        model_hidden_dims=(256, 256, 256),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        
        # IQL/MPO specific
        expectile=0.7,
        mpo_alpha=5.0,
        mpo_clip=100.0,

        # Model-Based specific
        model_ensemble_size=7,
        rollout_horizon=5,
        imaginary_batch_size=256, # Number of rollout starting states

        # General
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        # Dataset hyperparameters...
    ))
