import copy
from typing import Any, Tuple, Dict

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
from utils.networks import GCActor, GCDiscreteActor

# Use tensorflow probability for distributions
dist = tfp.distributions

# --- New Network Definitions for Successor Measures ---

class GCSuccessorMeasure(nn.Module):
    """Goal-Conditioned Successor Measure (Value component) Network."""
    hidden_dims: Tuple[int, ...]
    layer_norm: bool = False
    ensemble: bool = True
    gc_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        # The SM net learns the expected discounted future state visitations, which is the value function
        # under an indicator reward.
        x = self.gc_encoder(observations, goals)
        
        def build_mlp(name):
            net = []
            for h in self.hidden_dims:
                net.append(nn.Dense(h, name=f'{name}_fc_{len(net)}'))
                if self.layer_norm:
                    net.append(nn.LayerNorm(name=f'{name}_ln_{len(net)}'))
                net.append(nn.relu)
            net.append(nn.Dense(1, name=f'{name}_out'))
            return nn.Sequential(net)

        if self.ensemble:
            v1 = build_mlp('V1')(x)
            v2 = build_mlp('V2')(x)
            return jnp.squeeze(v1, -1), jnp.squeeze(v2, -1)
        else:
            v = build_mlp('V')(x)
            return jnp.squeeze(v, -1)

class GCReward(nn.Module):
    """Goal-Conditioned Reward Network."""
    hidden_dims: Tuple[int, ...]
    layer_norm: bool = False
    gc_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        x = self.gc_encoder(observations, goals)
        net = []
        for h in self.hidden_dims:
            net.append(nn.Dense(h))
            if self.layer_norm:
                net.append(nn.LayerNorm())
            net.append(nn.relu)
        net.append(nn.Dense(1))
        r = nn.Sequential(net)(x)
        return jnp.squeeze(r, -1)

# --- The MPO Agent Implementation ---

class GCMPOAgent(flax.struct.PyTreeNode):
    """Goal-conditioned MPO with Successor Measures."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_sm_loss(self, batch: Dict, grad_params: Any) -> Tuple[jnp.ndarray, Dict]:
        """
        Computes the combined loss for the Successor Measure (SM) and Reward networks.
        This function replaces the separate value and critic losses from IQL.
        """
        # 1. Train the Reward Network
        pred_rewards = self.network.select('reward')(batch['observations'], batch['goals'], params=grad_params)
        reward_loss = ((pred_rewards - batch['rewards']) ** 2).mean()

        # 2. Train the Successor Measure Network (learns the value function)
        # The TD target is R(s,g) + gamma * V_target(s', g)
        next_sm1, next_sm2 = self.network.select('target_sm')(batch['next_observations'], batch['goals'])
        next_sm = jnp.minimum(next_sm1, next_sm2)
        
        # We use the predicted reward for the target, not the ground truth reward.
        # This makes the SM network learn the value function consistent with the learned reward function.
        target_q = jax.lax.stop_gradient(pred_rewards + self.config['discount'] * batch['masks'] * next_sm)
        
        sm1, sm2 = self.network.select('sm')(batch['observations'], batch['goals'], params=grad_params)
        sm_loss = ((sm1 - target_q) ** 2 + (sm2 - target_q) ** 2).mean()
        
        total_loss = reward_loss + sm_loss

        return total_loss, {
            'critic_sm_loss': total_loss,
            'reward_loss': reward_loss,
            'sm_loss': sm_loss,
            'sm_mean': ((sm1 + sm2) / 2).mean(),
            'reward_pred_mean': pred_rewards.mean(),
        }

    def actor_loss(self, batch: Dict, grad_params: Any, rng: Any) -> Tuple[jnp.ndarray, Dict]:
        """
        Computes the MPO actor loss (M-Step).
        """
        # --- E-Step (Implicit): Compute Advantage ---
        # Reconstruct Q-value from SM and Reward networks.
        # For the actor update, we use the actor's goal, not the value goal.
        rewards = self.network.select('reward')(batch['observations'], batch['actor_goals'])
        sm1, sm2 = self.network.select('sm')(batch['observations'], batch['actor_goals'])
        q = rewards + jnp.minimum(sm1, sm2)

        # To compute advantage A(s,a) = Q(s,a) - V(s), we need V(s).
        # V(s) = E_{a' ~ pi(a'|s)} [Q(s, a')]
        # We approximate this expectation by sampling actions from the current policy.
        def get_v(obs, goals):
            dist_v = self.network.select('actor')(obs, goals)
            # Sample multiple actions to get a stable estimate of V
            rng_v, key = jax.random.split(rng)
            actions_v = dist_v.sample(seed=key, sample_shape=(self.config['v_samples'],)) # (N, B, A)
            
            # Tile obs and goals to match the sampled actions shape
            tiled_obs = jnp.tile(obs, (self.config['v_samples'], 1, 1))
            tiled_goals = jnp.tile(goals, (self.config['v_samples'], 1, 1))
            
            rewards_v = self.network.select('reward')(tiled_obs, tiled_goals)
            sm1_v, sm2_v = self.network.select('sm')(tiled_obs, tiled_goals)
            q_v = rewards_v + jnp.minimum(sm1_v, sm2_v)
            return q_v.mean(axis=0)
            
        v = get_v(batch['observations'], batch['actor_goals'])
        adv = q - v

        # --- M-Step: Weighted Maximum Likelihood ---
        # The weights are based on the exponentiated advantage.
        weights = jnp.exp(adv * self.config['mpo_alpha'])
        weights = jax.lax.stop_gradient(jnp.minimum(weights, self.config['mpo_clip']))

        dist_actor = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist_actor.log_prob(batch['actions'])

        actor_loss = -(weights * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv_mean': adv.mean(),
            'weights_mean': weights.mean(),
            'bc_log_prob': log_prob.mean(),
        }

        if not self.config['discrete']:
            actor_info.update({
                'mse': jnp.mean((dist_actor.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist_actor.scale_diag),
            })
        
        return actor_loss, actor_info

    @jax.jit
    def update(self, batch: Dict) -> Tuple[Any, Dict]:
        """Update the agent and return a new agent with training info."""
        new_rng, actor_rng, v_rng = jax.random.split(self.rng, 3)

        def loss_fn(grad_params):
            info = {}
            critic_sm_loss, critic_sm_info = self.critic_sm_loss(batch, grad_params)
            for k, v in critic_sm_info.items():
                info[f'critic_sm/{k}'] = v

            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            
            total_loss = critic_sm_loss + actor_loss
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
        # Target network updates
        new_network = self.target_update(new_network, 'sm')
        new_network = self.target_update(new_network, 'reward')

        return self.replace(network=new_network, rng=new_rng), info

    def target_update(self, network: TrainState, module_name: str) -> TrainState:
        """Soft update for a target network."""
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
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['sm'] = GCEncoder(concat_encoder=encoder_module())
            encoders['reward'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
        
        # Define network components
        sm_def = GCSuccessorMeasure(
            hidden_dims=config['sm_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('sm')
        )

        reward_def = GCReward(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('reward')
        )
        
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=True, # MPO often works better with fixed std
                gc_encoder=encoders.get('actor'),
            )
        
        network_info = dict(
            sm=(sm_def, (ex_observations, ex_goals)),
            target_sm=(copy.deepcopy(sm_def), (ex_observations, ex_goals)),
            reward=(reward_def, (ex_observations, ex_goals)),
            target_reward=(copy.deepcopy(reward_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        # Initialize target networks
        params = network.params.unfreeze()
        params['modules_target_sm'] = params['modules_sm']
        params['modules_target_reward'] = params['modules_reward']
        network = network.replace(params=FrozenDict(params))

        return cls(rng, network=network, config=FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(dict(
        # Agent hyperparameters
        agent_name='gcmpo',
        lr=3e-4,
        batch_size=1024,
        actor_hidden_dims=(256, 256),
        sm_hidden_dims=(256, 256),
        reward_hidden_dims=(256, 256),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        
        # MPO specific
        mpo_alpha=10.0,      # Temperature param 'eta' for weighting advantage. Lower is more greedy.
        mpo_clip=100.0,      # Maximum weight clip for stability.
        v_samples=16,        # Number of actions to sample for V-function estimate.

        # General
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        
        # Dataset hyperparameters (can be inherited)
        dataset_class='GCDataset',
        value_p_curgoal=0.2,
        value_p_trajgoal=0.5,
        value_p_randomgoal=0.3,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
    return config
