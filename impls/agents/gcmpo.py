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


class GCOfflineMPOAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Offline MPO with Conservative Q-Learning."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch: Dict, grad_params: Any) -> Tuple[jnp.ndarray, Dict]:
        """Compute the IQL value loss to learn V(s,g)."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['goals'], params=grad_params)
        loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return loss, {
            'value_loss': loss,
            'v_mean': v.mean(),
        }

    def critic_loss(self, batch: Dict, grad_params: Any, rng: Any) -> Tuple[jnp.ndarray, Dict]:
        """Compute the Conservative Q-Learning (CQL) critic loss."""
        # 1. Standard Bellman Error
        next_v = self.network.select('value')(batch['next_observations'], batch['goals'])
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['goals'], batch['actions'], params=grad_params
        )
        td_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        # 2. CQL Regularizer
        batch_size = batch['observations'].shape[0]
        cql_n_actions = self.config['cql_n_actions']
        
        rng, rand_key, policy_key = jax.random.split(rng, 3)

        # Sample actions from a uniform distribution
        if self.config['discrete']:
             # Not implemented for simplicity, focus is on continuous control for MPO
            raise NotImplementedError("Discrete action space not supported for this CQL implementation.")
        else:
            action_dim = batch['actions'].shape[-1]
            rand_actions = jax.random.uniform(rand_key, (batch_size, cql_n_actions, action_dim), minval=-1.0, maxval=1.0)
        
        # Sample actions from the current policy
        policy_dist = self.network.select('actor')(batch['observations'], batch['goals'])
        policy_actions = policy_dist.sample(seed=policy_key, sample_shape=(cql_n_actions,)).transpose(1, 0, 2)
        
        # Repeat observations and goals to match sampled action dimensions
        repeated_obs = jnp.repeat(batch['observations'][:, jnp.newaxis, :], cql_n_actions, axis=1)
        repeated_goals = jnp.repeat(batch['goals'][:, jnp.newaxis, :], cql_n_actions, axis=1)

        # Get Q-values for sampled actions
        q1_rand, q2_rand = self.network.select('critic')(repeated_obs, repeated_goals, rand_actions, params=grad_params)
        q1_policy, q2_policy = self.network.select('critic')(repeated_obs, repeated_goals, policy_actions, params=grad_params)
        
        # Log-Sum-Exp over sampled actions for both critics
        cql_logsumexp1 = jnp.logsumexp(jnp.concatenate([q1_rand, q1_policy], axis=1), axis=1).mean()
        cql_logsumexp2 = jnp.logsumexp(jnp.concatenate([q2_rand, q2_policy], axis=1), axis=1).mean()

        # Get Q-values for actions from the dataset
        dataset_q1, dataset_q2 = self.network.select('critic')(
            batch['observations'], batch['goals'], batch['actions'], params=grad_params
        )

        # The CQL loss penalizes high Q-values for OOD actions
        cql_loss = (cql_logsumexp1 - dataset_q1.mean()) + (cql_logsumexp2 - dataset_q2.mean())
        
        # Final critic loss is TD loss + weighted CQL regularizer
        total_loss = td_loss + self.config['cql_alpha'] * cql_loss

        return total_loss, {
            'critic_loss': total_loss,
            'td_loss': td_loss,
            'cql_loss': cql_loss,
            'q_mean': ((q1 + q2) / 2).mean(),
        }

    def actor_loss(self, batch: Dict, grad_params: Any) -> Tuple[jnp.ndarray, Dict]:
        """Compute the MPO actor loss."""
        # This part is identical to the online MPO implementation
        v = self.network.select('value')(batch['observations'], batch['actor_goals'])
        q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v
        
        weights = jnp.exp(adv * self.config['mpo_alpha'])
        weights = jax.lax.stop_gradient(jnp.minimum(weights, self.config['mpo_clip']))
        
        dist_actor = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist_actor.log_prob(batch['actions'])
        loss = -(weights * log_prob).mean()
        
        return loss, {
            'actor_loss': loss,
            'adv_mean': adv.mean(),
            'weights_mean': weights.mean(),
        }

    @jax.jit
    def update(self, batch: Dict) -> Tuple[Any, Dict]:
        """Update the agent and return a new agent with training info."""
        new_rng, critic_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            info = {}
            val_loss, val_info = self.value_loss(batch, grad_params)
            info.update({f'value/{k}': v for k, v in val_info.items()})

            cri_loss, cri_info = self.critic_loss(batch, grad_params, critic_rng)
            info.update({f'critic/{k}': v for k, v in cri_info.items()})
            
            act_loss, act_info = self.actor_loss(batch, grad_params)
            info.update({f'actor/{k}': v for k, v in act_info.items()})
            
            total_loss = val_loss + cri_loss + act_loss
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')

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
        action_dim = ex_actions.shape[-1]
        
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
        
        value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], gc_encoder=encoders.get('value'))
        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=True, gc_encoder=encoders.get('critic'))
        actor_def = GCActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim, gc_encoder=encoders.get('actor'))
        
        network_info = {
            'value': (value_def, (ex_observations, ex_goals)),
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        params = network.params.unfreeze()
        params['modules_target_critic'] = params['modules_critic']
        network = network.replace(params=FrozenDict(params))

        return cls(rng, network=network, config=FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='gcofflinempo',
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(256, 256),
        value_hidden_dims=(256, 256),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        
        # IQL specific
        expectile=0.9, # Higher expectile for more conservative V-function

        # MPO specific
        mpo_alpha=10.0,
        mpo_clip=100.0,
        
        # Offline (CQL) specific
        cql_alpha=5.0, # Weight of the conservative regularizer. This is a key hyperparameter.
        cql_n_actions=10, # Number of actions to sample for Log-Sum-Exp term.

        # General
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        # Dataset hyperparameters...
    ))
