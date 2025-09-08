import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.core import FrozenDict

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (GCActor, GCDiscreteActor, GCDiscreteCritic,
                            GCValue)


class GCXQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Xtreme Q-learning (GCXQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # --- Loss Helper Functions from XQL ---

    @staticmethod
    def gumbel_rescale_loss(diff, alpha, max_clip=None):
        """
        Gumbel loss J: E[e^x - x - 1], rescaled for stability.
        Source: https://github.com/Div99/XQL/blob/main/online/research/xql.py#L210
        """
        z = diff / alpha
        if max_clip is not None:
            z = jnp.minimum(z, max_clip)
        max_z = jnp.max(z, axis=0)
        max_z = jnp.where(max_z < -1.0, -1.0, max_z)
        max_z = jax.lax.stop_gradient(max_z)
        loss = (
            jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)
        )
        return loss

    @staticmethod
    def huber_loss(x, delta: float = 1.0):
        """
        Huber loss, less sensitive to outliers than MSE.
        Source: https://github.com/Div99/XQL/blob/main/online/research/xql.py#L268
        """
        abs_x = jnp.abs(x)
        quadratic = jnp.minimum(abs_x, delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic**2 + delta * linear

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Standard expectile loss for the vanilla IQL baseline."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    # --- Core XQL Loss Functions ---

    def value_loss(self, batch: Dict, grad_params: Any) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute the XQL value loss using the Gumbel or expectile objective.
        This function learns V(s, g).
        """
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        
        diff = q - v

        if self.config.get('vanilla', False):  # 'vanilla' means use standard expectile loss
            value_loss = self.expectile_loss(diff, diff, self.config['expectile']).mean()
        else:
            value_loss = self.gumbel_rescale_loss(
                diff,
                alpha=self.config['loss_temp'],
                max_clip=self.config['max_clip']
            ).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch: Dict, grad_params: Any) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute the XQL critic loss using the Huber objective.
        This function learns Q(s, a, g).
        """
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        
        # XQL uses Huber loss for the critic for robustness
        critic_loss = (self.huber_loss(q1 - target_q, delta=20.0) + self.huber_loss(q2 - target_q, delta=20.0)).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': ((q1 + q2) / 2).mean(),
        }

    def actor_loss(self, batch: Dict, grad_params: Any, rng: Any = None) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute the XQL actor loss (advantage-weighted regression).
        This part is structurally identical to AWR, but uses V and Q trained with XQL losses.
        """
        v = self.network.select('value')(batch['observations'], batch['actor_goals'])
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['actor_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['beta'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            })
        
        return actor_loss, actor_info

    # --- Agent Update and Action Sampling ---
    
    @jax.jit
    def update(self, batch: Dict) -> Tuple[Any, Dict]:
        """Update the agent and return a new agent with training info."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            info = {}
            
            value_loss, value_info = self.value_loss(batch, grad_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            critic_loss, critic_info = self.critic_loss(batch, grad_params)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v

            actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            total_loss = value_loss + critic_loss + actor_loss
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
    
    # --- Agent Creation (Boilerplate) ---

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
        
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
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
                const_std=True,
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
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
    config = ml_collections.ConfigDict(dict(
        # Agent hyperparameters
        agent_name='gcxql',
        lr=3e-4,
        batch_size=1024,
        actor_hidden_dims=(256, 256),
        value_hidden_dims=(256, 256),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        
        # XQL specific
        expectile=0.7,      # Used if vanilla=True
        beta=6.0,           # Actor advantage temperature
        vanilla=False,      # If True, uses expectile loss for V. If False, uses Gumbel loss.
        loss_temp=1.0,      # Temperature for Gumbel loss
        max_clip=7.0,       # Clipping value for Gumbel loss term for stability

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
    ))
    return config
