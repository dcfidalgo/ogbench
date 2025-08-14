from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict, Union
import json
from agents import agents
from utils.env_utils import make_env_and_datasets
from utils.datasets import GCDataset, HGCDataset, Dataset
import random
import numpy as np
import flax
import pickle
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp


def restore_agent_and_env_from_flags(
    flags_path: str | Path, params_path: Optional[str | Path] = None
) -> Tuple['Agent', Tuple['Env', 'TrainDataset', 'ValDataset']]:
    """Restore the agent from a flags file and a parameters file.

    Args:
        flags_path: Path to the flags file.
        params_path: Path to the parameters file. If `None`, it will not restore the parameters.

    Returns:
        The restored agent.
    """
    flags_path = Path(flags_path)
    flags_dict = json.loads(flags_path.read_text())

    agent_class = agents[flags_dict['agent']['agent_name']]

    env, train_dataset, val_dataset = make_env_and_datasets(
        flags_dict['env_name'], frame_stack=flags_dict['agent']['frame_stack']
    )

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[flags_dict['agent']['dataset_class']]

    train_dataset = dataset_class(Dataset.create(**train_dataset), flags_dict['agent'])
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), flags_dict['agent'])

    # Initialize agent.
    random.seed(flags_dict['seed'])
    np.random.seed(flags_dict['seed'])

    example_batch = train_dataset.sample(1)
    if flags_dict['agent']['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[flags_dict['agent']['agent_name']]
    agent = agent_class.create(
        flags_dict['seed'],
        example_batch['observations'],
        example_batch['actions'],
        flags_dict['agent'],
    )

    if params_path is not None:
        params_path = Path(params_path)
        with params_path.open('rb') as f:
            load_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        print(f'Restored from {params_path}')

    return agent, (env, train_dataset, val_dataset)


def load_params(path: str | Path, agent: 'Agent') -> 'Agent':
    path = Path(path)
    with path.open('rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    # print(f"Loaded params from '{path}' to agent")

    return agent


def get_trajectories(dataset, n: Optional[int] = None, min_length: int = 2) -> List[Dict[str, np.ndarray]]:
    trajectories = []
    for i, j in zip(dataset.initial_locs, dataset.terminal_locs):
        trajectory = dataset.dataset.get_subset(np.arange(i, j + 1))

        if len(trajectory['observations']) < min_length:
            continue

        trajectories.append(trajectory)

        if n is not None and len(trajectories) >= n:
            break

    return trajectories


def interpolate(observations: np.ndarray, n_interpolations: int = 9) -> np.ndarray:
    new_observations = []
    for j in range(1, len(observations)):
        a, b = observations[j - 1], observations[j]
        delta = (b - a) / (n_interpolations + 1)
        new_observations.append(a)
        for k in range(1, n_interpolations + 1):
            new_observation = a + k * delta
            new_observations.append(new_observation)
    new_observations.append(observations[-1])

    return np.array(new_observations)


def get_states(intermediates: dict) -> np.ndarray:
    activations = intermediates['intermediates']['modules_actor']['actor_net']
    dense_layer_names = list(sorted([name for name in activations if 'Dense' in name]))
    activations = [activations[name]['__call__'][0] > 0 for name in dense_layer_names]
    if activations[0].ndim == 1:
        activations = [activation[None, ...] for activation in activations]
    neurons = np.concatenate(activations, axis=1, dtype=bool)

    return neurons


def halve(state: np.ndarray) -> np.ndarray:
    # Vectorized approach using numpy operations
    if len(state) % 2 == 0:
        # Even length: reshape and use logical OR along axis 1
        return np.logical_or(state[::2], state[1::2])
    else:
        # Odd length: handle separately
        pairs = np.logical_or(state[:-1:2], state[1::2])
        return np.concatenate([pairs, state[-1:]])


def save_rendered_goal(info, path: str | Path = './rendered_goal.png'):
    import matplotlib.pyplot as plt

    path = Path(path)

    plt.imshow(info['goal_rendered'])
    # plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def compute_training_states(
    folder: str | Path = 'impls/exp/OGBench/Debug/sd000_s_1134105.0.20250716_161902',
    param_paths: Optional[List[Path]] = None,
    n_trajectories: Optional[int] = None,
    include_initial: bool = True,
    n_interpolations: int = 9,
    batch_size: Optional[int] = None,
    max_obs_per_trajectory: Optional[int] = None,
) -> Dict[Path, List[List[np.ndarray]]]:
    folder = Path(folder)
    agent, (env, train_dataset, val_dataset) = restore_agent_and_env_from_flags(folder / 'flags.json')
    trajectories = get_trajectories(train_dataset, n=n_trajectories)

    param_paths = param_paths or get_sorted_param_paths(folder)
    if include_initial:
        param_paths.insert(0, Path('initial'))

    data = defaultdict(list)

    for param_path in tqdm(param_paths):
        if param_path != Path('initial'):
            agent = load_params(param_path, agent)
        actor = agent.network.select('actor')

        for trajectory in tqdm(trajectories):
            observations = trajectory["observations"]
            if n_interpolations > 0:
                observations = interpolate(observations, n_interpolations=n_interpolations)
            states = compute_states(
                observations, actor, batch_size=batch_size, max_obs_per_trajectory=max_obs_per_trajectory
            )

            data[param_path].append(states)

    return data


def get_sorted_param_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    param_paths = folder.glob('params_*.pkl')
    param_paths = sorted(param_paths, key=lambda p: int(p.stem.split('_')[1]))

    return param_paths


def compute_states(
    observations: np.ndarray,
    actor: 'Actor',
    batch_size: Optional[int] = None,
    max_obs_per_trajectory: Optional[int] = None,
) -> List[np.ndarray]:
    states = []
    goal = observations[-1]
    observations = observations[:-1]
    if max_obs_per_trajectory:
        observations = observations[-max_obs_per_trajectory:]
    batch_size = batch_size or len(observations)
    for i in range(0, len(observations), batch_size):
        obs_batch = observations[i : i + batch_size]
        goals = goal[None, ...].repeat(len(obs_batch), axis=0)
        intermediates = actor(observations=obs_batch, goals=goals, temperature=0.0, capture_intermediates=True)[1]
        states.append(get_states(intermediates))
    states = np.concatenate(states, axis=0)

    return [state for state in states]


def compute_transitions(states: List[np.ndarray], n_halving: int = 0) -> Tuple[int, int]:
    n_transitions = 0
    for i in range(0, n_halving):
        states = [halve(state) for state in states]

    for i in range(1, len(states)):
        if not np.array_equal(states[i - 1], states[i]):
            n_transitions += 1

    return n_transitions, len(states) - 1


def compute_test_states(
    folder: str | Path = 'impls/exp/OGBench/Debug/sd000_s_1134105.0.20250716_161902',
    param_path: Optional[Path] = None,
    n_steps: int = 999,
    n_trajectories: int = 1,
    task_id: Optional[int] = None,
    pbar: bool = False,
    return_observations_and_actor: bool = False,
) -> Union[List[Tuple[List[np.ndarray], bool]], Tuple[List[Tuple[List[np.ndarray]]], List[List[np.ndarray]], "Actor"]]:
    folder = Path(folder)
    param_path = param_path or get_sorted_param_paths(folder)[-1]
    agent, (env, train_dataset, val_dataset) = restore_agent_and_env_from_flags(folder / 'flags.json', param_path)

    actor = agent.network.select('actor')
    trajectories_states, trajectories_observations = [], []
    options: Dict[str, Any] = {'render_goal': False}
    if task_id is not None:
        options['task_id'] = task_id

    for _ in tqdm(range(n_trajectories), total=n_trajectories):
        observation, info = env.reset(options=options)
        goal = info['goal']
        observations = []

        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        rng, seed = jax.random.split(rng)
        states, terminated = [], False
        for _ in tqdm(range(n_steps), desc='Computing test states', total=n_steps, disable=not pbar):
            dist, intermediates = actor(observations=observation, goals=goal, temperature=0, capture_intermediates=True)
            state = get_states(intermediates)
            states.append(state[0])

            actions = dist.sample(seed=seed)
            if not agent.config['discrete']:
                actions = jnp.clip(actions, -1, 1)

            action = np.array(actions)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if terminated:
                    terminated = True
                break

            observations.append(observation)
            observation = next_observation

        trajectories_states.append((states, terminated))
        trajectories_observations.append(observations)

    if return_observations_and_actor:
        return trajectories_states, trajectories_observations, agent.network.select('actor')

    return trajectories_states
