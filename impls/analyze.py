from pathlib import Path
from typing import Optional, Tuple
import json
from agents import agents
from utils.env_utils import make_env_and_datasets
from utils.datasets import GCDataset, HGCDataset, Dataset
import random
import numpy as np
import flax
import pickle


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


def get_trajectory(): ...


def get_state(intermediates: dict) -> Tuple[bool, ...]:
    activations = intermediates['modules_actor']['actor_net']
    dense_layer_names = list(sorted([name for name in activations if 'Dense' in name]))
    neurons = np.array([])
    for name in dense_layer_names:
        neurons = np.concatenate((neurons, activations[name]['__call__'][0][0]))
    state = tuple((neurons > 0).tolist())

    return state


if __name__ == '__main__':
    folder = Path('impls/exp/OGBench/Debug/sd000_20250701_125114')
    # Example usage
    agent, (env, train_dataset, val_dataset) = restore_agent_and_env_from_flags(
        folder / 'flags.json', folder / 'params_200.pkl'
    )

    states = []

    train_dataset.terminal_locs

    data_point = train_dataset.sample(1)
    observations = data_point['observations']
    actor = agent.network.select('actor')
    dist = actor(observations=observations, goals=observations, temperature=0.0, capture_intermediates=True)
    intermediates = dist[1]['intermediates']
    state = get_state(intermediates)
    states.append(state)
