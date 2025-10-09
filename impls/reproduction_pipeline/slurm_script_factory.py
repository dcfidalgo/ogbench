import argparse
from datetime import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


TEMPLATE = """#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_out_err/%x.%j.job.out
#SBATCH -e ./job_out_err/%x.%j.job.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J {job_name}
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#
# --- default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=12:00:00

module purge
module load apptainer/1.4.1

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

export WANDB_MODE={wandb_mode}
export MUJOCO_GL=osmesa


mkdir -p {save_dir}

cd {code_dir}
srun -o {save_dir}/job.out -e {save_dir}/job.err \\
    apptainer exec -B .:$HOME -B {save_dir} -B {data_dir} {container_path} \\
        {command}
"""

TEMPLATE2 = """#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_out_err/%x.%j.job.out
#SBATCH -e ./job_out_err/%x.%j.job.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J {job_name}
#
#SBATCH --ntasks=2
#SBATCH --constraint="apu"
#
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=110000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=12:00:00

module purge
module load apptainer/1.4.1

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

export WANDB_MODE=offline
export MUJOCO_GL=osmesa


SAVE_DIR1={save_dir1}
SAVE_DIR2={save_dir2}

mkdir -p $SAVE_DIR1
mkdir -p $SAVE_DIR2

cd {code_dir}
srun -n 1 -o $SAVE_DIR1/job.out -e $SAVE_DIR1/job.err \\
    apptainer exec -B .:$HOME -B $SAVE_DIR1 -B {data_dir} {container_path} \\
        {command1} \\
    &
PID=$!

srun -n 1 -o $SAVE_DIR2/job.out -e $SAVE_DIR2/job.err \\
    apptainer exec -B .:$HOME -B $SAVE_DIR2 -B {data_dir} {container_path} \\
        {command2}

wait $PID
"""


def extract_commands_from_hyperparameters_script(path: Path) -> List[str]:
    """
    Extract all python commands from the hyperparameters.sh script.

    Args:
        path: Path to the hyperparameters.sh file

    Returns:
        List of python command strings
    """
    commands = []

    try:
        with open(path, 'r') as f:
            content = f.read()

        # Find all lines that start with 'python main.py'
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('python main.py'):
                commands.append(line)

    except FileNotFoundError:
        print(f'Error: Could not find hyperparameters.sh at {path}')
        raise
    except Exception as e:
        print(f'Error reading hyperparameters.sh: {e}')
        raise

    return commands


def sort_commands(commands: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Group commands by the environment and agent.

    Args:
        commands: List of python command strings

    Returns:
        Dictionary with structure: {env_name: {agent_name: command}}
    """
    grouped = {}

    for command in commands:
        # Extract environment name using regex
        env_match = re.search(r'--env_name=([^\s]+)', command)
        if not env_match:
            continue
        env_name = env_match.group(1)

        # Extract agent name from the agent path
        agent_match = re.search(r'--agent=agents/([^\.]+)\.py', command)
        if not agent_match:
            continue
        agent_name = agent_match.group(1).upper()  # Convert to uppercase for consistency

        # Initialize nested dictionaries if they don't exist
        if env_name not in grouped:
            grouped[env_name] = {}
        if agent_name not in grouped[env_name]:
            grouped[env_name][agent_name] = command
        else:
            raise ValueError(f'Duplicate command for {env_name} and {agent_name}')

    return grouped


def create_slurm_script(
    command: str,
    env_name: str,
    agent_name: str,
    code_dir: Path,
    save_dir: Path,
    data_dir: Path,
    container_path: Path,
    wandb_mode: str = 'offline',
    env_name2: Optional[str] = None,
    agent_name2: Optional[str] = None,
    save_dir2: Optional[Path] = None,
    command2: Optional[str] = None,
) -> str:
    """
    Create a SLURM script for a specific command.

    Args:
        command: The python command to run
        env_name: Environment name
        agent_name: Agent name
        code_dir: Directory where the main.py script is located
        save_dir: Output directory path
        data_dir: Data directory path
        container_path: Path to the container

    Returns:
        SLURM script content as string
    """
    job_name = f'{env_name}_{agent_name}'

    if env_name2:
        job_name += f'_{env_name2}_{agent_name2}'

        return TEMPLATE2.format(
            job_name=job_name,
            code_dir=code_dir,
            save_dir1=save_dir,
            save_dir2=save_dir2,
            data_dir=data_dir,
            container_path=container_path,
            command1=command,
            command2=command2,
        )

    return TEMPLATE.format(
        job_name=job_name,
        code_dir=code_dir,
        save_dir=save_dir,
        data_dir=data_dir,
        container_path=container_path,
        command=command,
        wandb_mode=wandb_mode,
    )


def main():
    """Main CLI function for SLURM script factory."""
    parser = argparse.ArgumentParser(
        description='Generate SLURM scripts from hyperparameters.sh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and group commands from hyperparameters.sh
  python slurm_script_factory.py --scripts-dir /u/dcfidalgo/ogbench/impl --list

  # Generate SLURM scripts for all commands
  python slurm_script_factory.py --generate-all

  # Generate SLURM scripts for specific environment
  python slurm_script_factory.py --env pointmaze-medium-navigate-v0

  # Generate SLURM scripts for multiple environments
  python slurm_script_factory.py --env pointmaze-medium-navigate-v0 antmaze-large-play-v0

  # Generate SLURM scripts for specific agent
  python slurm_script_factory.py --agent GCBC

  # Generate SLURM scripts for multiple agents
  python slurm_script_factory.py --agent GCBC GCIVL HIQL

  # Generate SLURM scripts for specific environments and agents
  python slurm_script_factory.py --env pointmaze-medium-navigate-v0 antmaze-large-play-v0 --agent GCBC HIQL
        """,
    )

    parser.add_argument(
        '--list', action='store_true', help='List all environments and agents without generating scripts'
    )

    parser.add_argument('--generate-all', action='store_true', help='Generate SLURM scripts for all commands')

    parser.add_argument('--env', type=str, nargs='*', help='Generate scripts only for specific environments (can specify multiple)')

    parser.add_argument('--agent', type=str, nargs='*', help='Generate scripts only for specific agents (GCBC, GCIVL, etc.) (can specify multiple)')

    parser.add_argument('--seed', type=int, default=0, help='Seed value for the training (default: 0)')

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/ptmp/dcfidalgo/ogbench/exp',
        help='Output directory for results (default: /ptmp/dcfidalgo/ogbench/exp)',
    )

    parser.add_argument(
        '--run-group',
        type=str,
        default='baseline',
        help='Name of the run group (default: baseline)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='/ptmp/dcfidalgo/ogbench/data',
        help='Data directory path (default: /ptmp/dcfidalgo/ogbench/data)',
    )

    parser.add_argument(
        '--container',
        type=str,
        default='/u/dcfidalgo/ogbench/containers/viper/container.sif',
        help='Path to the container file (default: /u/dcfidalgo/ogbench/containers/viper/container.sif)',
    )

    parser.add_argument(
        '--code-dir',
        type=str,
        default='/u/dcfidalgo/ogbench/impls',
        help='Directory where hyperparameters.sh and main.py are located (default: /u/dcfidalgo/ogbench/impls)',
    )

    parser.add_argument(
        '--scripts-dir',
        type=str,
        default='/u/dcfidalgo/ogbench/impls/reproduction_pipeline/slurm_scripts',
        help='Directory to save generated SLURM scripts (default: /u/dcfidalgo/ogbench/impls/reproduction_pipeline/slurm_scripts)',
    )
    parser.add_argument('--share-resources', action='store_true', help='Share resources between two jobs in a single SLURM script')
    parser.add_argument('--wandb-mode', type=str, default='offline', help='Set WANDB_MODE (default: offline)')

    args = parser.parse_args()

    code_path = Path(args.code_dir)
    hyperparams_path = code_path / 'hyperparameters.sh'

    if not hyperparams_path.exists():
        print(f'Error: hyperparameters.sh script not found: {hyperparams_path}')
        return

    # Extract commands from hyperparameters script
    print('Extracting commands from hyperparameters script...')
    commands = extract_commands_from_hyperparameters_script(hyperparams_path)
    print(f'Found {len(commands)} commands')

    # Sort commands by environment and agent
    print('Sort commands by environment and agent...')
    sorted_commands = sort_commands(commands)

    if args.list:
        # List all environments and agents
        print(f'\nFound {len(sorted_commands)} environments:')
        for env_name, agents in sorted_commands.items():
            print(f'  {env_name}:')
            for agent_name, cmd in agents.items():
                print(f'    - {agent_name} ({cmd})')
        return

    # Filter commands based on arguments
    filtered_commands = {}
    for env_name, agents in sorted_commands.items():
        if args.env and env_name not in args.env:
            continue

        filtered_agents = {}
        for agent_name, cmds in agents.items():
            if args.agent and agent_name not in [a.upper() for a in args.agent]:
                continue
            filtered_agents[agent_name] = cmds

        if filtered_agents:
            filtered_commands[env_name] = filtered_agents

    if not filtered_commands:
        print('No commands match the specified filters.')
        return

    if args.generate_all or args.env or args.agent:
        # Create output directory for scripts
        scripts_dir = Path(args.scripts_dir)
        scripts_dir.mkdir(exist_ok=True)

        slurm_script_kwargs = []
        for env_name, agents in filtered_commands.items():
            for agent_name, command in agents.items():
                save_dir = build_save_dir(args.output_dir, args.run_group, env_name, agent_name, args.seed) 
                command += f' --save_dir {save_dir}'
                command += f' --dataset_dir {args.data_dir}'
                command += f' --run_group {args.run_group}'
                command += f' --seed {args.seed}'
                command += ' --eval_on_cpu 0'

                # Save slurm script kwargs
                slurm_script_kwargs.append(
                    dict(
                        command=command,
                        env_name=env_name,
                        agent_name=agent_name,
                        code_dir=code_path,
                        save_dir=Path(save_dir),
                        data_dir=Path(args.data_dir),
                        container_path=Path(args.container),
                    )
                )

        script_count = 0
        if args.share_resources:
            if len(slurm_script_kwargs) % 2 != 0:
                slurm_script_kwargs.append(slurm_script_kwargs[-1])  # Duplicate last entry if odd number
            for i in range(0, len(slurm_script_kwargs), 2):
                slurm_content = create_slurm_script(
                    command=slurm_script_kwargs[i]['command'],
                    env_name=slurm_script_kwargs[i]['env_name'],
                    agent_name=slurm_script_kwargs[i]['agent_name'],
                    code_dir=slurm_script_kwargs[i]['code_dir'],
                    save_dir=slurm_script_kwargs[i]['save_dir'],
                    data_dir=slurm_script_kwargs[i]['data_dir'],
                    container_path=slurm_script_kwargs[i]['container_path'],
                    env_name2=slurm_script_kwargs[i+1]['env_name'],
                    agent_name2=slurm_script_kwargs[i+1]['agent_name'],
                    save_dir2=slurm_script_kwargs[i+1]['save_dir'],
                    command2=slurm_script_kwargs[i+1]['command'],
                )

                # Save script to file
                env_name, env_name2 = slurm_script_kwargs[i]['env_name'], slurm_script_kwargs[i+1]['env_name']
                agent_name, agent_name2 = slurm_script_kwargs[i]['agent_name'], slurm_script_kwargs[i+1]['agent_name']
                script_filename = f'2_{env_name}_{agent_name}_{env_name2}_{agent_name2}_seed{args.seed:02d}.slurm'
                script_path = scripts_dir / script_filename
                with open(script_path, 'w') as f:
                    f.write(slurm_content)

                script_count += 1
                print(f'Generated: {script_path}')
        else:
            for params in slurm_script_kwargs:
                slurm_content = create_slurm_script(**params)

                # Save script to file
                script_filename = f'1_{params["env_name"]}_{params["agent_name"]}_seed{args.seed:02d}.slurm'
                script_path = scripts_dir / script_filename
                with open(script_path, 'w') as f:
                    f.write(slurm_content)

                script_count += 1
                print(f'Generated: {script_path}')

        print(f'\nGenerated {script_count} SLURM scripts in {scripts_dir}')
    else:
        print('Use --list to see available environments and agents, or --generate-all to generate scripts.')


def build_exp_name(env_name: str, agent_name: str, seed: int) -> str:
    exp_name = f"{env_name}_{agent_name}_seed{seed:03d}_"
    exp_name += f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    exp_name += '_$SLURM_JOB_ID'

    return exp_name


def build_save_dir(base_dir: str, run_group: str, env_name: str, agent_name: str, seed: int) -> str:
    exp_name = build_exp_name(env_name, agent_name, seed)
    save_dir = os.path.join(base_dir, run_group, exp_name)

    return save_dir


if __name__ == '__main__':
    main()
