#!/usr/bin/env python3
"""
OGBench Dataset Downloader

A script to download all environments/datasets mentioned in the hyperparameters.sh script.
Provides options to include/exclude specific environments and configure the download destination.
"""

import argparse
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Set
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants from ogbench.utils
DEFAULT_DATASET_DIR = '~/.ogbench/data'
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'

# Import only if available, otherwise use local implementation
try:
    from ogbench.utils import download_datasets
    HAS_OGBENCH = True
except ImportError:
    HAS_OGBENCH = False


def local_download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Local implementation of download_datasets function."""
    try:
        from tqdm import tqdm
        tqdm_available = True
    except ImportError:
        tqdm_available = False
    
    # Make dataset directory
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download datasets
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f'{dataset_name}.npz')
        dataset_file_names.append(f'{dataset_name}-val.npz')
    
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        if not os.path.exists(dataset_file_path):
            dataset_url = f'{DATASET_URL}/{dataset_file_name}'
            print('Downloading dataset from:', dataset_url)
            
            try:
                response = urllib.request.urlopen(dataset_url)
                tmp_dataset_file_path = f'{dataset_file_path}.tmp'
                
                if tqdm_available:
                    with tqdm.wrapattr(
                        open(tmp_dataset_file_path, 'wb'),
                        'write',
                        miniters=1,
                        desc=dataset_url.split('/')[-1],
                        total=getattr(response, 'length', None),
                    ) as file:
                        for chunk in response:
                            file.write(chunk)
                else:
                    with open(tmp_dataset_file_path, 'wb') as file:
                        for chunk in response:
                            file.write(chunk)
                            
                os.rename(tmp_dataset_file_path, dataset_file_path)
                
            except urllib.error.HTTPError as e:
                print(f"Warning: Could not download {dataset_file_name}: {e}")
                continue
            except Exception as e:
                print(f"Error downloading {dataset_file_name}: {e}")
                continue
        else:
            print(f"Dataset {dataset_file_name} already exists, skipping...")


def extract_env_names_from_hyperparameters(file_path: str) -> Set[str]:
    """
    Extract all unique environment names from the hyperparameters.sh file.
    
    Args:
        file_path: Path to the hyperparameters.sh file
        
    Returns:
        Set of unique environment names
    """
    env_names = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find all --env_name= patterns
        pattern = r'--env_name=([^\s]+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            env_names.add(match)
            
    except FileNotFoundError:
        print(f"Error: Could not find hyperparameters.sh at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading hyperparameters.sh: {e}")
        sys.exit(1)
        
    return env_names


def get_all_ogbench_environments() -> Set[str]:
    """
    Get all OGBench environments mentioned in the hyperparameters.sh file.
    
    Returns:
        Set of all environment names
    """
    # Try to find hyperparameters.sh in common locations
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / "hyperparameters.sh",
        script_dir / "impls" / "hyperparameters.sh",
        script_dir / "hyperparameters.sh",
        Path("impls/hyperparameters.sh"),
        Path("hyperparameters.sh")
    ]
    
    for path in possible_paths:
        if path.exists():
            return extract_env_names_from_hyperparameters(str(path))
    
    # If not found, return the hardcoded list extracted from the file
    print("Warning: hyperparameters.sh not found, using hardcoded environment list.")

    return {
        'pointmaze-medium-navigate-v0', 'pointmaze-large-navigate-v0', 'pointmaze-giant-navigate-v0',
        'pointmaze-teleport-navigate-v0', 'pointmaze-medium-stitch-v0', 'pointmaze-large-stitch-v0',
        'pointmaze-giant-stitch-v0', 'pointmaze-teleport-stitch-v0', 'antmaze-medium-navigate-v0',
        'antmaze-large-navigate-v0', 'antmaze-giant-navigate-v0', 'antmaze-teleport-navigate-v0',
        'antmaze-medium-stitch-v0', 'antmaze-large-stitch-v0', 'antmaze-giant-stitch-v0',
        'antmaze-teleport-stitch-v0', 'antmaze-medium-explore-v0', 'antmaze-large-explore-v0',
        'antmaze-teleport-explore-v0', 'humanoidmaze-medium-navigate-v0', 'humanoidmaze-large-navigate-v0',
        'humanoidmaze-giant-navigate-v0', 'humanoidmaze-medium-stitch-v0', 'humanoidmaze-large-stitch-v0',
        'humanoidmaze-giant-stitch-v0', 'antsoccer-arena-navigate-v0', 'antsoccer-medium-navigate-v0',
        'antsoccer-arena-stitch-v0', 'antsoccer-medium-stitch-v0', 'visual-antmaze-medium-navigate-v0',
        'visual-antmaze-large-navigate-v0', 'visual-antmaze-giant-navigate-v0', 'visual-antmaze-teleport-navigate-v0',
        'visual-antmaze-medium-stitch-v0', 'visual-antmaze-large-stitch-v0', 'visual-antmaze-giant-stitch-v0',
        'visual-antmaze-teleport-stitch-v0', 'visual-antmaze-medium-explore-v0', 'visual-antmaze-large-explore-v0',
        'visual-antmaze-teleport-explore-v0', 'visual-humanoidmaze-medium-navigate-v0',
        'visual-humanoidmaze-large-navigate-v0', 'visual-humanoidmaze-giant-navigate-v0',
        'visual-humanoidmaze-medium-stitch-v0', 'visual-humanoidmaze-large-stitch-v0',
        'visual-humanoidmaze-giant-stitch-v0', 'cube-single-play-v0', 'cube-double-play-v0',
        'cube-triple-play-v0', 'cube-quadruple-play-v0', 'cube-single-noisy-v0', 'cube-double-noisy-v0',
        'cube-triple-noisy-v0', 'cube-quadruple-noisy-v0', 'scene-play-v0', 'scene-noisy-v0',
        'puzzle-3x3-play-v0', 'puzzle-4x4-play-v0', 'puzzle-4x5-play-v0', 'puzzle-4x6-play-v0',
        'puzzle-3x3-noisy-v0', 'puzzle-4x4-noisy-v0', 'puzzle-4x5-noisy-v0', 'puzzle-4x6-noisy-v0',
        'visual-cube-single-play-v0', 'visual-cube-double-play-v0', 'visual-cube-triple-play-v0',
        'visual-cube-quadruple-play-v0', 'visual-cube-single-noisy-v0', 'visual-cube-double-noisy-v0',
        'visual-cube-triple-noisy-v0', 'visual-cube-quadruple-noisy-v0', 'visual-scene-play-v0',
        'visual-scene-noisy-v0', 'visual-puzzle-3x3-play-v0', 'visual-puzzle-4x4-play-v0',
        'visual-puzzle-4x5-play-v0', 'visual-puzzle-4x6-play-v0', 'visual-puzzle-3x3-noisy-v0',
        'visual-puzzle-4x4-noisy-v0', 'visual-puzzle-4x5-noisy-v0', 'visual-puzzle-4x6-noisy-v0',
        'powderworld-easy-play-v0', 'powderworld-medium-play-v0', 'powderworld-hard-play-v0'
    }


def filter_environments(all_envs: Set[str], include=None, exclude=None) -> Set[str]:
    """
    Filter environments based on include/exclude patterns.
    
    Args:
        all_envs: Set of all available environment names
        include: List of patterns to include (supports wildcards)
        exclude: List of patterns to exclude (supports wildcards)
        
    Returns:
        Filtered set of environment names
    """
    from fnmatch import fnmatch
    
    # Start with all environments
    filtered_envs = set(all_envs)
    
    # Apply include filter if specified
    if include:
        included_envs = set()
        for env in all_envs:
            for pattern in include:
                if fnmatch(env, pattern):
                    included_envs.add(env)
                    break
        filtered_envs = included_envs
    
    # Apply exclude filter if specified
    if exclude:
        excluded_envs = set()
        for env in filtered_envs:
            for pattern in exclude:
                if fnmatch(env, pattern):
                    excluded_envs.add(env)
                    break
        filtered_envs = filtered_envs - excluded_envs
    
    return filtered_envs


def main():
    parser = argparse.ArgumentParser(
        description="Download OGBench datasets mentioned in hyperparameters.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_ogbench_datasets.py

  # Download only pointmaze environments
  python download_ogbench_datasets.py --include "pointmaze-*"

  # Download all except visual environments
  python download_ogbench_datasets.py --exclude "visual-*"

  # Download specific environments
  python download_ogbench_datasets.py --include "antmaze-medium-*" "humanoidmaze-*"

  # Custom download directory
  python download_ogbench_datasets.py --dest /path/to/datasets

  # List available environments without downloading
  python download_ogbench_datasets.py --list-only
        """
    )
    
    parser.add_argument(
        '--dest', 
        type=str, 
        default=DEFAULT_DATASET_DIR,
        help=f'Destination directory for datasets (default: {DEFAULT_DATASET_DIR})'
    )
    
    parser.add_argument(
        '--include', 
        nargs='*', 
        help='Include only environments matching these patterns (supports wildcards like "pointmaze-*")'
    )
    
    parser.add_argument(
        '--exclude', 
        nargs='*', 
        help='Exclude environments matching these patterns (supports wildcards like "visual-*")'
    )
    
    parser.add_argument(
        '--list-only', 
        action='store_true',
        help='List available environments without downloading'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    args = parser.parse_args()
    
    # Get all available environments
    print("Extracting environment names from hyperparameters.sh...")
    all_envs = get_all_ogbench_environments()
    
    if args.list_only:
        print(f"\nFound {len(all_envs)} environments:")
        for env in sorted(all_envs):
            print(f"  - {env}")
        return
    
    # Filter environments based on include/exclude patterns
    filtered_envs = filter_environments(all_envs, args.include, args.exclude)
    
    if not filtered_envs:
        print("No environments match the specified filters.")
        return
    
    print(f"\nEnvironments to download ({len(filtered_envs)} total):")
    for env in sorted(filtered_envs):
        print(f"  - {env}")
    
    if args.dry_run:
        print(f"\nDry run - datasets would be downloaded to: {os.path.expanduser(args.dest)}")
        print(f"Download URL base: {DATASET_URL}")
        return
    
    # Confirm download
    response = input(f"\nDownload {len(filtered_envs)} datasets to {os.path.expanduser(args.dest)}? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        return
    
    # Download datasets
    try:
        print(f"\nStarting download to {os.path.expanduser(args.dest)}...")
        dataset_names = list(filtered_envs)
        
        # Use appropriate download function
        if HAS_OGBENCH:
            download_datasets(dataset_names, args.dest)
        else:
            local_download_datasets(dataset_names, args.dest)
            
        print(f"\nSuccessfully downloaded {len(dataset_names)} datasets!")
        
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()