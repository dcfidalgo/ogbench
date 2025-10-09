# OGBench Dataset Downloader

This script downloads all environments/datasets mentioned in the `hyperparameters.sh` script from the OGBench project.

## Features

- **Download all datasets**: Downloads all 85 environments mentioned in the hyperparameters file
- **Selective download**: Include/exclude environments using wildcard patterns
- **Configurable destination**: Specify custom download directory
- **Progress tracking**: Shows download progress with tqdm (if available)
- **Dry run mode**: Preview what would be downloaded without actually downloading
- **Resume capability**: Skip files that already exist

## Usage

### Basic Usage

```bash
# Download all datasets
python download_ogbench_datasets.py

# List available environments without downloading
python download_ogbench_datasets.py --list-only
```

### Filtering Options

```bash
# Download only pointmaze environments
python download_ogbench_datasets.py --include "pointmaze-*"

# Download all except visual environments
python download_ogbench_datasets.py --exclude "visual-*"

# Download multiple patterns
python download_ogbench_datasets.py --include "antmaze-medium-*" "humanoidmaze-*"

# Exclude multiple patterns
python download_ogbench_datasets.py --exclude "visual-*" "powderworld-*"
```

### Configuration Options

```bash
# Custom download directory
python download_ogbench_datasets.py --dest /path/to/datasets

# Dry run to see what would be downloaded
python download_ogbench_datasets.py --include "pointmaze-*" --dry-run
```

## Available Environments

The script automatically extracts environment names from `hyperparameters.sh`. Currently, it finds 85 unique environments including:

- **Navigation tasks**: pointmaze, antmaze, humanoidmaze (various sizes)
- **Soccer environments**: antsoccer 
- **Visual variants**: visual-antmaze, visual-humanoidmaze, etc.
- **Manipulation tasks**: cube, scene, puzzle (various configurations)
- **Powderworld environments**: easy, medium, hard

## Requirements

- Python 3.6+
- No additional dependencies required (tqdm optional for progress bars)
- Internet connection for downloading

## Notes

- Each environment has two files: `{env_name}.npz` and `{env_name}-val.npz`
- Default download location: `~/.ogbench/data`
- Files are downloaded from: `https://rail.eecs.berkeley.edu/datasets/ogbench`
- The script will skip files that already exist
- Failed downloads are logged but don't stop the process