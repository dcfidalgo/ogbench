#!/bin/bash
sbatch <<EOT
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./raven/slurm_logs/ogbench.out.%j
#SBATCH -e ./raven/slurm_logs/ogbench.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ogbench
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=nmilosevic@cbs.mpg.de
#SBATCH --time=24:00:00

source /etc/profile.d/modules.sh
module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

# for ogbench scripts
export WANDB_MODE=offline 
export MUJOCO_GL=osmesa

srun $@

exit 0
EOT
