#!/bin/bash
echo "Submitting job array with job indices $1. Slurm logs written to ./mpcdf/slurm_logs/ogbench_array.{out,err}.%j"
sbatch <<EOT
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./raven/slurm_logs/ogbench_array.out.%j
#SBATCH -e ./raven/slurm_logs/ogbench_array.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ogbench_array
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#SBATCH --array=$1
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=nmilosevic@cbs.mpg.de
#SBATCH --time=24:00:00

source /etc/profile.d/modules.sh
module purge
module load apptainer
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

srun ${@:2} --task_idx \${SLURM_ARRAY_TASK_ID}

exit 0
EOT
