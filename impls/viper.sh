#apptainer exec -B .:"$HOME" --env JAX_PLATFORMS=cpu --env WANDB_MODE=offline --env MUJOCO_GL=osmesa ../viper/container.sif python main.py
apptainer exec -B .:"$HOME" --env WANDB_MODE=offline --env MUJOCO_GL=osmesa ../viper/container.sif python main.py
