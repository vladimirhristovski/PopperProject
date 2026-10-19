#!/bin/bash
#SBATCH --job-name=popper
#SBATCH --gres=gpu:1
#SBATCH --time=HH:MM:SS
#SBATCH --output=popper_%j.log

singularity run --nv popper_upgrade.sif "$@"
