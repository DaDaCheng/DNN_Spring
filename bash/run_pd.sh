#!/bin/bash
#SBATCH --job-name=net
#SBATCH --time=6:00:00
#SBATCH --partition=a100,rtx8000,pascal  # or titanx
#SBATCH --gres=gpu:1  
#SBATCH --mem=10G
#SBATCH --output=out/output.o%j
#SBATCH --error=out/output.e%j



ml Python/3.10.4-GCCcore-11.3.0;source $HOME/geo_env/bin/activate


"$@"