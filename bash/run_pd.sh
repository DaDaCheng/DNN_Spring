#!/bin/bash
#SBATCH --job-name=net
#SBATCH --time=0:30:00
#SBATCH --qos=gpu30min
#SBATCH --partition=rtx4090,a100,a100-80g,titan    #a100,pascal,titanx  # or titanx
#SBATCH --gres=gpu:1  
#SBATCH --mem=60G
#SBATCH --output=out/output.o%j
#SBATCH --error=out/output.e%j

#bash
#source ~/.bashrc

echo $SHELL
eval "$(conda shell.bash hook)"
ml Python/3.10.4-GCCcore-11.3.0;ml CUDA/11.8.0; 
echo $CONDA_DEFAULT_ENV
conda deactivate
echo $CONDA_DEFAULT_ENV
conda activate env12
echo $CONDA_DEFAULT_ENV
#conda activate plz
"$@"