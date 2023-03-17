#!/bin/bash
#SBATCH --time=0-15:38:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=100000
module load Python/3.9.6-GCCcore-11.2.0
source /data/p308104/.envs/TensorF_venv_v0/bin/activate
python 	-u UMCG_V9_LFSeeker_Dynunet_WandB-Script_lf0.py
