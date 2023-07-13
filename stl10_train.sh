#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p gpu_quad
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --array=6-10
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o stl10_train_%A.%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 1 --expname stl10_rn18_RND1_keepclr --cj_prob 0.0 --random_gray_scale 0.0
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 2 --expname stl10_rn18_RND2_keepclr  --cj_prob 0.0 --random_gray_scale 0.0
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 3 --expname stl10_rn18_RND3_keepclr  --cj_prob 0.0 --random_gray_scale 0.0
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 4 --expname stl10_rn18_RND4_keepclr  --cj_prob 0.0 --random_gray_scale 0.0
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 5 --expname stl10_rn18_RND5_keepclr  --cj_prob 0.0 --random_gray_scale 0.0
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 1 --expname stl10_rn18_RND1_clrjit
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 2 --expname stl10_rn18_RND2_clrjit
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 3 --expname stl10_rn18_RND3_clrjit
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 4 --expname stl10_rn18_RND4_clrjit
--max_epochs 100 --num_workers 16 --batch_size 1024 --seed 5 --expname stl10_rn18_RND5_clrjit
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Prototype_Distribution
python3 train/simclr_STL10train_O2.py $unit_name
