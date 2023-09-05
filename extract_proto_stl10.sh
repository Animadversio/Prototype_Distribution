#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p gpu_quad
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --array=1-24
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o extr_proto_%A.%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--expdir ssl_train/stl10_rn18_RND1_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND2_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND3_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND1_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND2_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND3_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND1_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND2_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND3_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND1_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND2_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND3_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND1_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND2_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND3_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND1_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND2_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND3_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND1_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND2_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND3_keepclr --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND1_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND2_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101
--expdir ssl_train/stl10_rn18_RND3_clrjit --suffix _protodist_rnd43 --init_RND 43 --ckpt_beg 50 --ckpt_end 101

--expdir ssl_train/stl10_rn18_RND4_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND5_keepclr --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND4_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
--expdir ssl_train/stl10_rn18_RND5_clrjit --suffix _protodist_rnd42 --init_RND 42 --ckpt_beg 0 --ckpt_end 50
'


export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch_new

cd ~/Github/Prototype_Distribution
python3 proto_extract/proto_extract_batch_O2.py $unit_name
