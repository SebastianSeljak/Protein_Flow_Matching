#!/bin/bash -l
#$ -cwd
#$ -j y
#$ -o job_logs/train_all.$JOB_ID.log
#$ -l h_rt=24:00:00,h_data=32G
#$ -l gpu,A100=1

module load gcc/11.3.0
module load hdf5/1.14.3

source /u/scratch/s/sseljak/ProteinFold/Protein_Flow_Matching/.venv/bin/activate
cd /u/scratch/s/sseljak/ProteinFold/Protein_Flow_Matching

python train_all.py \
    --h5 proteins_small.h5 \
    --device cuda
