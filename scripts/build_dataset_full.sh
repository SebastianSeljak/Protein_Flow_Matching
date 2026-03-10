#!/bin/bash -l
#$ -cwd
#$ -j y
#$ -o job_logs/build_dataset_full.$JOB_ID.log
#$ -l h_rt=06:00:00,h_data=16G

module load gcc/11.3.0
module load hdf5/1.14.3

source /u/scratch/s/sseljak/ProteinFold/Protein_Flow_Matching/.venv/bin/activate
cd /u/scratch/s/sseljak/ProteinFold/Protein_Flow_Matching

python preprocessing/build_dataset.py \
    --codes protein_codes.txt \
    --n 10000 \
    --out proteins_full.h5
