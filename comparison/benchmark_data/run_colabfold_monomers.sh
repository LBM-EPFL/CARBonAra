#!/bin/sh

# environment
source ~/miniconda3/bin/activate
conda activate colabfold_latest

# run colabfold without and with msa
#colabfold_batch --num-recycle 3 --num-models 1 --model-type alphafold2_ptm . alphafold_models_msa
colabfold_batch --msa-mode single_sequence --num-recycle 3 --num-models 1 --model-type alphafold2_ptm . alphafold_models_ss
