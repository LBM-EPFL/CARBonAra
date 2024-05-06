#!/bin/sh

# environment
source ~/miniconda3/bin/activate
conda activate colabfold_latest

# run colabfold without and with msa
# NOTE: single sequence sucks: almost no recovery with WT sequence
#colabfold_batch --msa-mode single_sequence --num-recycle 3 --num-models 5 . alphafold_models_ss
# NOTE: alphafold2_multimer_v3 weights sucks: clashes between subunits even after 15 recycles
#colabfold_batch --num-recycle 15 --num-models 1 . alphafold_models_msa
colabfold_batch --num-recycle 5 --num-models 1 --model-type alphafold2_multimer_v2 . alphafold_models_msa
colabfold_batch --msa-mode single_sequence --num-recycle 5 --num-models 1 --model-type alphafold2_multimer_v2 . alphafold_models_ss
