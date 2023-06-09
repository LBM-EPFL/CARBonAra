#!/bin/sh -l

# activate anaconda
source $HOME/miniconda3/bin/activate
conda activate colabfold_latest

# quick fold
colabfold_batch --msa-mode single_sequence --num-recycle 3 --num-models 1 --model-type alphafold2_ptm seqs alphafold_models_ss
