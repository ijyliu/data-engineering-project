#!/bin/bash
#SBATCH --job-name=ToS_Mixedbread_Embedding_Creation
#SBATCH --output=ToS_Mixedbread_Embedding_Creation.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

echo "Starting NER on Company Names"

jupyter nbconvert --to notebook --execute --inplace ToS_Mixedbread_Embedding_Creation.ipynb

echo "Completed NER"
