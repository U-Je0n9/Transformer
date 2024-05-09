#!/bin/bash
#SBATCH -J mlip
#SBATCH --nodelist=ariel-v7
#SBATCH --partition batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o %A-%x.out
#SBATCH -e %A-%x.err
#SBATCH --time=1-0
python3 test.py