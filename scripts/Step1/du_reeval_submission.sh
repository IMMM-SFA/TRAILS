#!/bin/bash
#SBATCH -N 1 -n 10 -p normal
#SBATCH --tasks-per-node 10
#SBATCH --job-name=du_reeval
#SBATCH --output=logs/du_reeval.out
#SBATCH --error=logs/du_reeval.err
#SBATCH --time=200:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=10
module load py3-mpi4py
module load py3-numpy

START="$(date +%s)"

mpirun python3 du_reeval_submission.py

DURATION=$[ $(date +%s) - ${START} ]

echo ${DURATION}