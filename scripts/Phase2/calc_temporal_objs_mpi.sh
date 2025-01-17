#!/bin/bash
#SBATCH -N 1 -p normal
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=objs_tempdiagnostics_sol140_util5
#SBATCH --output=logs/objs_tempdiagnostics_sol140_util5.out
#SBATCH --error=logs/objs_tempdiagnostics_sol140_util5.err
#SBATCH --time=500:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=64

source /scratch/lbl59/py_env/bin/activate
mpirun python3 obj_temporal_diagnostics_functions_test.py