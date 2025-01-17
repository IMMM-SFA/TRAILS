#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=convert_to_HDF_packed_sol140
#SBATCH --output=logs/convert_to_HDF_packed_sol140.out
#SBATCH --error=logs/convert_to_HDF_packed_sol140.err
#SBATCH --time=100:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=10
module load py3-mpi4py
module load py3-numpy

mpirun python3 convert_to_hdf.py