import numpy as np 
import pandas as pd
import os 
from mpi4py import MPI

'''
Change these values to desired number of DU SOWs, realizations, and solution number.
'''
NUM_RDM = 10
NUM_REAL = 10
SOL_NUM = 140

original_folder = f"sol{SOL_NUM}"
hdf_folder = f"sol{SOL_NUM}_hdf_packed"

'''
Tailor this section to suit available resources on your HPC cluster/machine.
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

OMP_NUM_THREADS = 40
N_NODES = 1
N_TASKS_PER_NODE = 10
N_TASKS = int(N_TASKS_PER_NODE * N_NODES) # should be 10
N_RDMS_PER_TASK = int(NUM_RDM/N_TASKS)  # should be 1

# Ensure the output folder exists
os.makedirs(hdf_folder, exist_ok=True)

if rank == 0:
    os.makedirs(hdf_folder, exist_ok=True)

'''
Begin converting CSV files to HDF5 format.
'''
for i in range(N_RDMS_PER_TASK):
    rdm = rank + (N_TASKS * i)
    print("Current rdm: ", rdm)
    utils_file = f'Utilities_s{SOL_NUM}_RDM{rdm}'
    policies_file = f'Policies_s{SOL_NUM}_RDM{rdm}'
    watersources_file = f'WaterSources_s{SOL_NUM}_RDM{rdm}'

    for real in range(0, NUM_REAL):
        utils_file_real = f'{utils_file}_r{real}'
        policies_file_real = f'{policies_file}_r{real}'
        watersources_file_real = f'{watersources_file}_r{real}'

        df_util = pd.read_csv(os.path.join(original_folder, utils_file_real + ".csv"))
        df_util.to_hdf(os.path.join(hdf_folder, utils_file + ".h5"), key=f'r{real}', mode="w")

        df_policies = pd.read_csv(os.path.join(original_folder, policies_file_real + ".csv"))
        df_policies.to_hdf(os.path.join(hdf_folder, policies_file + ".h5"), key=f'r{real}', mode="w")

        df_watersources = pd.read_csv(os.path.join(original_folder, watersources_file_real + ".csv"))
        df_watersources.to_hdf(os.path.join(hdf_folder, watersources_file + ".h5"), key=f'r{real}', mode="w")
    
    print(f"RDM {rdm} converted to hdf5")
comm.barrier()

# Get a list of all CSV files in the input folder
print(f"Converted files in {original_folder} to files in {hdf_folder}")
