# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:49:25 2022

@author: lbl59
"""

from mpi4py import MPI
import numpy as np
import subprocess, sys, time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# change these values depending on the number of realizations and RDMs
N_REALIZATIONS = 10
N_RDMs = 10

# change these values depending on HPC cluster/machine architecture
OMP_NUM_THREADS = 10
N_NODES = 1
N_TASKS_PER_NODE = 10
N_TASKS = int(N_TASKS_PER_NODE * N_NODES) # should be 10
N_RDMS_PER_TASK = int(N_RDMs/N_TASKS)  # should be 1

# change this path depending on current directory
DATA_DIR = "/scratch/lbl59/DUPathwaysERAS/code/reevaluation/"  

# set filename where optimization solutions are stored
SOLS_FILE_NAME = "refset_DVs.csv"   

# change this number to reflect current solution number
SOL_NUM = 132

for i in range(N_RDMS_PER_TASK):
    current_RDM = rank + (N_TASKS * i)

    command_run_rdm = "time ./triangleSimulation -T 32 -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C -1 -E pwl -D pwl -F 1 -O rof_tables_reeval/RDM_${RDM}/ -s TestFiles/refset_DVs.csv -U TestFiles/WJLWTP_rdm_utilities_reeval.csv -W TestFiles/WJLWTP_rdm_watersources_reeval.csv -P TestFiles/WJLWTP_rdm_policies_reeval.csv -m 0 -R ${RDM} -p false".format(OMP_NUM_THREADS, N_REALIZATIONS, DATA_DIR, current_RDM, current_RDM, SOLS_FILE_NAME, SOL_NUM)
    
    print(command_run_rdm)
    os.system(command_run_rdm)

comm.Barrier()