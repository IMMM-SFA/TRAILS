## Calculating time-varying performance objectives and robustness.

This directory contains the input files and scripts to calculate the time-varying performance objectives (reliability, restrictrion frequency, peak financial cost, worst-case cost and unit cost) of **Solution 140**. This subdirectory provides the scripts necessary to replicate the the calculation of Solution 140's metrics under a subset of hydroclimatic realizations and DU SOWs.

:exclamation: **IMPORTANT** :exclamation: It is highly recommended to familiarize yourself with the scripts provided prior to attempting the following steps.

### :open_file_folder: Folder contents

| Subfolder | Description |
| --- | --- |
| `tv_objs/` | Will store all the performance objectives for each of your utilities once the code is run. |

### :walking: Steps 

1. **Ensure correct filepaths to the HDF5 files**

    You will first need to ensure that all the variables associated with `hdf5_file` in the `obj_temporal_diagnostics_functions_test.py` file corresponds to the correct file path such that the code will be able to locate the HDF5 files generated in Phase 1.

2. **Identifying the utility of interest**

    In line 281 of the `obj_temporal_diagnostics_functions_test.py` file, associate the  `util_num` variable with the index of the utility you would like to generate performance objectives for. 

3. **Calculating time-varying performance objectives**

    Once the necessary changes have been made, run the code shown in the table below to submit a re-evaluation job to the queue. 
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `calc_temporal_objs_mpi.sh` | Submits the `obj_temporal_diagnostics_functions_test.py` Python script to your computing resource's queue. This step can take up to a week complete (I know :skull:), so regularly check your job queue by typing `squeue` into your command line. | `sbatch calc_temporal_objs_mpi.sh` |

    Once this step has been completed, you will find the CSV files each containing your selected utility's performance objectives as it varies across all simulation weeks in your `tv_objs/` folder.