## Calculating time-varying performance objectives.

This directory contains the input files and scripts to calculate the time-varying performance objectives (reliability, restrictrion frequency, peak financial cost, worst-case cost and unit cost) of **Solution 140**. This subdirectory provides the scripts necessary to replicate the the calculation of Solution 140's metrics under a subset of hydroclimatic realizations and DU SOWs.

:exclamation: **IMPORTANT** :exclamation: It is highly recommended to familiarize yourself with the scripts provided prior to attempting the following steps.

### :open_file_folder: Folder contents

| Subfolder | Description |
| --- | --- |
| `tv_objs/` | Will store all the performance objectives for each of your utilities once the code is run. |
| `output/` | Will store CSV files that containing satisficing timeseries for each utility. |

### :walking: Steps 

1. **Ensure correct filepaths to the HDF5 files**

    You will first need to ensure that all the variables associated with `hdf5_file` in the `obj_temporal_diagnostics_functions_test.py` file corresponds to the correct file path such that the code will be able to locate the HDF5 files generated in Phase 1.

2. **Identifying the utility of interest**

    In line 281 of the `obj_temporal_diagnostics_functions_test.py` file, associate the  `util_num` variable with the index of your utility of interest.

3. **Calculating time-varying performance objectives**

    Run the code shown in the table below to calculate the performance objective for each utility and for the region.
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `calc_temporal_objs_mpi.sh` | Submits the `obj_temporal_diagnostics_functions_test.py` Python script calculate the time-varying performance objectives for your utility of choice. This step can take up to 24 hours to complete (I know :skull:), so regularly check your job queue by typing `squeue` into your command line. | `sbatch calc_temporal_objs_mpi.sh` |

    Once this step has been completed, you will find the your selected utility's performance objectives in the `tv_objs/` folder.

4. **Rinse and repeat for all utilities**

    Repeat Steps 1-3 for all other utilities.

5. **Calculate time-varying robustness**

    Run the code shown in the table below to calculate the robustness of each utility and that of the region.
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `calc_satisficing.py` | Determines if each utility and the region meets their robustness satisficing criteria and calculates their consequent robustness. | `python calc_satisficing.py` |

    Once this step has been completed, you will find your robustness and satisficing files in the `output/` folder.
