## Calculating time-varying performance objectives.

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

    In line 281 of the `obj_temporal_diagnostics_functions_test.py` file, associate the  `util_num` variable with the index of the 

3. **Running DU Re-Evaluation**

    Once you have successfully generated your ROF tables, you are ready to perform DU Re-Evaluation. Run the code shown in the table below to submit a re-evaluation job to the queue. 
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `du_reeval_submission.sh` | Submits the `du_reeval_script.py` Python script to your computing resource's queue. This step can take up to 48 hours to complete (depending on your architecture), so regularly check your job queue by typing `squeue` into your command line. | `sbatch du_reeval_script.sh` |

    Once this step has been completed, you will find the output containing information on your utilities (`Utilities_*.csv`), drought mitigation policies (`Policies_*.csv`) and water source expansion (`WaterSource_*.csv`) in the `output/1/sol140/` folder (the solution number will change depending on what you have selected) . These files are very large - and that's normal! 

4. **Converting your CSV files to HDF5**

    We will now compress you CSV files into HDF5 format to streamline the data read/write process. To do this, first open the `convert_to_hdf.py` file and modify lines 13 and 14 to more accurately reflect where you have stored your CSV files, and where you would like to store your HDF5 files. Then run the code shown in the table below. 
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `convert_to_hdf.sh` | Submits the `convert_to_hdf.py` Python script to your computing resource's queue. This step can take up to 2 hours to complete (depending on your architecture), so regularly check your job queue by typing `squeue` into your command line. | `sbatch convert_to_hdf.sh` |
    
    You should be able to locate all your HDF5 in your selected folder once this step is completed. Move your HDF5 folder to the main `Step1/` directory for easier access in Step 2.