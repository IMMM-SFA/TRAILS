## Re-evaluating a selected pathway policy.

This directory contains the input files and scripts to perform the DU Re-Evaluation of a selected pathway policy (referred to as a Solution from hereon out). In this example we will be selecting **Solution 132**, which corresponds to the Regionally Robust (RR) pathway policy shown in the paper. This subdirectory provides the scripts necessary to replicate the DU Re-Evaluation of Solution 132 under a subset of hydroclimatic realizations and DU factors.

:exclamation: **IMPORTANT** :exclamation: Using a Linux interface and a HPC resource is highly recommended for completing the following steps. It is also highly recommended to familiarize yourself with the scripts provided prior to attempting the following steps.

### :open_file_folder: Folder contents

| Subfolder | Description |
| --- | --- |
| `logs/` | Will store all the `.out` and `.err` files that store any output or error messages generated while running your scripts. |
| `output/` | Will store all the CSV output files resulting from DU Re-Evaluation. Find a sample of the CSV files at the [MSDLive data repository here](10.57931/2524573). |
| `rof_tables_reeval/` | Will store all the ROF tables generated for DU Re-Evaluation. |
| `src/` | Contains all the source code needed to compile and run WaterPaths. |
| `TestFiles/` | Contains all the input files required for WaterPaths to run. |
| `updated_RDM_inflows_demands/` | Contains subfolders (`RDM_i/`) representing independent deeply uncertain states of the world (DU SOWs) that each store their associated hydroclimatic realizations. This folder only contains 10 `RDM` subfolders representing a subset of 10 DU SOWs out of the 1,000 DU SOWs used to run the full experiment.  |
| `RDM_i/final_synthetic_inflows/` | Contains the inflow timeseries of all water sources modeled in WaterPaths. |
| `RDM_i/final_synthetic_inflows/evaporation/` | Contains the evaporation timeseries of all reservoirs modeled in WaterPaths. |
| `RDM_i/synthetic_demands_pwl/` | Contains the demand timeseries of all utilities modeled in WaterPaths. |

### :walking: Steps 

1. **Modifying and compiling WaterPaths**

    You will first need to update line 2219 in `src/Problem/Triangle.cpp` such that `rdm_tseries_dir` is set to the correct directory in which your `updated_RDM_inflows_demands` folder is stored. In your command line, run `make gcc` to compile WaterPaths.  You will know that this step has been completed once you see a `triangleSimulation` file appear in this folder.

2. **Generating the ROF tables**

    To perform DU Re-Evaluation, you will first need to generate the ROF tables corresponding to each hydroclimatic realization. To do this, run the code in the order in which they appear in the table below.
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `make_rof_rdm_dir.sh` | Creates the subfolders that will store each DU SOW's ROF tables in `rof_tables_reeval/` folder | `sbatch make_rof_rdm_dir.sh` |
    | `make_rof_tables.sh` | Submits the `rof_table_gen_script.py` Python script to your computing resource's queue. This step can take up to 24 hours to complete (depending on your architecture), so regularly check your job queue by typing `squeue` into your command line. | `sbatch make_rof_tables.sh` |

    Once the ROF tables have been generated, check that they are stored in their respective `RDM_i` subfolders in `rof_tables_reeval/`.

3. **Running DU Re-Evaluation**

    Once you have successfully generated your ROF tables, you are ready to perform DU Re-Evaluation. Run the code shown in the table below to submit a re-evaluation job to the queue. 
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `du_reeval_submission.sh` | Submits the `du_reeval_script.py` Python script to your computing resource's queue. This step can take up to 48 hours to complete (depending on your architecture), so regularly check your job queue by typing `squeue` into your command line. | `sbatch du_reeval_script.sh` |

    Once this step has been completed, you will find that an `output/` folder has been created in the  containing information on your utilities (`Utilities_*.csv`), drought mitigation policies (`Policies_*.csv`) and water source expansion (`WaterSource_*.csv`) in the `output/1/pwl/` folder. These files are very large - and that's normal! You will also find the performance objectives `.csv` and pathways `.out` files in this new folder.

4. **Organizing your performance objectives' pathways files**

    For ease of use further into our replication process, you will want to create a folder called `solXX__objs_pathways/` in the `output/`  folder that has just been created. Replace XX with the number of the solution that you have selected. Then, move all your performance objectives `.csv` and pathways `.out` files into the `solXX__objs_pathways/` folder.

5. **Converting your CSV files to HDF5**

    We will now compress you CSV files into HDF5 format to streamline the data read/write process. To do this, first open the `convert_to_hdf.py` file and modify lines 13 and 14 to more accurately reflect where you have stored your CSV files, and where you would like to store your HDF5 files. Then run the code shown in the table below. 
    | Script Name | Description | How to Run |
    | --- | --- | --- |
    | `convert_to_hdf.sh` | Submits the `convert_to_hdf.py` Python script to your computing resource's queue. This step can take up to 2 hours to complete (depending on your architecture), so regularly check your job queue by typing `squeue` into your command line. | `sbatch convert_to_hdf.sh` |
    
    You should be able to locate all your HDF5 in your selected folder once this step is completed. Move your HDF5 folder to the main `Phase1/` directory for easier access in Step 2.

[Back to main README](https://github.com/lbl59/TRAILS)