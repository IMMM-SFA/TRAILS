## Re-evaluating a selected pathway policy.

This directory contains the input files and scripts to perform the DU Re-Evaluation of a selected pathway policy (referred to as a Solution from hereon out). 
In this example we will be selecting **Solution 140**, which corresponds to the  Regionally Robust (RR) pathway policy shown in the paper. 
This subdirectory provides the scripts necessary to replicate the DU Re-Evaluation of Solution 140 under a subset of hydroclimatic realizations and DU factors.

## What each folder contains
- `rof_tables_reeval`: Will store all the ROF tables generated for DU Re-Evaluation
- `src`: Contains all the source code needed to compile and run WaterPaths
- `TestFiles`: Contains all the input files required for WaterPaths to run
- `updated_RDM_inflows_demands`: Contains subfolders (`RDM_i`) representing independent deeply uncertain states of the world (DU SOWs) that each store their associated hydroclimatic realizations.
    - `RDM_i/final_synthetic_inflows`: Contains the inflow timeseries of all water sources modeled in WaterPaths.
    - `RDM_i/final_synthetic_inflows`: Contains the inflow timeseries of all water sources modeled in WaterPaths.

## Phases 

1. **Modifying and compiling WaterPaths **
    You will first need to update line 2219 in `src/Problem/Triangle.cpp` such that `rdm_tseries_dir` is set to the correct directory in which your `updated_RDM_inflows_demands` folder is stored.
    - `final_synthetic_inflows/`
    - `synthetic_demands_pwl/`


| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the second part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |