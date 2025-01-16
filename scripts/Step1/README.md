## Re-evaluating a selected pathway policy.

This directory contains the input files and scripts to perform the DU Re-Evaluation of a selected pathway policy. 
In this example we will be selecting Solution (what we will call a pathway policy for brevity) 140, which corresponds to the 
Regionally Robust (RR) pathway policy shown in the paper. This subdirectory provides the scripts necessary to replicate the 
DU Re-Evaluation of Solution 140 under a subset of hydroclimatic realizations and DU factors.

1. Modify line 2219 in `src/Problem/Triangle.cpp` such that `rdm_tseries_dir` is set to the correctdirectory where the following subdirectories for each RDM (or SOW) are being stored:
    - `final_synthetic_inflows/`
    - `synthetic_demands_pwl/`


| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the second part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |