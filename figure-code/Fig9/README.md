## Figure 9

### :white_check_mark: Pre-requisites
There are three pre-requisite folders for generating Figure 9. 
1. The folder containing all the HDF files in your [Phase 1 directory](https://github.com/lbl59/TRAILS/tree/main/scripts/Phase1).
2. `../Fig6/critical_periods/` stores the CSV files containing information on the start and end of the critical robustness conflict periods for each solution of your choice.
3. `../Fig7-8/important_SOWs/` stores the CSV files containing information on the key SOWs that have characteristics defining success (positive, `p`) or failure (negative `n`).

### :computer: Generating the figure
To generate each of the columns in Figure 9, run the code in the order shown in the table below. The figures can then be combined or rearranged using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run | Pre-requisite files | 
| --- | --- | --- | --- | --- |
| NA | `itsa_functions.py` | Contains the functions that enables the ITSA indices to be calculated for a given SOW in `rof_exceedances_analysis_oneutil.py`. | None | None |
| NA | `rof_exceedances_analysis_oneutil.py` | Calculates and plots the ITSA plots for each utility at the start, middle and end of each critical period. | NA | CSV files `../Fig6/critical_periods/` and `../Fig7-8/important_SOWs/`. |
| Figure 9 | `plot_itsa_contributions.sh` | Submits the `rof_exceedances_analysis_oneutil.py`. | `sbatch ./plot_itsa_contributions.sh` | NA |

### :pushpin: Figure generation outputs
If `rof_exceedances_analysis_oneutil.py` was run correctly when submitting it using `plot_itsa_contributions.sh`, a folder called `rof_exceedances/` should be created in your current directory. This folder stores the ITSA plots for each utility for your selected solution.