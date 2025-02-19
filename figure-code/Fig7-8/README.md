## Figures 7 and 8

### :white_check_mark: Pre-requisites
The scripts in this folder require that you have completed running and generating Figure 6 by [following the steps detailed here](https://github.com/lbl59/TRAILS/tree/main/figure-code/Fig6). This code requires information contained within the CSV files contained in the `critical_periods/` folder found in the [`Fig6/` directory](https://github.com/lbl59/TRAILS/tree/main/figure-code/Fig6).

### :computer: Generating the figure
To generate each of the columns in Figures 7 and 8, run the code in the order shown in the table below. The figures can then be combined using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run | Pre-requisite files | 
| --- | --- | --- | --- | --- |
| NA | `robustness_temporal_diagnostics_functions.py` | Contains the functions that enables the SHAP values to be calculated for all SOWs in `plot_shap_values.py`. | None | None |
| 7 and 8 | `plot_shap_values.py` | Calculates and plots the SHAP bar plots for each individual utility. | `python ./plot_shap_values.py` | Critical period files generated using the code from Figure 6(a and b). |

### :pushpin: Figure generation outputs
Running `plot_shap_values.py` will create a two new folders inside your current directory:
    - `shap_bar_figures/` contains the SHAP bar plots for all your selected solutions at the start, middle and end of the critical robustness conflict periods. 
    - `important_SOWs/` contains CSV files of the SOWs that contain the key characteristics determining success (positive) or failure (negative). This will be important for generating Figure 9.