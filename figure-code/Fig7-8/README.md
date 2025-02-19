## Figures 7 and 8

### :white_check_mark: Pre-requisites
Explain pre-requisites here. The consequential_periods, critical_periods, and output_files folders are some of them.

### :computer: Generating the figure
To generate each of the columns in Figures 7 and 8, run the code in the order shown in the table below. The figures can then be combined using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run | Pre-requisite files | 
| --- | --- | --- | --- | --- |
| NA | `robustness_temporal_diagnostics_functions.py` | Contains the functions that enables the SHAP values to be calculated for all SOWs in `plot_shap_values.py`. | None | None |
| NA | `plot_shap_values.py` | Calculates and plots the SHAP bar plots for each individual utility. | `python ./plot_shap_values.py` | Critical period files generated using the code from Figure 6(a and b). |

