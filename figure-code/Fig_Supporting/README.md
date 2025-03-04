## Supporting Figures

### :white_check_mark: Pre-requisites
The scripts in this folder require that you have completed running and generating Figure 6 by [following the steps detailed here](https://github.com/lbl59/TRAILS/tree/main/figure-code/Fig6). This code requires information contained within the CSV files contained in the `critical_periods/` folder found in the [`Fig6/` directory](https://github.com/lbl59/TRAILS/tree/main/figure-code/Fig6).

### :computer: Generating the figure
To generate the factor maps for a chosen solution, run the code in the order shown in the table below. The figures' orientation can then be combined and rearranged using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run | Pre-requisite files | 
| --- | --- | --- | --- | --- |
| NA | `robustness_temporal_diagnostics_functions.py` | Contains the `plot_sd_figures()` function that allows the factor maps to be plotted in `plot_factor_maps.py`. | None | None |
| FigureSI_factormaps.jpg | `plot_factor_maps.py` | Calculates and plots the factor maps for a given solution. | `python ./plot_factor_maps.py` | NA |

### :pushpin: Figure generation outputs
Running `plot_factor_maps.py` will create a a new folder inside your current directory. This folder contains the factor map plots for all the utilities for a given solution at the start, middle and end of the critical robustness conflict periods. 

[Back to main README](https://github.com/lbl59/TRAILS)
