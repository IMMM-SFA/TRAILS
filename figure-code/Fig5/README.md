## Figure 5

### :white_check_mark: Pre-requisites
The code scripts in this directory requires the CSV satisficing files in `scripts/Phase2/output`. Please navigate to the [Phase 2 page](https://github.com/lbl59/TRAILS/tree/main/scripts/Phase2) to complete all the steps listed there before attempting this section.

### :computer: Generating the figure
To generate Figures 5(a and b), (c and d) separately, run the code in the order shown in the table below. The figures can then be combined using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run | Pre-requisite files | 
| --- | --- | --- | --- | --- |
| 5(a and b) | `calc_plot_robustness_change.py` | Plots the robustness timeseries for each utility and the region as a whole. | `python ./calc_plot_robustness_change.py` | The satisficing CSV files in `scripts/Phase2/output/` |
| 5(c and d) | `plot_contributing_objs.py` | Plots the performance objectives and the corresponding utilities that are driving robustness change. | `python ./plot_contributing_objs.py` | The driving utility's CSV file in `scripts/Phase2/output/` |

[Back to main README](https://github.com/lbl59/TRAILS)
