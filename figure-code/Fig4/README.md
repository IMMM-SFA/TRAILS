## Figure 4

### :open_file_folder: File contents
This directory contains the code to create Figure 4. See the data file descriptions below.

| File | Description |
| --- | --- |
| `robustness_DFSR.csv` | Contains the robustness of the utilities for each of the 169 solutions identified using MORL. |
| `refset_objs_utils.csv` | Contains the performance objectives for all utilities across each of the 169 solutions identified using MORL. |
| `refset_DVs_headers_abbrev.csv` | Contains the decision variables for all utilities across each of the 169 solutions identified using MORL. |


### :computer: Generating the figure
To generate Figures 4(a), (b) and (c) separately, run the code in the order shown in the table below. The figures can then be combined using a figure editor like Adobe Illustrator.

| Figure| Script Name | Description | How to Run |
| --- | --- | --- | --- |
| NA | `plot_parallel.py` | Contains all the functions used to plot a parallel axis plot. | No implementation. This functions in this `.py` file is used in `fig4a_plot_robustness_tradeoffs.py`. |
| 4(a) | `fig4a_plot_robustness_tradeoffs.py` | Plots the robustness tradeoffs across the six utilities. | `python ./fig4a_plot_robustness_tradeoffs.py` |
| 4(b) | `fig4b_plot_enoki_objs.py` | Plots the change in performance relative to the regionally-robust pathway strategy. | `python ./fig4b_plot_enoki_objs.py` |
| 4(c) | `fig4c_plot_dv_bars.py` | Plots the use of the restriction, transfer, and infrastructure trggers for each utility. | `python ./fig4c_plot_dv_bars.py` |


