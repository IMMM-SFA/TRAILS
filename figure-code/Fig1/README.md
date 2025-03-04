## Figure 1

### :open_file_folder: Folder contents
This directory contains the code to create Figure 1. See the subfolder descriptions below.

| Subfolder | Description |
| --- | --- |
| `ChathamPrimaryServiceArea/` | Contains the files that describe Chatham County's water utility service area boundaries. |
| `demand_projections/` | Contains the annual demand projections for each utility over the next 60 years. |
| `NCDOT_County_Boundaries/` | Contains the formal boundaries for all boundaries in North Carolina. |
| `OWASAPrimaryServiceArea/` | Contains the files that describe OWASA's water utility service area boundaries. |
| `PittsboroPrimaryServiceArea/` | Contains the files that describe TriRiver Water's (Pittsboro's water utility) service area boundaries. |
| `RT_Lakes/` | Contains the formal boundaries for all lakes in the Research Triangle. |
| `RT_Region/` | Contains the formal boundaries of the Research Triangle. |
| `Wake_Lakes/` | Contains the formal boundaries for all lakes in Wake County. |
| `WakeCounty/` | Contains the formal boundaries of Cary and Raleigh. |

### :computer: Generating the figure
To generate Figures 1(a) and (b), run the code in the order shown in the table below.

| Figure| Script Name | Description | How to Run |
| --- | --- | --- | --- |
| 1(a) | `plot_map.py` | Plots the map of the Research Triangle, its water bodies, and water utilities. | `python ./plot_demand_projections.py` |
| 1(b) | `plot_demand_projections.py` | Plots the demand projections of each utility over the next 60 years. | `python ./plot_map.py` |

[Back to main README](https://github.com/lbl59/TRAILS)
