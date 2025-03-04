# Lau et al. (2025) DU Pathways TRAILS

**Exploiting multi-objective reinforcement learning and explainable artificial intelligence to navigate robust regional water supply investment pathways.**

Lillian Lau<sup>1\*</sup>, Patrick M. Reed<sup>1</sup>,  and David F. Gold<sup>2</sup>

<sup>1 </sup>Cornell University, Ithaca, NY, USA.

<sup>2 </sup>Utrecht University, Utrecht, Netherlands.

\* corresponding author:  lbl59@cornell.edu

## :mailbox: Contents
- [Abstract](#memo-abstract)
- [Journal Reference](#pencil2-journal-reference)
- [Data and Code Reference](#1234-data-and-code-reference)
- [Contributing Software](#computer-contributing-software)
- [Reproduce my experiment](#file_folder-reproduce-my-experiment)
- [Reproduce my figures](#bar_chart-reproduce-my-figures)

## :memo: Abstract
Urban water utilities are adopting more advanced dynamic and adaptive infrastructure investment frameworks in the face of hydrologic extremes, accelerating demand, and financial constraints. The inclusion of evolutionary multi-objective reinforcement learning (eMORL) has enhanced the identification of high-performing infrastructure investment pathways that balance conflicting objectives and remain robustness amid hydrologic and demand vulnerabilities. However, current evaluations of robustness are based on highly aggregated regional metrics that potentially conceal potential individual robustness conflicts between cooperating utility actors that may emerge over time and largely fail to effectively demonstrate the path-dependent, state-aware nature of these adaptive investment pathways. This study address this nontrivial challenge by contributing the Deeply Uncertain (DU) Pathways Time-varying Regional Assessment of Infrastructure Pathways for the Long- and Short-term (TRAILS) framework. We apply the TRAILS framework on the North Carolina Research Triangle, a challenging six-utility cooperative regional system confronting \$1 billion in in investments to support the maintenance and expansion of its water infrastructure by 2060. The TRAILS pathways diagnostic framework reveals individual robustness preferences have the potential for fundamentally changing consequential dynamics and deeply uncertain drivers of the system. We also discover critical periods of robustness conflict between the individual cooperating actors' infrastructure pathways.  We apply explainable artificial intelligence (xAI) methods to reveal the key DU factors that drive changes in robustness during these critical conflict periods. We utilize Information Theoretic sensitivity analyses to clarify the most consequential state information-action feedbacks that shape individual and regional costs, reliability, and robustness. Overall, the analytics facilitated by the DU Pathways TRAILS framework elucidate how financially significant long-term investments and short-term operational actions shape individual utilities and overall regional robustness. 

[Back to contents](#mailbox-contents)

## :pencil2: Journal reference
To cite this paper, please use the following citation _(Note: This work is currently in-prep and does not yet have a formal citation)_

> Lau, L.B, Reed, P.M., and Gold, D.F. (2025). Exploiting multi-objective reinforcement learning and explainable artificial intelligence to navigate robust regional water supply investment pathways. _In prep_.

[Back to contents](#mailbox-contents)

## :1234: Data and Code Reference

### Input data
Detailed information on generating the hydroclimatic realizations used in this experiment can be found in the GitHub repository linked to the following paper:

> Gold, D.G., Reed, P.M., Gorelick, D.E., and Characklis, G.W. (2023). Advancing Regional Water Supply Management and Infrastructure Investment Pathways That Are Equitable, Robust, Adaptive, and Cooperatively Stable. Water Resources Research. doi.org/10.1029/2022WR033671. [GitHub Repository here](https://github.com/davidfgold/DUPathwaysERAS)

### Output data
A subset of the output data containing the values needed to calculate time-varying performance, robustness, and generate infrastructure pathways can be found at this [MSDLive data repository](10.57931/2524573). 
To cite the data, use the citation below:

> Lau, L.B, Reed, P.M., and Gold, D.F. (2025). TRAILS Output Files. [Data]. https://doi.org/10.57931/2524573.

### Cite the code in this repository
To cite this repository, use the citation below:

> Lau, L. B., Reed, P. M., & Gold, D. F. (2025). Data and code for 'Exploiting multi-objective reinforcement learning and explainable artificial intelligence to navigate robust regional water supply investment pathways.' [Computer software]. https://doi.org/10.57931/252598

[Back to contents](#mailbox-contents)

## :computer: Contributing software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| HDF5 for Python | v3.12.1 | https://github.com/h5py/h5py | NA |
| Seaborn | v0.13.2 | https://github.com/mwaskom/seaborn | 10.21105/joss.03021 |
| SHAP | v0.46.0 | https://shap.readthedocs.io/en/stable/ | https://doi.org/10.1038/s42256-019-0138-9 |
| Scikit-Learn | v1.6.1 | https://scikit-learn.org/stable/ | https://doi.org/10.1038/s42256-019-0138-9 |
| WaterPaths | v1.0 | https://github.com/bernardoct/WaterPaths | 10.1016/j.envsoft.2020.104772 |

[Back to contents](#mailbox-contents)

## :file_folder: Reproduce my experiment
Clone this repository to get access to code scripts used to generate risk of failure (ROF) tables, run DU Re-Evaluation, and reproduce the figures. 
Navigate into each folder (listed below) to refer to their detailed README files that provide step-by-step guidelines on how to navigate and execute their respective scripts.

### What each folder contains 

1. [`scripts`](https://github.com/lbl59/TRAILS/tree/main/scripts): Contains all code and guidelines required to perform a smaller replication of the full experiment. The subfolders in this directory are labeled in the order of which they should be opened and attempted. 
2. [`figure-code`](https://github.com/lbl59/TRAILS/tree/main/figure-code): Contains the code required to generate most of the figures found in `figures`.
3. [`figures`](https://github.com/lbl59/TRAILS/tree/main/figures): Contains all the figures that can be found in the paper.

### Prerequisites
1. Install the software components required to conduct the experiment from [contributing modeling software](#contributing-modeling-software)
2. Download and install the supporting [input data](#input-data) required to conduct the experiment
3. Follow the guidelines detailed in the README files of the `scripts` directory [here](https://github.com/lbl59/TRAILS/tree/main/scripts) to re-create this experiment.

[Back to contents](#mailbox-contents)

## :bar_chart: Reproduce my figures
Use the files found in the `figures` directory to reproduce the figures used in this publication. Follow the guidelines detailed in the README file of the `figure-code` directory.

**Note**: Please complete the Phases listed in the `scripts` folder prior to reproducing the figures.

[Back to contents](#mailbox-contents)
