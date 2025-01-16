import numpy as np
import pandas as pd 
from plot_parallel import custom_parallel_coordinates

util_abbrevs = ["O", "D", "C", "R", "Ch", "P", 'Rg']
util_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro', 'Regional']
objs = ['REL', 'RF', 'NPC', 'PFC', 'WCC', 'UC']
dvs = ['RT', 'TT', 'InfT']

objs_allutils = ['O_REL', 'O_RF', 'O_NPC', 'O_PFC', 'O_WCC', 'O_UC',
                 'D_REL', 'D_RF', 'D_NPC', 'D_PFC', 'D_WCC', 'D_UC',
                 'C_REL', 'C_RF', 'C_NPC', 'C_PFC', 'C_WCC', 'C_UC',
                 'R_REL', 'R_RF', 'R_NPC', 'R_PFC', 'R_WCC', 'R_UC',
                 'Ch_REL', 'Ch_RF', 'Ch_NPC', 'Ch_PFC', 'Ch_WCC', 'Ch_UC',
                 'P_REL', 'P_RF', 'P_NPC', 'P_PFC', 'P_WCC', 'P_UC',
                 'Rg_REL', 'Rg_RF', 'Rg_NPC', 'Rg_PFC', 'Rg_WCC', 'Rg_UC']

# set parallel plot function parameters
fontsize = 10
figsize = (14, 6)

robustness_file = 'robustness_DFSR.csv'

robustness_df = pd.read_csv(robustness_file, index_col=0)

selected_rows = [92, 132, 140]
colors = ['#DC851F', '#48A9A6','#355544']
robustness_filename = f'Robustness_tradeoffs_allsolns.pdf'
robustness_title = f'Robustness Tradeoffs Across Utilities'
custom_parallel_coordinates(robustness_df, columns_axes=util_abbrevs, axis_labels = util_names, 
                                color_by_continuous=6, zorder_by=6, ideal_direction='upwards', zorder_direction='ascending',
                                alpha_base=0.4, lw_base=3, fontsize=fontsize, figsize=figsize,
                                color_palette_continuous = 'gist_yarg',
                                minmaxs=['max']*6, many_solutions = True, many_solutions_idx=selected_rows, many_solutions_color=colors,
                                figtitle=robustness_title,
                                save_fig_filename = robustness_filename)