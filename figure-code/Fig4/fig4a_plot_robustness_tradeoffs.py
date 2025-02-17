#%%
import numpy as np
import pandas as pd 
from plot_parallel import custom_parallel_coordinates

util_abbrevs = ["O", "D", "C", "R", "Ch", "P"]
util_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']

#%%
# set parallel plot function parameters
fontsize = 14
figsize = (14, 6)

robustness_file = 'robustness_DFSR.csv'

robustness_df = pd.read_csv(robustness_file, index_col=0)
robustness_df_noreg = robustness_df.drop(columns=['Rg'])    
selected_rows = [92, 132, 140]
colors = ['#DC851F', '#48A9A6','#355544']
robustness_filename = f'fig4a_robustness_tradeoffs_allsolns.pdf'
robustness_title = f'Robustness Tradeoffs Across Utilities'

#%%
custom_parallel_coordinates(robustness_df_noreg, columns_axes=util_abbrevs, axis_labels = util_names, 
                                color_by_continuous=2, zorder_by=2, ideal_direction='upwards', zorder_direction='ascending',
                                alpha_base=0.05, lw_base=3, fontsize=fontsize, figsize=figsize,
                                color_palette_continuous = 'gist_yarg',
                                minmaxs=['max']*6, many_solutions = True, many_solutions_idx=selected_rows, many_solutions_color=colors,
                                figtitle=robustness_title,
                                save_fig_filename = robustness_filename)