import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sns.set_style("white")

utility_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
objs_names = ['REL', 'RF', 'PFC', 'WCC', 'UC']
util_colors = ['#FF5733', '#2E86C1', '#28B463', '#D4AC0D', '#8E44AD', '#C70039']
objs_colors = ['#15616D', '#808C96', '#DDFFD1', '#931F1D', '#F5724A']
sol_num = [92, 132, 140]
sol_names = ['Durham-focused', 'Raleigh-focused', 'Regionally-robust']
robustness_dict = {}
robustness_change_dict = {}

sol_num_selected = 0  # 0: Durham-focused, 1: Raleigh-focused, 2: Regionally-robust
window_size = 52

for i in range(len(sol_num)):
       sol_num_selected = i
       
       # Import the driving actors csv file
       driving_actors_df = pd.read_csv(f'util_driving_robustness_sol{sol_num[sol_num_selected]}_test.csv', header=0, index_col=None)

       # Normalize the objectives then take the rolling mean
       driving_actors_df['REL'] = driving_actors_df['REL']*100
       driving_actors_df['RF'] = driving_actors_df['RF']*100
       driving_actors_df['PFC'] = driving_actors_df['PFC']*100
       driving_actors_df['WCC'] = driving_actors_df['WCC']*100
       driving_actors_df['UC'] = driving_actors_df['UC']*100

       # Take the rolling mean
       driving_actors_df['REL'] = driving_actors_df['REL'].rolling(window_size).mean()
       driving_actors_df['RF'] = driving_actors_df['RF'].rolling(window_size).mean()
       driving_actors_df['PFC'] = driving_actors_df['PFC'].rolling(window_size).mean()
       driving_actors_df['WCC'] = driving_actors_df['WCC'].rolling(window_size).mean()
       driving_actors_df['UC'] = driving_actors_df['UC'].rolling(window_size).mean()

       NUM_WEEKS = driving_actors_df.shape[0]
       driving_actors_df['Weeks'] = np.arange(0, NUM_WEEKS)

       sns.set_style("white")
       fig, ax = plt.subplots(1,1, figsize=(12, 2))
       sol_name = sol_names[sol_num_selected]

       # Plot the layers
       bottom = np.zeros(NUM_WEEKS)
       ax.fill_between(driving_actors_df['Weeks'], bottom, driving_actors_df['REL'], color=objs_colors[0], label='REL', alpha=0.8)
       bottom += driving_actors_df['REL']
       ax.fill_between(driving_actors_df['Weeks'], bottom, bottom + driving_actors_df['RF'], color=objs_colors[1], label='RF', alpha=0.8)
       bottom += driving_actors_df['RF']
       ax.fill_between(driving_actors_df['Weeks'], bottom, bottom + driving_actors_df['PFC'], color=objs_colors[2], label='PFC', alpha=0.8)
       bottom += driving_actors_df['PFC']
       ax.fill_between(driving_actors_df['Weeks'], bottom, bottom + driving_actors_df['WCC'], color=objs_colors[3], label='WCC', alpha=0.8)
       bottom += driving_actors_df['WCC']
       ax.fill_between(driving_actors_df['Weeks'], bottom, bottom + driving_actors_df['UC'], color=objs_colors[4], label='UC', alpha=0.8)

       ax.set_xlabel('Driving actor')
       ax.set_xlim(0, NUM_WEEKS)
       ax.set_xticks(np.arange(0, NUM_WEEKS, 52*5))
       xtick_labels = driving_actors_df['Util'].values
       ax.set_xticklabels(xtick_labels[::52*5])

       # Turn off the top and right borders 
       ax.spines['top'].set_visible(False)
       ax.spines['right'].set_visible(False)

       ax.set_ylim([0, 150])
       ax.set_yticks(np.arange(0, 151, 50))
       ax.set_yticklabels(np.arange(0, 151, 50))

       ax.set_ylabel('Num. SOWs failing')
       ax.set_title(f'Objectives driving failure over time for the {sol_name} solution')

       ax.legend(loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.5), frameon=False)

       fig_name = f'sol{sol_num[sol_num_selected]}_driving_objs.jpg'
       plt.savefig(fig_name, dpi=300, bbox_inches='tight')
