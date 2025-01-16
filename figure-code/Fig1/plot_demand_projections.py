import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

#%% import all demand projections
util_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_colors = ['#AF3E41', '#3B8FE8', '#0CC9AA', '#E39D33', '#38891F', '#8352EE']
demand_proj_dir = 'demand_projections/'
demand_files_endname = '_annual_demand_projections_MGD.csv'
demand_proj_dict = {}
for util in util_names:
    demand_proj_dict[util] = np.loadtxt(demand_proj_dir + util + demand_files_endname, delimiter=',')

#%% convert to dataframe
demand_proj_df = pd.DataFrame(demand_proj_dict)

#%% plot all demand projections
fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.2})
for util in ['Cary', 'Durham', 'Raleigh']:
    axs[0].plot(demand_proj_df[util], label=util, color=util_colors[util_names.index(util)], 
                linewidth=3)
axs[0].set_ylabel('Demand (MGD)', fontsize=14, fontdict={'family': 'Arial'})
axs[0].set_xticks([])
#axs[0].legend()
axs[0].spines['bottom'].set_color('black')
axs[0].spines['left'].set_color('black')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].tick_params(axis='both', colors='black')
axs[0].grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
axs[0].set_title('Large Utilities', fontsize=16, fontdict={'family': 'Arial'})

for util in ['OWASA', 'Chatham', 'Pittsboro']:
    axs[1].plot(demand_proj_df[util], label=util, color=util_colors[util_names.index(util)], linewidth=3)
axs[1].set_xlabel('Year', fontsize=14, fontdict={'family': 'Arial'})
axs[1].set_xticks(np.arange(0, 60, 10))
axs[1].set_xticklabels(np.arange(2010, 2070, 10))
axs[1].set_ylabel('Demand (MGD)', fontsize=14, fontdict={'family': 'Arial'})
axs[1].set_yticks(np.arange(0, 20, 10))
axs[1].set_yticklabels(np.arange(0, 20, 10))
#axs[1].legend()
axs[1].spines['bottom'].set_color('black')
axs[1].spines['left'].set_color('black')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].tick_params(axis='both', colors='black')
axs[1].grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
axs[1].set_title('Small-to-Moderate Utilities', fontsize=16, fontdict={'family': 'Arial'})


plt.savefig('demand_projections.pdf', bbox_inches='tight')
plt.show()
# %% plot bar chart of demand projections

