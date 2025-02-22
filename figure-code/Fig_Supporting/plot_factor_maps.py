import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os 

from robustness_temporal_diagnostics_functions import boosted_trees_t, \
    find_num_important_factors, find_sows, plot_feature_bars, plot_sd_figures

sns.set_style('white')

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_RDM = 1000  # number of DU SOWs

sol_selected = 0   # CHANGE THIS: 0 - DF solution, 1 - RF solution, 2 - RR solution

all_util_nums = np.arange(0,6)
sol_nums = [92, 132, 140]   # to be changed to solutions of your interest

# Each solutions start and end of their robustness critical period
# These are example start and end periods
# Replace with the actual start and end periods relevant to your analysis
start_t_arr = [429, 215, 245]
end_t_arr = [1221, 487, 715]


# regional critical period
print('Selecting solution and its regional critical period...')
sol_num = sol_nums[sol_selected]
start_t = start_t_arr[sol_selected]
end_t = end_t_arr[sol_selected]
mid_t = int(start_t + (end_t - start_t)//2)

times = [start_t, mid_t, end_t]

util_list = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']

rdm_headers = ['Demand (first 15)', 'Demand (mid 15)', 'Demand (last 15)', 'Demand (overall)', 'Bond term', 
    'Bond interest', 'Discount rate', 'Restriction efficiency', 'Evaporation intensity', 'Permitting time', 
    'Construction time', 'Inflow variation', 'Inflow return period', 'Inflow seasonality']

colors = ['#E5B137', '#BC6A2C', '#78370E', '#EFA177', '#B5F7E6', 
          '#7DAF9C', '#23967F','#B088CA', '#633570', '#ABBDC3', 
          '#718792', '#84CAE8', '#43A2E5', '#3842A9']

window_size = 52  # one year rolling window
robustness = np.zeros([1, NUM_WEEKS - window_size + 1], dtype=float)
du_factors = pd.read_csv('../../scripts/Phase1/RDM_inputs_final_combined_headers.csv', index_col=None)
du_factor_values = du_factors.values[:NUM_RDM, :]
du_factor_names = du_factors.columns.values.tolist()

print('Begin plotting figure for Solution ', sol_selected)

fig = plt.figure(figsize=(12, 20))
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1])

# adjust spaceing between the plots
plt.subplots_adjust(wspace=0.2, hspace=0.8)

# Create subplots using the GridSpec
axs = []
for row in range(len(util_list)):
    axs_sub = []
    for col in range(len(times)):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f'Row {row + 1}, Col {col + 1}')
        axs_sub.append(ax)
    axs.append(axs_sub)

# adjust the space between the plots
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# dataframe of important SOW values 
important_sow_values_df = pd.DataFrame()

# begin plotting each facto map
for util_num in range(len(util_list)):
    satisficing_filename = f'../../scripts/Phase2/output/satisficing_sol{sol_num}_util{util_num}.csv'
    satisficing_np = np.loadtxt(satisficing_filename, delimiter=',').astype(int)
    robustness_np = np.sum(satisficing_np, axis=0)/NUM_RDM

    sow_idx_pos = np.zeros(len(times)+1, dtype=int)
    sow_idx_neg = np.zeros(len(times)+1, dtype=int)
    sow_idx_pos[0] = 1
    sow_idx_neg[0] = -1

    # plot the sd plots 
    for t in range(len(times)):
        time_index = times[t]
        # if robustness is not 0 or 1, can perform boosted trees to find important factors
        if robustness_np[time_index] < 1.0 and robustness_np[time_index] > 0.0:

            # perform boosted trees
            clf, clf_2factors, factor_influence_sorted, top2factor_values, feature_importances,\
                        shap_values_t, shap_pos_idx, shap_neg_idx = boosted_trees_t(satisficing_np, du_factor_values, rdm_headers, time_index)
            
            # find important DU SOWs
            sow_pos_idx_t, sow_neg_idx_t = find_sows(shap_values_t, du_factor_values, time_index)

            plot_sd_figures(factor_influence_sorted, top2factor_values, clf_2factors, 
                            feature_importances, sow_pos_idx_t, sow_neg_idx_t, 
                            du_factor_values, du_factor_names, satisficing_np, time_index, axs[util_num][t])
        
# make the shap_bar_figures directory if it doesn't exist
if not os.path.exists('factor_maps'):
    os.makedirs('factor_maps')

# set the title of the figure
axs[0][0].set_title('Start of critical period', fontsize=10)
axs[0][1].set_title('Middle of critical period', fontsize=10)
axs[0][2].set_title('End of critical period', fontsize=10)

# label each utility
axs[0][0].text(-0.25, 0.5, "OWASA", transform=axs[0][0].transAxes, verticalalignment='center', fontsize=10)
axs[1][0].text(-0.25, 0.5, "Durham", transform=axs[1][0].transAxes, verticalalignment='center', fontsize=10)
axs[2][0].text(-0.25, 0.5, "Cary", transform=axs[2][0].transAxes, verticalalignment='center', fontsize=10)
axs[3][0].text(-0.25, 0.5, "Raleigh", transform=axs[3][0].transAxes, verticalalignment='center', fontsize=10)
axs[4][0].text(-0.25, 0.5, "Chatham", transform=axs[4][0].transAxes, verticalalignment='center', fontsize=10)
axs[5][0].text(-0.25, 0.5, "Pittsboro", transform=axs[5][0].transAxes, verticalalignment='center', fontsize=10)

plt.savefig(f'factor_maps/factor_map_s{sol_num}.jpg', dpi=300, bbox_inches='tight')

