import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os 

from robustness_temporal_diagnostics_functions import boosted_trees_t, find_num_important_factors, \
    find_sows, plot_feature_bars, plot_mean_shap_bars, find_threshold_values,\
    plot_threshold_heatmap

sns.set_style('white')

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_RDM = 1000  # number of DU SOWs

all_util_nums = np.arange(0,6)
sol_nums = [92, 132, 140]   # to be changed to solutions of your interest

# Each solutions start and end of their robustness critical period
# These are example start and end periods
# Replace with the actual start and end periods relevant to your analysis
start_t_arr = [429, 215, 245]
end_t_arr = [1221, 487, 715]
sol_selected = 0

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

for util_num in range(len(util_list)):
    satisficing_filename = f'../../scripts/Phase2/output/satisficing_sol{sol_num}_util{util_num}.csv'
    satisficing_np = np.loadtxt(satisficing_filename, delimiter=',').astype(int)
    robustness_np = np.sum(satisficing_np, axis=0)/NUM_RDM

    print('Begin plotting figure for ', util_list[util_num])
    # Create a 3x3 figure
    #fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 5, 5, 1])

    # adjust spaceing between the plots
    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    # Create subplots using the GridSpec
    axs = []
    for row in range(4):
        axs_sub = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f'Row {row + 1}, Col {col + 1}')
            axs_sub.append(ax)
        axs.append(axs_sub)

    # adjust the space between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # dataframe of important SOW values 
    important_sow_values_df = pd.DataFrame()

    sow_idx_pos = np.zeros(len(times)+1, dtype=int)
    sow_idx_neg = np.zeros(len(times)+1, dtype=int)
    sow_idx_pos[0] = 1
    sow_idx_neg[0] = -1

    # plot the sd plots 
    for t in range(len(times)):
        if robustness_np[times[t]] < 1.0 and robustness_np[times[t]] > 0.0:
            # perform boosted trees
            clf, clf_2factors, factor_influence_sorted, top2factor_values, feature_importances,\
                        shap_values_t, shap_pos_idx, shap_neg_idx = boosted_trees_t(satisficing_np, du_factor_values, rdm_headers, times[t])
            
            # save the dataframes
            sow_pos_idx_t, sow_neg_idx_t = find_sows(shap_values_t, du_factor_values, t)

            thres_feature_values_pos, thres_feature_values_neg, max_shap_values, min_shap_values, \
                mean_shap_values_pos_idx_sorted, mean_shap_values_neg_idx_sorted = find_threshold_values(shap_values_t, du_factor_values)
            
            threshold_feature_values_pos = du_factor_values[sow_pos_idx_t, :]
            threshold_feature_values_neg = du_factor_values[sow_neg_idx_t, :]

            plot_mean_shap_bars(shap_values_t, rdm_headers, axs[1][t], axs[2][t], legend=False)

            plot_threshold_heatmap(du_factor_values, threshold_feature_values_pos, threshold_feature_values_neg, 
                                mean_shap_values_pos_idx_sorted.flatten(), mean_shap_values_neg_idx_sorted.flatten(), 
                                axs[0][t], axs[3][t], fig)
            
            if t == 2:
                plot_mean_shap_bars(shap_values_t, rdm_headers, axs[1][t], axs[2][t], legend=True) 
                plot_threshold_heatmap(du_factor_values, threshold_feature_values_pos, threshold_feature_values_neg, 
                                mean_shap_values_pos_idx_sorted.flatten(), mean_shap_values_neg_idx_sorted.flatten(), 
                                axs[0][t], axs[3][t], fig, colorbar=True)
            
        else:
            # remove the top, left and right borders
            axs[1][t].spines['top'].set_visible(False)
            axs[1][t].spines['right'].set_visible(False)
            axs[1][t].spines['left'].set_visible(False)

            axs[2][t].spines['bottom'].set_visible(False)
            axs[2][t].spines['right'].set_visible(False)
            axs[2][t].spines['left'].set_visible(False)
            
            sow_pos_idx_t, sow_neg_idx_t = 0, 0
        
        sow_idx_pos[t+1] = sow_pos_idx_t
        sow_idx_neg[t+1] = sow_neg_idx_t

    important_sow_values_df = important_sow_values_df.append(pd.Series(sow_idx_pos), ignore_index=True)
    important_sow_values_df = important_sow_values_df.append(pd.Series(sow_idx_neg), ignore_index=True)
    
    # make the shap_bar_figures directory if it doesn't exist
    if not os.path.exists('shap_bar_figures'):
        os.makedirs('shap_bar_figures')

    if not os.path.exists('important_SOWs'):
        os.makedirs('important_SOWs')
                    
    # save the dataframes
    important_sow_values_df.to_csv(f'shap_bar_figures/impt_sow_sol{sol_num}_util{util_num}_mean_70_new.csv', index=False, header=['SHAP sign', 't=0', 't=1', 't=2'])
    important_sow_values_df.to_csv(f'important_SOWs/impt_sow_sol{sol_num}_regional_mean_70_single_sow.csv', index=False, header=['SHAP sign', 't=0', 't=1', 't=2'])

    ylabel_pos = 'Increased likelihood\n' + r'of success $\longrightarrow$'
    axs[1][0].set_ylabel(ylabel_pos, rotation=90, size=8)
    ylabel_neg = r'$\longleftarrow$ Increased likelihood' + '\nof failure'
    axs[2][0].set_ylabel(ylabel_neg, rotation=90, size=8)
            
    axs[0][0].set_title('Start of critical period', fontsize=10)
    axs[0][1].set_title('Middle of critical period', fontsize=10)
    axs[0][2].set_title('End of critical period', fontsize=10)

    plt.savefig(f'shap_bar_figures/shap_bars_sol{sol_num}_util{util_num}_mean_70_new.pdf', dpi=300, bbox_inches='tight')

