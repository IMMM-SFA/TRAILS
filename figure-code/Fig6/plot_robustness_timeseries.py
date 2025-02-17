#%% Import all libraries and set constants
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sns.set_style("white")

NUM_WEEKS = 2344
NUM_DU = 1000

window_size = 52

utility_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_colors = ['#FF5733', '#2E86C1', '#28B463', '#D4AC0D', '#8E44AD', '#C70039', 'darkgray']
sol_num = [92, 132, 140]
sol_names = ['Durham-focused', 'Raleigh-focused', 'Regionally-robust']
robustness_dict = {}
robustness_change_dict = {}

#%% Define functions
def robustness_moving_avg(robustness, window_size):
    robust_conv = np.zeros(robustness.shape[0] - window_size + 1)
    for w in range(0, robustness.shape[0] - window_size + 1):
        robust_conv[w] = np.mean(robustness[w:w+window_size])
    return robust_conv

# Calculate the percent change in avg robustness between two consecutive years
def calc_change_robustness(robustness, window_size):
    change_robustness = np.zeros(NUM_WEEKS - window_size + 1)
    for w in range(0, robustness.shape[0] - window_size):    
        curr_robustness = robustness[w]
        next_robustness = robustness[w+window_size-1]
        change_robustness[w] = next_robustness - curr_robustness

    return change_robustness

def calc_robustness(util_num, sol_num):
    #satisficing_sol140_util3_new.csv
    filename = f"output_files/satisficing_sol{sol_num}_util{util_num}.csv"
    satisficing_np = np.loadtxt(filename, delimiter=',')

    robustness = np.sum(satisficing_np, axis=0) / NUM_DU

    # smooth the data
    robustness_conv = robustness_moving_avg(robustness, window_size)

    robustness_dx = calc_change_robustness(robustness_conv, window_size)
    return robustness_conv, robustness_dx

def plot_robustness_timeseries(robustness_df, critical_periods, sol_num_selected):
    sns.set_style("white")
    fig, ax = plt.subplots(1,1, figsize=(12, 2))
    sol_name = sol_names[sol_num_selected]
    s = sol_num[sol_num_selected]
    all_actors = robustness_df.columns

    start_idx = critical_periods['start'].values
    end_idx = critical_periods['end'].values
    for i in range(len(start_idx)):
        # remove border
        if end_idx[i] - start_idx[i] <= 52:
            continue
        else:
            ax.axvspan(start_idx[i], end_idx[i], color='indianred', alpha=0.2, edgecolor=None)

    for i in range(len(all_actors)):
        util_name = all_actors[i]
        robustness = robustness_df[util_name].values
        time = np.arange(0, len(robustness))
        if i != 6:
            ax.plot(robustness, label=util_name, color=util_colors[i], linewidth=2.5)
        else:
            ax.plot(time[::104], robustness[::104], 'X', label=util_name, color=util_colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Years')
    ax.set_xlim(0, len(robustness))
    ax.set_xticks(np.arange(0, NUM_WEEKS-window_size+1, 52*5))
    ax.set_xticklabels(np.arange(0, 45, 5))

    # turn off the top and right borders 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Robustness')
    ax.set_title(f'Robustness over time for the {sol_name} solution')
    
    ax.legend(loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.5), frameon=False)
    
    fig_name = f'figures/sol{s}_robustness_timeseries_critical.pdf'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

#%%Plot the robustness timeseries

sol_num_selected = 0  # 0: Durham-focused, 1: Raleigh-focused, 2: Regionally-robust

print('Calculating robustness...')
for i in range(len(utility_names)):
    util_num = i
    util_name = utility_names[i]
    s = sol_num[sol_num_selected]
    robustness_i, robustness_change_i = calc_robustness(util_num, s)
    robustness_dict[util_name] = robustness_i
    robustness_change_dict[util_name] = robustness_change_i

# Convert dictionary to DataFrame
robustness_df = pd.DataFrame(robustness_dict)

#for each column, convolve the robustness
robustness_df['Regional'] = robustness_df.min(axis=1)
robustness_change_df = pd.DataFrame(robustness_change_dict)
robustness_change_df['Regional'] = calc_change_robustness(robustness_df['Regional'].values, window_size)

# normalize the robustness_change_df
all_actors = robustness_df.columns

print('Plotting the figure...')
# import the critical periods df
critical_periods = pd.read_csv(f'critical_periods/periods_sol{sol_num[sol_num_selected]}.csv', 
                               header=0, index_col=None)

plot_robustness_timeseries(robustness_df, critical_periods, sol_num_selected)

