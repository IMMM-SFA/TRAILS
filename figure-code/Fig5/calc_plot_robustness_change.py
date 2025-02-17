#%% Import libraries and set constants
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sns.set_style("white")

NUM_WEEKS = 2344
NUM_DU = 1000

#%% Define functions

# Calculates the moving average of the robustness values using a user-specified window size
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

# Calculate the robustness and the change in robustness for a given utility and solution
def calc_robustness(util_num, sol_num):
    filename = f"../../scripts/Phase2/output/satisficing_sol{sol_num}_util{util_num}.csv"
    satisficing_np = np.loadtxt(filename, delimiter=',')

    robustness = np.sum(satisficing_np, axis=0) / NUM_DU

    # smooth the data
    robustness_conv = robustness_moving_avg(robustness, window_size)

    robustness_dx = calc_change_robustness(robustness_conv, window_size)
    return robustness_conv, robustness_dx

# plot robustness change as a heatmap
def plot_robustness_heatmap(robustness_change_df, sol_num_selected):
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 3))
    
    sol_name = sol_names[sol_num_selected]
    s = sol_num[sol_num_selected]

    robustness_change_np = robustness_change_df.to_numpy()
    robustness_change_np = np.where(robustness_change_np > 0.2, 0.2, robustness_change_np)
    robustness_change_np= np.where(robustness_change_np < -0.2, -0.2, robustness_change_np)
    robustness_change_lim = pd.DataFrame(robustness_change_np, columns=robustness_change_df.columns)

    min_robustness_change = np.min(robustness_change_df.values)
    max_robustness_change = np.max(robustness_change_df.values)
    #min_robustness_change_str = f'{min_robustness_change:.2f}'
    #max_robustness_change_str = f'{max_robustness_change:.2f}'

    norm_factor = np.max([np.abs(min_robustness_change), np.abs(max_robustness_change)])
    #robustness_change_norm_df = robustness_change_df / norm_factor
    # Create the heatmap using a colorblind-friendly diverging colormap
    heatmap=sns.heatmap(robustness_change_lim.T, cmap='BrBG', center=0, cbar_kws={'label': 'Change in robustness'})

    heatmap.set_xticks(np.arange(0, NUM_WEEKS-window_size+1, 52*5))  # Set x-ticks every 5 years
    heatmap.set_xticklabels(np.arange(0, 45, 5))  # Set x-tick labels every 5 years

    # Add labels and title
    plt.xlabel('Years')
    heatmap.set_ylabel('Utility')
    heatmap.set_title(f'Change in robustness over time for the {sol_name} solution')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

    # Customize colorbar with yticks
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([np.min(robustness_change_lim.values), 0, np.max(robustness_change_lim.values)])
    colorbar.set_ticklabels([-0.2, 0, 0.2])

    plt.xticks(rotation=0)
    # Save the plot
    fig_name = f'sol{s}_robustness_change_heatmap.jpg'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

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

#%% Main code

utility_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_colors = ['#FF5733', '#2E86C1', '#28B463', '#D4AC0D', '#8E44AD', '#C70039', 'darkgray']
sol_num = [92, 132, 140]
sol_names = ['Durham-focused', 'Raleigh-focused', 'Regionally-robust']
robustness_dict = {}
robustness_change_dict = {}

sol_num_selected = 0  # 0: Durham-focused, 1: Raleigh-focused, 2: Regionally-robust
window_size = 52

#%% Calculate robustness and convert it from a dict into a dataframe
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

#%% Calculate robustness and convert it from a dict into a dataframe
# For each column, convolve the robustness
robustness_df['Regional'] = robustness_df.min(axis=1)
robustness_change_df = pd.DataFrame(robustness_change_dict)
robustness_change_df['Regional'] = calc_change_robustness(robustness_df['Regional'].values, window_size)

# Normalize the robustness_change_df
all_actors = robustness_df.columns

# save to csv
print('Saving robustness values to CSV')
robustness_df.to_csv(f'output_files/robustness_sol{sol_num[sol_num_selected]}.csv')
robustness_change_df.to_csv(f'output_files/robustness_change_sol{sol_num[sol_num_selected]}.csv')

print('Plotting the figure...')

# Import the critical periods df
critical_periods = pd.read_csv(f'critical_periods/periods_sol{sol_num[sol_num_selected]}.csv', 
                               header=0, index_col=None)

# uncomment to plot the robustness timeseries
plot_robustness_heatmap(robustness_change_df, sol_num_selected)
#plot_robustness_timeseries(robustness_df, critical_periods, sol_num_selected)

