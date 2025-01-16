import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from SALib.analyze import delta
from sklearn.feature_selection import mutual_info_regression
from matplotlib.colors import LinearSegmentedColormap

sns.set_style('white')

def custom_cmap(base_color, light_color, dark_color, reverse=False):
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [light_color, base_color, dark_color])

    if reverse:
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [dark_color, base_color, light_color])

    return custom_cmap

def calc_cf_diff(cf_arr):
    cf_diff = np.zeros([len(cf_arr),1], dtype=float)
    cf_diff[0] = cf_arr[0]

    for i in range(1, len(cf_arr)):
        cf_diff[i] = cf_arr[i] - cf_arr[i-1]
    
    cf_dff = [0.0 if x < 0.0 else x for x in cf_diff]
    return cf_diff

def find_input_ranges(input_file):
    """
    Finds the bounds of the decision variables or DU factor multipliers.

    Parameters
    ----------
    input_file : numpy matrix
        A numpy matrix that specifies the lower and upper bounds of each decision
        variable or DU factor multiplier.

    Returns
    -------
    bounds : tuple
        The lower and upper bound of a decision variable or DU factor multiplier.

    """
    input_ranges = np.zeros((input_file.shape[1],2), dtype=float)
    for i in range(input_file.shape[1]):
        input_ranges[i,0] = min(input_file[:,i])
        input_ranges[i,1] = max(input_file[:,i])

    return input_ranges

def normalize_inputs(inputs, num_inputs):
    """
    This function normalizes the inputs given a numpy array of input ranges.
    :param inputs: the inputs to normalize
    :param input_ranges: the input ranges to normalize the inputs to

    :return: the normalized inputs
    """
    norm_inputs = np.zeros(np.shape(inputs))
    min_input = np.min(np.min(inputs))
    max_input = np.max(np.max(inputs))

    if np.max(inputs) == 0 or (max_input - min_input) == 0:
        print("Input ranges are all zeros. Returning inputs as is.")
        return inputs
    else:
        for i in range(num_inputs):
            norm_inputs[:, i] = (inputs[:, i] - min_input) / (max_input - min_input)
        return norm_inputs

def get_quantiles(input_arr, num_quantiles):
    quantile_edges = np.quantile(input_arr, np.arange(0, 1.0, 1.0/num_quantiles))
    quantile_edges = np.unique(quantile_edges)
    #print('Quantile edges: ', quantile_edges)
    quantile_bins = np.digitize(input_arr, quantile_edges, right=True)

    quantile_bins -= 1

    return quantile_bins

def get_binned(y, bins = -1):
    if bins > 0:
        return pd.cut(y, bins = bins, labels=False)
    
    else:
        return y
    
def single_entropy(y, bins = -1, log_base=2):
    #y = get_binned(y, bins)
    y = get_quantiles(y, bins)
    ny = len(y)
    summation = 0.0
    values_y = set(y)

    #print('Set of values in y: ', values_y)

    for value_y in values_y:
        py = np.shape(np.where(y == value_y))[1] / ny

        if py > 0.0:
            summation -= py * math.log(py, log_base)
    return summation

def mutual_information(y, x, y_bins = -1, x_bins = -1, log_base = 2):
    #y = get_binned(y, y_bins)
    #x = get_binned(x, x_bins)
    y = get_quantiles(y, y_bins)
    x = get_quantiles(x, x_bins)

    ny = len(y)
    # make sure x and y have the same length
    assert ny == len(x)

    summation = 0.0
    values_x = set(x)
    values_y = set(y)

    for value_y in values_y:
        for value_x in values_x:
            px = np.shape(np.where(x == value_x))[1] / ny
            py = np.shape(np.where(y == value_y))[1] / ny
            pxy = len(np.where(np.isin(np.where(x == value_x)[0], np.where(y == value_y)[0]) == True)[0]) / ny

            if pxy > 0.0:
                summation += pxy * math.log((pxy/(px * py)), log_base)
            
    return summation

def itsa_index_one_timestep(y, x_arr, y_bins = -1, x_bins = -1, log_base = 2):
    itsa_arr = np.zeros(x_arr.shape[1], dtype=float)

    if np.var(y) == 0.0 or np.var(x_arr) == 0:
        print('All values of y or x are the same. Returning zeros for ITSA.')
        return itsa_arr
    
    #print('Iterating through all state variables')
    
    for i in range(x_arr.shape[1]):
        x = x_arr[:,i]
        mi_y_x = mutual_information(y, x, y_bins, x_bins, log_base)
        #mi_y_x = mutual_info_regression(y, x)
        entropy_y = single_entropy(y, y_bins, log_base)
        if entropy_y == 0.0:
            itsa_arr[i] = 0.0
        else:
            itsa_arr[i] = mi_y_x / entropy_y

    return itsa_arr

def itsa_index_alltimesteps(inputs_alltime, outputs_alltime, input_names, output_name, 
                            num_timesteps, util_num, sol_num, du_num, y_bins=-1, x_bins=-1, 
                            log_base=2):

    itsa_alltime = np.zeros((num_timesteps, len(input_names)), dtype=float)
    
    inputs_alltime_norm = np.zeros(np.shape(inputs_alltime))
    #outputs_alltime_norm = np.zeros(np.shape(outputs_alltime))
    #inputs_alltime_mean = np.zeros(np.shape(inputs_alltime))

    # normalize the inputs
    # outputs do not need to be normalized since they are 1's and 0's

    for t in range(num_timesteps):
        inputs_alltime_norm[t,:,:] = normalize_inputs(inputs_alltime[t,:,:], len(input_names))

    for t in range(num_timesteps):
        itsa_t = itsa_index_one_timestep(outputs_alltime[t,:], 
                                        inputs_alltime_norm[t,:,:],
                                        y_bins=y_bins, x_bins=x_bins,
                                        log_base=log_base)
        itsa_alltime[t,:] = itsa_t

    itsa_alltime_df = pd.DataFrame(itsa_alltime, columns=input_names)

    # save the sensitivity indices to a csv file
    itsa_alltime_df.to_csv(f'itsa_results/itsa_s{sol_num}_u{util_num}_{output_name}_du{du_num}.csv',
                           index=False)

    print(f"Saved SI for utility {util_num} and output {output_name} to itsa_results/")
    return itsa_alltime_df

def plot_itsa_figure(util_num, itsa_alltime, input_names, 
                     utility_name, window_size, ax, colors, start_time, end_time):
    
    time_range = np.arange(start_time, end_time+1, 1)
    time_steps = len(time_range)
    x = time_range
    ymin_1, ymax_1 = 0.0, 0.001
    y_base_itsa = np.zeros(time_steps, dtype=float)

    # take moving average of the ITSA indices
    itsa_rolling = itsa_alltime.rolling(window=window_size, min_periods=1).mean()
    itsa_rolling = itsa_rolling.fillna(0)
    itsa_rolling_slice_arr = itsa_rolling.iloc[start_time:end_time+1, :].values
    itsa_rolling_slice = pd.DataFrame(itsa_rolling_slice_arr, columns=input_names)

    for i in range(len(input_names)):
        y_pos = itsa_rolling_slice[input_names[i]].values + y_base_itsa
        
        ax.fill_between(x, y_base_itsa, y_pos, where=y_pos > y_base_itsa, color=colors[i], edgecolor='none')

        y_base_itsa = y_pos
        ymax_1 = max(ymax_1, np.max(y_pos))
    
    # round the value of y_max1 to the nearest integer
    ymax_plot = math.ceil(ymax_1)
    #ax.set_ylim(ymin_1, ymax_plot)
    ax.set_ylim(ymin_1, 4.0)
    '''
    if np.isin(util_num, np.array([3,4,5])):
        #ax.set_xticks(np.arange(0, num_timesteps+52*5, 52*5))
        xtick_nums = np.arange(start_time, end_time+52*5, 52*5)
        ax.set_xticks(xtick_nums)
        print('xtick_nums length: ', len(xtick_nums))
        #ax.set_xticklabels(np.arange(0, 50, 5))
        xtick_labels = np.arange(np.int(np.ceil(start_time/52)), np.int(np.ceil((end_time+(52*5))/52))+5, 5)
        print('xtick_labels length: ', len(xtick_labels))
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Years")
    else:
        ax.set_xticks(np.arange(start_time, end_time+52*5, 52*5))
        ax.set_xticklabels([])
        ax.set_xlabel(" ")
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(utility_name, fontsize=10)
