import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from copy import deepcopy
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import shap

sns.set_style('white')

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_RDM = 1000  # number of DU SOWs

def calc_robustness(satisficing):
    robustness = np.sum(satisficing, axis=0)/NUM_RDM
    return robustness

def calc_avg_satisficing(satisficing, t, window_size):
    avg_satisficing = np.mean(satisficing[:, t:t+window_size], axis=1).astype(int)
    return avg_satisficing

def calc_shap_mean(shap_t, sign='pos'):
    shap_mean = np.zeros(shap_t.shape[1], dtype=float)
    shap_vals = np.where(shap_t > 0, shap_t, 0)
    if sign == 'neg':
        shap_vals = np.where(shap_t < 0, shap_t, 0)
    
    for i in range(shap_vals.shape[1]):
        # find the number of nonzero values in each column
        num_nonzero = np.count_nonzero(shap_vals[:, i])
        # calculate the mean of the nonzero values
        shap_mean[i]= np.sum(shap_vals[:, i]) / num_nonzero
    return shap_mean

def calc_shap_median(shap_t, sign='pos'):
    shap_median = np.zeros(shap_t.shape[1], dtype=float)
    shap_vals = np.where(shap_t > 0, shap_t, 0)
    if sign == 'neg':
        shap_vals = np.where(shap_t < 0, shap_t, 0)
    
    for i in range(shap_vals.shape[1]):
        # drop all non-zero values in current column
        shap_vals_nonzero = shap_vals[:, i][shap_vals[:, i] != 0]
        # calculate the median of the nonzero values
        shap_median[i] = np.median(shap_vals_nonzero)

    return shap_median

def calc_shap_maxmin(shap_t, sign='pos'):
    shap_maxmin = np.zeros(shap_t.shape[1], dtype=float)
    shap_vals = np.where(shap_t > 0, shap_t, 0)
    if sign == 'neg':
        shap_vals = np.where(shap_t < 0, shap_t, 0)

    for i in range(shap_vals.shape[1]):
        shap_maxmin[i] = np.max(shap_vals[:, i])
        if sign == 'neg':
            shap_maxmin[i] = np.min(shap_vals[:, i])

    return shap_maxmin

def boosted_trees_t(satisficing, all_params, param_names, t):
    '''
    Performs boosted trees at a specific timestep t for a given utility
    '''
    # fit a boosted tree classifier to rank and extract important factors 
    print('Fitting boosted tree classifier...')
    
    clf = GradientBoostingClassifier(n_estimators=500,
                                            learning_rate=0.1,
                                            max_depth=4)
    
    satisficing_t = satisficing[:, t]
    
    clf.fit(all_params, satisficing_t)

    feature_importances = deepcopy(clf.feature_importances_)
    factor_influence_sorted = np.argsort(feature_importances)[::-1]
    print('Factor influence sorted idx:', factor_influence_sorted)
    
    shap_explainer_t = shap.TreeExplainer(clf, feature_perturbation='tree_path_dependent')

    shap_values_t = shap_explainer_t.shap_values(all_params)

    # find all positive SHAP values
    shap_most_positive = np.max(np.max(shap_values_t, axis=1))
    shap_all_pos = np.where(shap_values_t > 0)
    shap_all_pos_mean = np.mean(shap_values_t[shap_all_pos], axis=0)
    shap_all_pos_mean_sorted = np.argsort(shap_all_pos_mean)[::-1]
    shap_most_pos_factor = shap_all_pos_mean_sorted[0]
    
    shap_most_negative = np.min(np.min(shap_values_t, axis=1))
    shap_all_neg = np.where(shap_values_t < 0)
    shap_all_neg_mean = np.mean(np.abs(shap_values_t[shap_all_neg]), axis=0)
    shap_all_neg_mean_sorted = np.argsort(shap_all_neg_mean)[::-1]
    shap_most_neg_factor = shap_all_neg_mean_sorted[0]

    mean_shap_values_t = np.mean(np.abs(shap_values_t), axis=0)
    mean_shap_values_t_sorted = np.argsort(mean_shap_values_t)[::-1]

    # find highest positive SHAP value for the two most important features and its position
    max_shap_value_pos_idx = np.where(shap_values_t == shap_most_positive)[0][0]

    min_shap_value_neg_idx = np.where(shap_values_t == shap_most_negative)[0][0]

    clf_2factors = GradientBoostingClassifier(n_estimators=500,
                                            learning_rate=0.1,
                                            max_depth=5)
    
    if shap_most_pos_factor == shap_most_neg_factor:
        shap_most_neg_factor_idx = np.where(mean_shap_values_t_sorted != shap_most_neg_factor)[0]
        shap_most_neg_factor = mean_shap_values_t_sorted[shap_most_neg_factor_idx[0]]

    clf_2factors.fit(all_params[:, factor_influence_sorted[:2]], satisficing_t)
    
    top2factor_values = all_params[:, factor_influence_sorted[:2]]

    return clf, clf_2factors, factor_influence_sorted, top2factor_values, feature_importances,\
              shap_values_t, max_shap_value_pos_idx, min_shap_value_neg_idx

def plot_sd_figures(factor_influence_sorted, top2factor_values, clf_2factors, feature_importances, 
                    shap_pos_idx, shap_neg_idx,
                    all_params, param_names, satisficing, t, ax):

    satisficing_t = satisficing[:, t]

    x_data = top2factor_values[:, 0]
    y_data = top2factor_values[:, 1]

    x_min, x_max = (x_data.min(), x_data.max())
    y_min, y_max = (y_data.min(), y_data.max())

    xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                         np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
    dummy_points = list(zip(xx.ravel(), yy.ravel()))

    z = clf_2factors.predict(dummy_points)
    z[z < 0] = 0.
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdBu', alpha=0.35, vmin=0.0, vmax=1.0)
    ax.scatter(top2factor_values[:, 0], top2factor_values[:, 1], 
               c=satisficing_t, cmap='Reds_r', edgecolor='grey', s=80, alpha=0.4)
    
    ax.scatter(all_params[shap_pos_idx, factor_influence_sorted[0]], all_params[shap_pos_idx, factor_influence_sorted[1]], 
               c='k', s=500, marker='*', facecolors='none', linewidth=5)
    ax.scatter(all_params[shap_neg_idx, factor_influence_sorted[0]], all_params[shap_neg_idx, factor_influence_sorted[1]], 
               c='k', s=350, marker='x', linewidth=8, facecolors='none')
    
    ax.set_xlabel(param_names[np.argsort(feature_importances)[::-1][0]], size=16)
    ax.set_ylabel(param_names[np.argsort(feature_importances)[::-1][1]], size=16)
    ax.set_xlim([min(top2factor_values[:,0]), max(top2factor_values[:,0])])
    ax.set_ylim([min(top2factor_values[:,1]), max(top2factor_values[:,1])])
    ax.set_xticks(np.round(np.linspace(min(top2factor_values[:,0]), max(top2factor_values[:,0]), 3), 2))
    ax.set_yticks(np.round(np.linspace(min(top2factor_values[:,1]), max(top2factor_values[:,1]), 3), 2))
    ax.set_xticklabels(np.round(np.linspace(min(top2factor_values[:,0]), max(top2factor_values[:,0]), 3), 2), size=14)
    ax.set_yticklabels(np.round(np.linspace(min(top2factor_values[:,1]), max(top2factor_values[:,1]), 3), 2), size=14)

def find_num_important_factors(mean_shap_values_pos_sorted, mean_shap_values_neg_sorted, epsilon=0.2):
    shap_pos_norm = (mean_shap_values_pos_sorted-np.min(mean_shap_values_pos_sorted)) / (np.max(mean_shap_values_pos_sorted) - np.min(mean_shap_values_pos_sorted))
    num_important_factors_pos = 1
    
    # epsilon calculation
    p = 0.70
    certainty_eps = np.log(p/(1-p))
    #print('Certainty epsilon:', certainty_eps)

    # calculate the rate of change of positive SHAP values
    shap_pos_diff = np.abs(np.diff(shap_pos_norm))
    for i in range(1,len(shap_pos_diff)):
        if mean_shap_values_pos_sorted[i] >= certainty_eps:
            num_important_factors_pos = i

    shap_neg_norm = ((mean_shap_values_neg_sorted-np.min(mean_shap_values_neg_sorted)) / (np.max(mean_shap_values_neg_sorted) - np.min(mean_shap_values_neg_sorted)))
    num_important_factors_neg = 1
    shap_neg_perc = 0

    # calculate the rate of change of negative SHAP values
    shap_neg_diff = np.abs(np.diff(shap_neg_norm))
    for i in range(1,len(shap_neg_diff)):
        if shap_neg_norm[i] != 0:
            shap_neg_perc = abs(shap_neg_diff[i] / shap_neg_norm[i-1])
        if shap_neg_perc > epsilon and mean_shap_values_neg_sorted[i] <= (-1)*certainty_eps:
            num_important_factors_neg = i

    return num_important_factors_pos, num_important_factors_neg

def plot_feature_bars(feature_importances, factor_influence_sorted, ax):
    xtick_labels = ['F' + str(i) for i in factor_influence_sorted]

    ax.bar(xtick_labels, feature_importances[factor_influence_sorted], color='indianred')
    ax.set_xlabel('DU Factors', size=8)
    ax.set_ylabel('Factor\nimportance', rotation=90, size=8)
    ax.set_xticklabels(xtick_labels, size=8)

def find_threshold_values(shap_values_t, all_params):
    # find max SHAP value for each feature
    max_shap_values = np.max(shap_values_t, axis=0)
    max_shap_sow = np.zeros(all_params.shape[1], dtype=float)
    threshold_feature_values_pos = np.zeros(all_params.shape[1], dtype=float)
    for i in range(all_params.shape[1]):
        max_sow = np.where(shap_values_t[:, i] == max_shap_values[i])[0][0]
        print(f'SOW with max SHAP for F{i} is {max_sow}')
        max_shap_sow[i] = max_sow
        threshold_feature_values_pos[i] = all_params[max_sow, i]
        print(f'Value of F{i} for SOW {max_sow} is {all_params[max_sow, i]}')
    
    # find min SHAP value for each feature
    min_shap_values = np.min(shap_values_t, axis=0)
    min_shap_sow = np.zeros(all_params.shape[1], dtype=float)
    threshold_feature_values_neg = np.zeros(all_params.shape[1], dtype=float)
    for i in range(all_params.shape[1]):
        min_sow = np.where(shap_values_t[:, i] == min_shap_values[i])[0][0]
        print(f'SOW with min SHAP for F{i} is {min_sow}')
        min_shap_sow[i] = min_sow
        threshold_feature_values_neg[i] = all_params[min_sow, i]
        print(f'Value of F{i} for SOW {min_sow} is {all_params[min_sow, i]}')

    # find highest positive SHAP value for each feature
    mean_shap_values_pos = calc_shap_mean(shap_values_t, sign='pos')
    mean_shap_values_pos_sorted = np.sort(mean_shap_values_pos)[::-1]
    mean_shap_values_pos_idx_sorted  = np.argsort(mean_shap_values_pos)[::-1]

    # find highest negative SHAP value for each feature
    mean_shap_values_neg = calc_shap_mean(shap_values_t, sign='neg')
    mean_shap_values_neg_sorted = np.sort(mean_shap_values_neg)
    mean_shap_values_neg_idx_sorted = np.argsort(mean_shap_values_neg)
    
    return threshold_feature_values_pos, threshold_feature_values_neg, \
        max_shap_values, min_shap_values, \
        mean_shap_values_pos_idx_sorted, mean_shap_values_neg_idx_sorted

def find_max_min_shap_values(shap_values_t):
    max_pos_shap_values = np.max(np.max(shap_values_t, axis=0))
    max_pos_shap_values_idx = np.where(shap_values_t == max_pos_shap_values)[0][0]

    min_neg_shap_values = np.min(np.min(shap_values_t, axis=0))
    min_neg_shap_values_idx = np.where(shap_values_t == min_neg_shap_values)[0][0]

    return max_pos_shap_values_idx, min_neg_shap_values_idx

def find_sows(shap_values_t, all_params, t):

    # find highest positive SHAP value for each feature
    mean_shap_values_pos = calc_shap_mean(shap_values_t, sign='pos')
    mean_shap_values_pos_sorted = np.sort(mean_shap_values_pos)[::-1]
    mean_shap_values_pos_idx_sorted  = np.argsort(mean_shap_values_pos)[::-1]
    max_shap_value = np.max(np.max(shap_values_t, axis=0))
    max_shap_value_idx = np.where(shap_values_t == max_shap_value)[0][0]

    # find highest negative SHAP value for each feature
    mean_shap_values_neg = calc_shap_mean(shap_values_t, sign='neg')
    mean_shap_values_neg_sorted = np.sort(mean_shap_values_neg)
    mean_shap_values_neg_idx_sorted = np.argsort(mean_shap_values_neg)
    min_shap_value = np.min(np.min(shap_values_t, axis=0))
    min_shap_value_idx = np.where(shap_values_t == min_shap_value)[0][0]

    threshold_feature_values_pos, threshold_feature_values_neg, max_shap_values, min_shap_values, thres_vals_pos, thres_vals_neg  = find_threshold_values(shap_values_t, all_params)

    num_important_factors_pos, num_important_factors_neg = \
        find_num_important_factors(mean_shap_values_pos_sorted, mean_shap_values_neg_sorted, epsilon=0.2)
    
    fv_pos_important = threshold_feature_values_pos[mean_shap_values_pos_idx_sorted[:num_important_factors_pos]]
    fv_neg_important = threshold_feature_values_neg[mean_shap_values_neg_idx_sorted[:num_important_factors_neg]]
    print(f'A SOW that has factors encouraging success has values: {fv_pos_important} in the factors {mean_shap_values_pos_idx_sorted[:num_important_factors_pos]}')
    print(f'A SOW that has factors encouraging failure has values: {fv_neg_important} in the factors {mean_shap_values_neg_idx_sorted[:num_important_factors_neg]}') 

    # find a SOW that matches the character
    # extract columns of the SOWs that have the highest positive/negative SHAP values
    du_factor_pos_cols = all_params[:, mean_shap_values_pos_idx_sorted[:num_important_factors_pos]]
    du_factor_neg_cols = all_params[:, mean_shap_values_neg_idx_sorted[:num_important_factors_neg]]

    # find the row that most closely matches the character
    # minimize least squares erro
    sow_pos_idx = np.argmin(np.sum((du_factor_pos_cols - fv_pos_important)**2, axis=1))
    sow_neg_idx = np.argmin(np.sum((du_factor_neg_cols - fv_neg_important)**2, axis=1))

    print(f'The SOW that most closely matches the character for success at t={t} is SOW {sow_pos_idx} wih values {all_params[sow_pos_idx, mean_shap_values_pos_idx_sorted[:num_important_factors_pos]]}')
    print(f'The SOW that most closely matches the character for failure at t={t} is SOW {sow_neg_idx} with values {all_params[sow_neg_idx, mean_shap_values_neg_idx_sorted[:num_important_factors_neg]]}')
    
    return sow_pos_idx, sow_neg_idx

def plot_threshold_heatmap(all_params, threshold_feature_values_pos, threshold_feature_values_neg,
                           mean_shap_values_pos_idx_sorted, mean_shap_values_neg_idx_sorted, 
                           ax_pos, ax_neg, fig, colorbar=False):
    
    all_params_s = all_params + np.abs(np.min(all_params, axis=0))
    all_params_min = np.min(all_params_s, axis=0)
    all_params_max = np.max(all_params_s, axis=0)
    threshold_feature_values_pos = threshold_feature_values_pos + np.abs(np.min(all_params, axis=0))
    threshold_feature_values_neg = threshold_feature_values_neg + np.abs(np.min(all_params, axis=0))

    norm_threshold_feature_values_pos = (threshold_feature_values_pos - all_params_min) / (all_params_max - all_params_min)

    norm_threshold_feature_values_neg = (threshold_feature_values_neg - all_params_min) / (all_params_max - all_params_min)
    
    f_nums_pos = ['F' + str(i) for i in mean_shap_values_pos_idx_sorted]
    f_nums_neg = ['F' + str(i) for i in mean_shap_values_neg_idx_sorted]

    threshold_pos_df = pd.DataFrame(norm_threshold_feature_values_pos[mean_shap_values_pos_idx_sorted].reshape(1, -1), 
                                    columns=f_nums_pos, index=None)
    threshold_neg_df = pd.DataFrame(norm_threshold_feature_values_neg[mean_shap_values_neg_idx_sorted].reshape(1, -1), 
                                    columns=f_nums_neg, index=None)

    sns.heatmap(threshold_pos_df, cmap='BrBG_r', ax=ax_pos, cbar=False)
    ax_pos.set_xticklabels(f_nums_pos, size=8, rotation=0)
    ax_pos.set_ylabel('')
    ax_pos.set_yticklabels([''])
    
    if colorbar == True:
        cbar_ax = fig.add_axes([0.3, 0.01, 0.4, 0.02])
        h_neg = sns.heatmap(threshold_neg_df, ax=ax_neg, cmap='BrBG_r', 
                            cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})
        
        mappable = h_neg.get_children()[0]
        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Low DU Factor', 'Expected DU Factor', 'High DU Factor'])
        cbar.set_label('DU Factor Values')
        cbar.outline.set_visible(False)
    else: 
        sns.heatmap(threshold_neg_df, cmap='BrBG_r', ax=ax_neg, cbar=False)
    
    ax_neg.set_xticklabels(f_nums_neg, size=8, rotation=0)
    ax_neg.set_ylabel('')
    ax_neg.set_yticklabels([''])
    
    ax_pos.set_title('')
    ax_neg.set_title('')

def plot_mean_shap_bars(shap_values_t, param_names, ax_pos, ax_neg, legend=False):

    # find highest positive SHAP value for each feature
    mean_shap_values_pos = calc_shap_mean(shap_values_t, sign='pos')
    
    # normalize the SHAP values
    mean_shap_values_pos_sorted = np.sort(mean_shap_values_pos)[::-1]
    mean_shap_values_pos_idx_sorted  = np.argsort(mean_shap_values_pos)[::-1]

    # find highest negative SHAP value for each feature
    mean_shap_values_neg = calc_shap_mean(shap_values_t, sign='neg')
    
    # normalize the SHAP values
    mean_shap_values_neg_sorted = np.sort(mean_shap_values_neg)
    mean_shap_values_neg_idx_sorted = np.argsort(mean_shap_values_neg)

    num_important_factors_pos, num_important_factors_neg = \
        find_num_important_factors(mean_shap_values_pos_sorted, mean_shap_values_neg_sorted)
    
    f_nums_pos = ['F' + str(i) for i in mean_shap_values_pos_idx_sorted]
    f_nums_neg = ['F' + str(i) for i in mean_shap_values_neg_idx_sorted]

    ylim_max = max(max(mean_shap_values_pos_sorted), min(mean_shap_values_neg)*(-1))
    if ylim_max > 10:
        ylim_max = 10
    ylim_min = min(max(mean_shap_values_pos_sorted)*(-1), min(mean_shap_values_neg))
    if ylim_min < -10:
        ylim_min = -10

    # plot the positive SHAP values
    ax_pos.bar(f_nums_pos, mean_shap_values_pos_sorted, color='#49838B')
    ax_pos.axvline(x=num_important_factors_pos+0.5, color='black', linestyle='--')
    ax_pos.set_ylim([0, ylim_max+0.01])
    ax_pos.set_yticks(np.round(np.linspace(0, ylim_max, 3), 1))
    ax_pos.set_yticklabels(np.round(np.linspace(0, ylim_max, 3), 1))
    ax_pos.set_xticklabels(f_nums_pos, size=8)

    # remove the top, left and right borders
    ax_pos.spines['top'].set_visible(False)
    ax_pos.spines['right'].set_visible(False)
    ax_pos.spines['left'].set_visible(False)
    
    ax_neg.bar(f_nums_neg, mean_shap_values_neg_sorted, color='#DE6957')
    ax_neg.axvline(x=num_important_factors_neg+0.5, color='black', linestyle='--')
    ax_neg.set_ylim([ylim_min-0.01, 0])
    ax_neg.set_yticks(np.round(np.linspace(ylim_min, 0, 3), 1))
    ax_neg.set_yticklabels(np.round(np.linspace(ylim_min, 0, 3), 1))
    ax_neg.set_xticklabels(f_nums_neg, size=8)

    # remove the top, left and right borders
    ax_neg.spines['bottom'].set_visible(False)
    ax_neg.spines['right'].set_visible(False)
    ax_neg.spines['left'].set_visible(False)

     # Add legend
    if legend == True:
        legend_labels = ['Avg positive SHAP', 'Avg negative SHAP']
        legend_colors = ['#49838B', '#DE6957']
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
        ax_neg.legend(legend_handles, legend_labels, loc='lower right', ncol=1, fontsize=8, frameon=False)
    
    ax_pos.set_title('')
    ax_neg.set_title('')


def plot_shap_features(shap_values_t, factor_influence_sorted, shap_pos_idx, shap_neg_idx, 
                       all_params, param_names, ax, colorbar=False):
    print('Plotting the beeswarm plot...')
    # plot the beeswarm plot

    plt.sca(ax)
    shap.summary_plot(shap_values_t, all_params, feature_names=np.arange(len(param_names)), plot_type='dot',
                      sort=True, show=False, color_bar=colorbar)
    
    ax.set_yticklabels(factor_influence_sorted, size=6)
    # reduce distance between figure and axis label
    ax.tick_params(axis='y', which='major', pad=0.05)  # Adjust the padding here

    # Save the current x-axis limits
    x_min, x_max = ax.get_xlim()
    
    shap_pos = shap_values_t[shap_pos_idx,:]
    shap_neg = shap_values_t[shap_neg_idx,:]

    ax.scatter(shap_pos, np.arange(len(param_names)), color='black', marker='*', s=30, zorder=10, linewidth=1.5,
               facecolors='none')
    ax.scatter(shap_neg, np.arange(len(param_names)), color='black', marker='x', s=25, zorder=11, linewidth=2.5)

    # Ensure the x-axis limits are the same
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.round(np.linspace(x_min, x_max, 3), 2))
    ax.set_xticklabels(np.round(np.linspace(x_min, x_max, 3), 2), size=6)
    ax.set_xlabel('SHAP Value', size=8)
    ax.set_ylabel('DU Factor', size=8)

def plot_factor_barplot(feature_importances, param_names, ax):
    # check that feature importances have the same length as the parameter names
    if len(feature_importances) != len(param_names):
        raise ValueError('Feature importances and parameter names must have the same length')
    
    # create a dataframe of feature importances and parameter names
    feature_importances_df = pd.DataFrame({
        'Feature Importance': feature_importances,
        'Parameter': np.arange(len(param_names))
    })

    # plot vertical barplot of feature importances
    sns.barplot(x='Parameter', y='Feature Importance', data=feature_importances_df, ax=ax, color='indianred')
    ax.set_xlabel('DU Factors', size=8)
    ax.set_ylabel('Factor\nimportance', rotation=90, size=8)

def boosted_trees_factor_ranking(satisficing, all_params, param_names, window_size, sol_num, util_list, util_num):
    '''
    This function calculates the factor ranking for each week in the simulation period for the satisficing 
    criteria of a given utility.

    :param satisficing: a binary array of size NUM_RDM x (NUM_WEEKS - window_size + 1) that indicates whether
        the satisficing criteria is met for each week in the simulation period for each DU SOW
    :param all_params: a dataframe of size (NUM_WEEKS - NUM_YEARS + 1) x NUM_DU_factors that contains the values of all DU factors
        for each DU SOW
    :param param_names: a list of strings that contains the names of all DU factors
    
    :param window_size: int, the size of the rolling window for which the satisficing criteria is evaluated
    :param sol_num: int, the DU SOW number for which the satisficing criteria is evaluated

    :return factor_ranking_df: a dataframe of size (NUM_WEEKS - window_size + 1) x NUM_DU_factors that contains
    '''
    factor_ranking = np.zeros([NUM_WEEKS - window_size + 1, len(param_names)], dtype=float)

    # fit a boosted tree classifier to rank and extract important factors 
    print('Fitting boosted tree classifier...')
    
    clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, random_state=42, 
                            use_label_encoder=False, eval_metric='logloss')

    print('For each timestep across all DU SOWs, fit all parameters to the satisficing criteria...')
    for t in range(0, NUM_WEEKS - window_size + 1):

        satisficing_t = np.nan_to_num(satisficing[:NUM_RDM, t])
        if np.sum(satisficing_t) == NUM_RDM or np.sum(satisficing_t) == 0:
            continue
        else:
            clf.fit(all_params, satisficing_t)

            factor_ranking_t = clf.feature_importances_
            factor_ranking[t, :] = factor_ranking_t
    
    factor_ranking_df = pd.DataFrame(factor_ranking, columns=param_names, index=None)
    
    print('Saving factor ranking dataframe...')
    factor_ranking_filename = f'output_files/GBR_factor_ranking_sol{sol_num}_util{util_num}.csv'
    factor_ranking_df.to_csv(factor_ranking_filename)
    return factor_ranking_df

# function to perform SHAP analysis and obtain the SHAP values of each feature at every timestep 
def do_the_shap(model, all_params, param_names):
    #pred = model.predict(all_params, output_margin=True)
    explainer = shap.TreeExplainer(model)
    #explanation = explainer(all_params)
    shap_values = explainer.shap_values(all_params)

    # split the shap values up into positive and negative contributions
    shap_values_pos = np.where(shap_values > 0, shap_values, 0)
    shap_values_neg = np.where(shap_values < 0, shap_values, 0)
    #shap_values_zero = np.where(shap_values == 0, shap_values, 0)

    feature_values_pos = np.where(shap_values > 0, all_params, 0)
    feature_values_neg = np.where(shap_values < 0, all_params, 0)
    #feature_values_zero = np.where(shap_values == 0, all_params, 0)

    # find the highest positive SHAP value for each feature 
    max_shap_values_pos = np.max(shap_values_pos, axis=0)
    # find the feature values associated with the highest positive SHAP value
    max_feature_values_pos = np.zeros(len(param_names), dtype=float)
    for i in range(len(param_names)):
        max_feature_values_pos[i] = feature_values_pos[np.argmax(shap_values_pos[:, i]), i].flatten() 

    # find the highest negative SHAP value for each feature
    max_shap_values_neg = np.min(shap_values_neg, axis=0)
    # find the feature values associated with the highest negative SHAP value
    max_feature_values_neg = np.zeros(len(param_names), dtype=float)
    for i in range(len(param_names)):
        max_feature_values_neg[i] = feature_values_neg[np.argmin(shap_values_neg[:, i]), i].flatten()   
    
    return max_shap_values_pos, max_shap_values_neg, max_feature_values_neg, max_feature_values_pos

def boosted_trees_factor_ranking_with_shap_consequential(satisficing, all_params, param_names, 
                                                         periods_df, window_size, sol_num, util_list, 
                                                         util_num, lag=0, period_name='conseq'):
    
    if periods_df.shape[0] == 0:
        print(f'Robustness does not change at a consequential rate for {util_list[util_num]}')
        return None
    
    shap_values_pos = np.zeros([NUM_WEEKS - window_size + 1, len(param_names)], dtype=float)
    shap_values_neg = np.zeros([NUM_WEEKS - window_size + 1, len(param_names)], dtype=float)
    feature_values_pos = np.zeros([NUM_WEEKS - window_size + 1, len(param_names)], dtype=float)
    feature_values_neg = np.zeros([NUM_WEEKS - window_size + 1, len(param_names)], dtype=float)

    # fit a boosted tree classifier to rank and extract important factors 
    print('Fitting boosted tree classifier...')
    clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, random_state=42, 
                            use_label_encoder=False, eval_metric='logloss')

    for row in range(periods_df.shape[0]):
        start_t = max(periods_df.iloc[row, 0]-lag, 0)
        #print('start_t:', start_t)
        end_t = periods_df.iloc[row, 1]

        print(f'Currently in the period {row} from {start_t} to {end_t}...')

        for t in range(start_t, end_t):
            satisficing_t = np.nan_to_num(satisficing[:NUM_RDM, t])
            if np.sum(satisficing_t) == NUM_RDM or np.sum(satisficing_t) == 0:
                continue
            else:
                clf.fit(all_params, satisficing_t)

                max_shap_values_pos, max_shap_values_neg, \
                    max_feature_values_neg, max_feature_values_pos = do_the_shap(clf, all_params, param_names)
                
                shap_values_pos[t, :] = max_shap_values_pos
                shap_values_neg[t, :] = max_shap_values_neg
                feature_values_pos[t, :] = max_feature_values_pos
                feature_values_neg[t, :] = max_feature_values_neg
    
    shap_values_pos_df = pd.DataFrame(shap_values_pos, columns=param_names, index=None)
    shap_values_neg_df = pd.DataFrame(shap_values_neg, columns=param_names, index=None)
    feature_values_pos_df = pd.DataFrame(feature_values_pos, columns=param_names, index=None)
    feature_values_neg_df = pd.DataFrame(feature_values_neg, columns=param_names, index=None)
    
    print('Saving dataframes...')
    shap_values_pos_filename = f'output_files/SHAP_values_pos_s{sol_num}_u{util_num}_{period_name}.csv'
    shap_values_pos_df.to_csv(shap_values_pos_filename)
    shap_values_neg_filename = f'output_files/SHAP_values_neg_s{sol_num}_u{util_num}_{period_name}.csv'
    shap_values_neg_df.to_csv(shap_values_neg_filename)
    feature_values_pos_filename = f'output_files/feature_values_pos_s{sol_num}_u{util_num}_{period_name}.csv'
    feature_values_pos_df.to_csv(feature_values_pos_filename)
    feature_values_neg_filename = f'output_files/feature_values_neg_s{sol_num}_u{util_num}_{period_name}.csv'
    feature_values_neg_df.to_csv(feature_values_neg_filename)

    shap_list = [shap_values_pos_df, shap_values_neg_df, feature_values_pos_df, feature_values_neg_df]
    return shap_list

def plot_timevarying_gbr_shap(shap_values_list, periods_df, 
                              util_num, util_list, sol_num, factor_names, 
                              factor_colors, period_name='conseq'):
    shap_pos_df = shap_values_list[0]
    shap_neg_df = shap_values_list[1]
    feature_pos_df = shap_values_list[2]
    feature_neg_df = shap_values_list[3]

    # normalize the features to be between 0 and 1
    feature_pos_df_shited = feature_pos_df + np.abs(np.min(feature_pos_df, axis=0))
    feature_pos_df_norm = feature_pos_df_shited / np.max(feature_pos_df_shited, axis=0)
    feature_neg_df_shited = feature_neg_df + np.abs(np.min(feature_neg_df, axis=0))
    feature_neg_df_norm = feature_neg_df_shited / np.max(feature_neg_df_shited, axis=0)

    shap_pos_values = shap_pos_df.values
    shap_neg_values = shap_neg_df.values
    features_pos_values = feature_pos_df_norm.values
    features_neg_values = feature_neg_df_norm.values

    # any shap values that are < 1 are set to 0
    shap_pos_values = np.where(np.abs(shap_pos_values) < 0.5, 0, shap_pos_values)
    shap_neg_values = np.where(np.abs(shap_neg_values) < 0.5, 0, shap_neg_values)

    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.set_figheight(4)
    fig.set_figwidth(12)

    # plot the positive SHAP values 
    x_values_pos = np.arange(0, shap_pos_df.shape[0])
    ymax = 1.0
    ymin = 0.0
    ymin_shap = -1.0
    
    y_base = np.zeros(shap_pos_df.shape[0], dtype=float)
    y_base_neg = np.zeros(shap_neg_df.shape[0], dtype=float)

    # iterate through each timestep
    for t in range(shap_pos_df.shape[0]):
        std_pos = np.var(np.abs(shap_pos_values[t,:]))
        std_neg = np.var(np.abs(shap_neg_values[t,:]))

        if std_pos < 0.125:
            shap_pos_values[t,:] = 0.0
        if std_neg < 0.125:
            shap_neg_values[t,:] = 0.0


    # find and plot positive SHAP values
    for i in range(len(factor_names)):
        y_pos = np.zeros(shap_pos_df.shape[0], dtype=float)
        y_pos_idx = np.where(shap_pos_values[:,i] > 0)

        y_pos[y_pos_idx] = shap_pos_values[y_pos_idx,i]
        y_pos = y_pos + y_base

        ax.fill_between(x_values_pos, y_base, y_pos, where=y_pos>y_base, color=factor_colors[i], 
                        alpha=1.0, label=factor_names[i])
        
        y_base = y_pos

        y_neg = np.zeros(shap_neg_df.shape[0], dtype=float)
        y_neg_idx = np.where(shap_neg_values[:,i] < 0)
        y_neg[y_neg_idx] = shap_neg_values[y_neg_idx,i]
        y_neg = y_neg + y_base_neg
        
        ax.fill_between(x_values_pos, y_base_neg, y_neg, where=y_neg<y_base_neg, color=factor_colors[i], 
                        alpha=1.0, label=factor_names[i])

        y_base_neg = y_neg

    # insert a horizontal dotted line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # place text immediately above the line on the right hand size
    ax.text(1.0, 0.55, 'No influence on prediction', color='black', fontsize=10, 
                ha='right', va='center', transform=ax.transAxes)

    ax.set_xlim(0, shap_pos_values.shape[0])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_ylabel(r'$\longleftarrow$ Drives failure   Drives success $\longrightarrow$', fontsize=8)
    ax.set_xlabel('Simulation years', fontsize=12)

    ax.set_xticks(np.arange(0, NUM_WEEKS, 52*5))
    ax.set_xticklabels(np.arange(0, 50, 5), fontsize=10)

    # turn off top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-20,20)

    fig_title = f'{util_list[util_num]} SHAP values'
    ax.set_title(fig_title,fontsize=12)
    
    print('Setting legend...')
    legend_labels = []

    for l in range(len(factor_names)):
        input = plt.Rectangle((0,0), 1, 1, fc=factor_colors[l], edgecolor='none')
        legend_labels.append(input)      

    plt.legend(legend_labels, factor_names, ncol=7, loc='lower center', fontsize='small', 
               frameon=False, bbox_to_anchor=(0.5, -0.4))
    plt.tight_layout()
    plt.savefig(f'GBR_SHAP_sol{sol_num}_util{util_num}_{period_name}.jpg',dpi=300, bbox_inches='tight')
