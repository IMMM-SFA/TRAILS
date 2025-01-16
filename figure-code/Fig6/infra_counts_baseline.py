
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pathways_processing_functions import *
from custom_colormap import custom_cmap

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_REAL = 1000   # number of hydroclimatic realizations
NUM_RDM = 1000  # number of DU SOWs
SOL_NUMS = [92, 132, 140]
SOL_IDX = 2
SOL_NUM = SOL_NUMS[SOL_IDX]  
window_size = 52  # one year rolling window
util_nums = np.arange(0,6)

util_list = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
'''
infra_dict = {9: 'Little River Reservoir', 10: 'Richland Creek Quarry', 
              11: 'Teer Quarry', 12:'Neuse River Intake',
              13: 'Harnett Intake', 15:'Stone Quarry Exp (L)', 
              16: 'Stone Quarry Exp (H)', 17:'University Lake Exp', 
              18: 'Michie Exp (L)', 19: 'Michie Exp (H)', 
              20: 'Falls Lake Realloc', 21: 'Reclaimed Water (L)', 
              22: 'Reclaimed Water (H)', 23: 'Cane Creek Res Exp', 
              24: 'Haw Intake Exp (L)', 25: 'Haw Intake Exp (H)', 
              26:'Cary WTP Upgrade', 27: 'Sanford WTP Upgrade (L)', 
              28: 'Sanford WTP Upgrade (H)', 29:'Jordan Lake WTP Fixed (L)', 
              30: 'Jordan Lake WTP Fixed (H)', 33: 'Cary Sanford Intake'}
'''
infra_dict = {9: 'LRR', 10: 'RCQ', 11: 'TQ', 12:'NRI', 13: 'HI', 15:'SQE-L', 
              16: 'SQE-H', 17:'ULE', 18: 'ME-L', 19: 'ME-H', 20: 'FLR', 21: 'RW-L', 
              22: 'RW-H', 23: 'CCRE', 24: 'HIE-L', 25: 'HIE-H', 26:'CWU', 27: 'SWU-L', 
              28: 'SWU-H', 29:'JLWTP-L', 30: 'JLWTP-H', 33: 'CSI'}

infra_num_owasa = [15,16,17,23,29,30]
infra_num_durham = [11,18,19,21,22,29,30]
infra_num_cary = [13,26,33]
infra_num_raleigh = [9,10,12,20]
infra_num_chatham = [27,28,29,30]
infra_num_pittsboro = [24,25,27,28,29,30]

infra_num_util_dict = {0: infra_num_owasa, 1: infra_num_durham, 2: infra_num_cary, 
                       3: infra_num_raleigh, 4: infra_num_chatham, 5: infra_num_pittsboro}

infra_owasa = [infra_dict[key] for key in infra_num_owasa if key in infra_dict]
infra_durham = [infra_dict[key] for key in infra_num_durham if key in infra_dict]
infra_cary = [infra_dict[key] for key in infra_num_cary if key in infra_dict]
infra_raleigh = [infra_dict[key] for key in infra_num_raleigh if key in infra_dict]
infra_chatham = [infra_dict[key] for key in infra_num_chatham if key in infra_dict]
infra_pittsboro = [infra_dict[key] for key in infra_num_pittsboro if key in infra_dict]

pathways_folder = 'sol{}_objs_pathways'.format(SOL_NUM)
output_folder = 'sol{}_objs_pathways/infra_counts_baseline_byreals'.format(SOL_NUM)   

def count_infrastructure_likelihood_row_weeks(util_num, sol_num):
    util_infra_nums = infra_num_util_dict[util_num]
    util_allinfra = [infra_dict[infra_num] for infra_num in util_infra_nums]
    #print(f'all infrastructure to be built by {util_list[util_num]}: {util_allinfra}')
    
    all_rdm_infra_counts = np.zeros([NUM_WEEKS, len(util_infra_nums)], dtype=int)
    
    for rdm in range(NUM_RDM):
        pathways_file = pathways_folder + '/Pathways_s{}_RDM{}.out'.format(sol_num, rdm)
        pathways = pd.read_csv(pathways_file, delimiter='\t', header=0)
        
        util_pathways = pathways[pathways['utility'] == util_num]

        #util_infra_counts_byweek = np.zeros([NUM_WEEKS, len(util_infra_nums)], dtype=int)

        for infra in range(len(util_infra_nums)):
            if util_pathways[util_pathways['infra.'] == util_infra_nums[infra]].empty:
                continue
            else:
                infra_week_list = util_pathways[util_pathways['infra.'] == util_infra_nums[infra]]['week'].values
                #if util_infra_nums[infra] == 30:
                    #print(f'infra {util_infra_nums[infra]} detected in RDM: {rdm}')
                infra_week = infra_week_list[0]
                
                # util_infra_counts_byweek[infra_week, infra] = len(infra_week_list)
                all_rdm_infra_counts[infra_week, infra] += len(infra_week_list)
                
            # get the week in which infrastructure is most often built across all realizations 
            # for a given DU SOW
            #all_rdm_median_infra_week[rdm, infra] = np.median(util_infra_counts_byreal[:, infra])
        
        # the number in each cell represents the total number of times a utility builds a given infrastructure 
        # option in a given week across all realizations
        # output_file = output_folder + '/util{}_rdm{}_counts_byweek.csv'.format(util_num, rdm)
        # output_file_rdm = output_folder + '/util{}_allRDM_counts_byweek.csv'.format(util_num, rdm)
        
        # save each RDM as a csv
        # util_infra_counts_byweek_df = pd.DataFrame(util_infra_counts_byweek, columns=util_allinfra)

        # probability that a utility builds a given infrastructure option in a given week across all realizations 
        # and across all RDMs
        # represents likelihood that an infrastructure option will be triggered in a given week 

        # util_infra_counts_byweek_df.to_csv(output_file, index=False)
        # all_rdm_median_infra_week_df.to_csv(output_file_rdm, index=False)
    
    # save the probability across all RDMs and realization for a given utility
    all_rdm_infra_counts_df = pd.DataFrame(all_rdm_infra_counts, columns=util_allinfra)
    #all_rdm_infra_counts_df.to_csv(output_folder + '/util{}_allRDM_counts_byweek.csv'.format(util_num), index=False)
    return all_rdm_infra_counts_df

owasa_infra_counts_df = count_infrastructure_likelihood_row_weeks(0, SOL_NUM)
durham_infra_counts_df = count_infrastructure_likelihood_row_weeks(1, SOL_NUM)
cary_infra_counts_df = count_infrastructure_likelihood_row_weeks(2, SOL_NUM)
raleigh_infra_counts_df = count_infrastructure_likelihood_row_weeks(3, SOL_NUM)
chatham_infra_counts_df = count_infrastructure_likelihood_row_weeks(4, SOL_NUM)
pittsboro_infra_counts_df = count_infrastructure_likelihood_row_weeks(5, SOL_NUM)

owasa_infra_counts_rolling_df = (owasa_infra_counts_df.apply(lambda col: 
                                                            col.rolling(window=window_size).sum(), 
                                                            axis=0)/(NUM_RDM*NUM_REAL))*100
durham_infra_counts_rolling_df = (durham_infra_counts_df.apply(lambda col: 
                                                              col.rolling(window=window_size).sum(), 
                                                              axis=0)/(NUM_RDM*NUM_REAL))*100
cary_infra_counts_rolling_df = (cary_infra_counts_df.apply(lambda col: 
                                                          col.rolling(window=window_size).sum(), 
                                                          axis=0)/(NUM_RDM*NUM_REAL))*100
raleigh_infra_counts_rolling_df = (raleigh_infra_counts_df.apply(lambda col: 
                                                                col.rolling(window=window_size).sum(), 
                                                                axis=0)/(NUM_RDM*NUM_REAL))*100
chatham_infra_counts_rolling_df = (chatham_infra_counts_df.apply(lambda col: 
                                                                col.rolling(window=window_size).sum(), 
                                                                axis=0)/(NUM_RDM*NUM_REAL))*100
pittsboro_infra_counts_rolling_df = (pittsboro_infra_counts_df.apply(lambda col: 
                                                                    col.rolling(window=window_size).sum(), 
                                                                    axis=0)/(NUM_RDM*NUM_REAL))*100

def count_infrastructure_medians_row_reals(util_num, sol_num):
    util_infra_nums = infra_num_util_dict[util_num]
    util_allinfra = [infra_dict[infra_num] for infra_num in util_infra_nums]
    #print(f'all infrastructure to be built by {util_list[util_num]}: {util_allinfra}')
    
    all_rdm_median_infra_week = np.zeros([NUM_RDM, len(util_infra_nums)], dtype=int)
    
    for rdm in range(NUM_RDM):
        pathways_file = pathways_folder + '/Pathways_s{}_RDM{}.out'.format(sol_num, rdm)
        pathways = pd.read_csv(pathways_file, delimiter='\t', header=0)
        
        util_pathways = pathways[pathways['utility'] == util_num]

        util_infra_counts_byreal = np.zeros([NUM_REAL, len(util_infra_nums)], dtype=int)

        for infra in range(len(util_infra_nums)):
            if util_pathways[util_pathways['infra.'] == util_infra_nums[infra]].empty:
                continue
            else:
                infra_real_list = util_pathways[util_pathways['infra.'] == util_infra_nums[infra]]['Realization'].values
                util_infra_counts_byreal[infra_real_list, infra] = util_pathways[util_pathways['infra.'] == 
                                                                                 util_infra_nums[infra]]['week'].values

            # get the week in which infrastructure is most often built across all realizations 
            # for a given DU SOW
            util_infra_counts_byreal_infra = util_infra_counts_byreal[:, infra]
            util_infra_counts_byreal_nozeros = util_infra_counts_byreal_infra[util_infra_counts_byreal_infra != 0]

            all_rdm_median_infra_week[rdm, infra] = np.median(util_infra_counts_byreal_nozeros)
            #print(f'median weeks: {np.median(util_infra_counts_byreal_nozeros)}')
        # the number in each cell represents the total number of times a utility builds a given infrastructure 
        # option in a given week across all realizations
        # output_file = output_folder + '/util{}_rdm{}_counts_byweek.csv'.format(util_num, rdm)
        # output_file_rdm = output_folder + '/util{}_allRDM_counts_byweek.csv'.format(util_num, rdm)
        
        # save each RDM as a csv
        # util_infra_counts_byweek_df = pd.DataFrame(util_infra_counts_byweek, columns=util_allinfra)

        # probability that a utility builds a given infrastructure option in a given week across all realizations 
        # and across all RDMs
        # represents likelihood that an infrastructure option will be triggered in a given week 

        #util_infra_counts_byweek_df.to_csv(output_file, index=False)
        #all_rdm_median_infra_week_df.to_csv(output_file_rdm, index=False)
    
    # save the probability across all RDMs and realization for a given utility
    all_rdm_median_infra_reals_df = pd.DataFrame(all_rdm_median_infra_week, columns=util_allinfra)
    #all_rdm_median_infra_reals_df.to_csv(output_folder + '/util{}_allRDM_median_infra_reals.csv'.format(util_num), index=False)
    return all_rdm_median_infra_reals_df

owasa_infra_medians_df = count_infrastructure_medians_row_reals(0, SOL_NUM)
durham_infra_medians_df = count_infrastructure_medians_row_reals(1, SOL_NUM)
cary_infra_medians_df = count_infrastructure_medians_row_reals(2, SOL_NUM)
raleigh_infra_medians_df = count_infrastructure_medians_row_reals(3, SOL_NUM)
chatham_infra_medians_df = count_infrastructure_medians_row_reals(4, SOL_NUM)
pittsboro_infra_medians_df = count_infrastructure_medians_row_reals(5, SOL_NUM)

def plot_infra_kde_byutil(util_infra_medians_df, axes, base_color, robustness_oneutil):
    util_infra_names = util_infra_medians_df.columns

    # Iterate through subplots
    for i, infra in enumerate(util_infra_medians_df.columns):
        # Create KDE plot with color-coded fill representing robustness
        infra_name = util_infra_names[i]
        nonzero_weeks = util_infra_medians_df[infra]
        axes_i_twinx = axes[i].twinx()
        sns.kdeplot(x=nonzero_weeks[nonzero_weeks != 0], fill=True, color=base_color, 
                    shade=True, ax=axes_i_twinx, alpha=0.85)
        
        # Label each subplot with the infrastructure name
        # get the associated infrastructure number 
        axes[i].set_ylabel(infra_name, fontsize=8, rotation=0, ha='left')
        
        # Hide y-tick labels
        axes[i].tick_params(axis='y', labelleft=False)
        axes_i_twinx.tick_params(axis='y', labelleft=False)
        axes_i_twinx.tick_params(axis='y', labelright=False)
        axes_i_twinx.set_yticks([])
        axes_i_twinx.set_ylabel('')

        # format the twin axis
        axes_i_twinx.spines['right'].set_visible(False)
        axes_i_twinx.spines['left'].set_visible(False)
        axes_i_twinx.spines['top'].set_visible(False)
        axes_i_twinx.set_xlim([0, 2344])
    
    weeks = np.arange(0, NUM_WEEKS+1)
    years = np.array(weeks) // 52

    plt.xticks(np.arange(min(weeks), max(weeks), step=5*52), 
               labels=range(min(years), max(years)+1, 5))

from matplotlib.colors import BoundaryNorm

def plot_infrastructure_likelihood_bars_byweek(util_infra_counts_df, fig, axes, custom_cmap):
    infra_names = util_infra_counts_df.columns
    NUM_WEEKS = util_infra_counts_df.shape[0]

    # Create a ScalarMappable to map the normalized values to colors
    #cmap = plt.cm.pink_r
    cmap = custom_cmap

    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #norm = Normalize(vmin=0, vmax=1.0)
    norm = BoundaryNorm(bounds, cmap.N, extend='both')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    infra_max = util_infra_counts_df.max().max()
    infra_min = util_infra_counts_df.min().min()

    util_infra_counts_df = util_infra_counts_df.fillna(0)

    util_infra_counts_norm_df = (util_infra_counts_df - infra_min)/(infra_max - infra_min)
    
    # Plotting the stacked bar plot
    for i, infra in enumerate(infra_names):
        #util_infra_counts_df[infra] = util_infra_counts_norm_df[infra].fillna(0)
        inf_im = util_infra_counts_norm_df[infra].to_numpy() 

        # normalize inf_im 
        #inf_im_norm = (inf_im - infra_min)/(infra_max - infra_min)
        #inf_im = np.nan_to_num(inf_im)
        inf_im = inf_im.reshape(1, NUM_WEEKS)

        # Iterate through each week and add a stack to the bar
        axes[i].imshow(inf_im, cmap=cmap, aspect='auto', alpha=0.85, 
                       vmin=0, vmax=1.0)
        axes[i].set_yticks([])
        #print(np.max(inf_im))
        # Add a colorbar
    cax = fig.add_axes([0.1, -0.05, 0.80, 0.025])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.5, aspect=30, cax=cax)
    cbar.set_label('Fraction of DU SOWs', fontsize=6)
    x_ticks = np.linspace(infra_min, infra_max, 6)
    # round the ticks to 2 decimal places
    x_ticks = np.round(x_ticks, 1)
    cax.set_xticks(x_ticks)
    cax.set_xticklabels(x_ticks, fontsize=6)

def plot_infra_kde_likelihood(util_infra_medians_df, util_infra_likelihood_df, util_num, 
                              infra_num, robustness_allutils, base_color):
    util_name = util_list[util_num]
    robustness_oneutil = robustness_allutils[util_num]
    util_infra_names = util_infra_medians_df.columns
    util_infra_num = len(util_infra_names)
    
    sns.set_style('white')

    print("Initializing figure and axes")
    fig, axes = plt.subplots(infra_num, 1, sharex=True)
    fig.set_figwidth(8)
    fig.set_figheight(int(infra_num)*0.75)

    # Remove vertical space between subplots
    #print("Removing vertical space between subplots")
    plt.subplots_adjust(hspace=0.0)

    plot_infrastructure_likelihood_bars_byweek(util_infra_likelihood_df, fig, axes, custom_cmap)
    plot_infra_kde_byutil(util_infra_medians_df, axes, base_color, robustness_oneutil)
    
    # Format each axis
    for ax in range(len(axes)):
        axes[ax].spines['right'].set_visible(False)
        axes[ax].spines['left'].set_visible(False)
        axes[ax].spines['top'].set_visible(False)
        axes[ax].set_xlim([0, 2344])

    # Increase the space between the y-label and subplots
    fig.subplots_adjust(left=0.10)

    # set the x-axis tick values 
    weeks = np.arange(0, NUM_WEEKS+1)
    years = np.array(weeks) // 52
    
    axes[util_infra_num-1].set_xticks(np.arange(min(weeks), max(weeks), step=5*52))
    axes[util_infra_num-1].set_xticklabels(range(min(years), max(years)+1, 5), fontsize=10)
    axes[util_infra_num-1].set_xlabel('Simulation year', labelpad=0.3, fontsize=8)
    #plt.tight_layout()
    plt.suptitle(f'Median and likelihood of first year of infrastructure construction for {util_name}',
                 fontsize=10, y=0.99)
    #fig.subplots_adjust(top=0.99)
    plt.savefig(f'infra_kde_likelihood_s{SOL_NUM}_u{util_num}_withbar.pdf', dpi=300, bbox_inches='tight')
    #plt.show()

infra_medians_df_toplot = ''
infra_counts_df_toplot = ''
infra_num_toplot = []

dark_colors = ['#BD6C10', '#227471', '#355544']
light_colors = ["#FFFFFF", "#FFFFFF", "#FFFFFF"]
base_colors = ['#EC9839', '#48A9A6', '#69957E']
kde_color = ['#EACBA6', '#A0E3E1', '#CDE2D7']

light_color = light_colors[SOL_IDX]
dark_color = dark_colors[SOL_IDX]  # Dark variant of the base color [#7A3D01, #045955, #0C522D]
base_color = base_colors[SOL_IDX]   # Base color [#DC851F, #48A9A6, #355544]

# get the robustness value 
robustness_dfsr = pd.read_csv('../objs_space_analysis/robustness_DFSR.csv', index_col=0)
robustness_allutils = robustness_dfsr.iloc[SOL_NUM,:6].values

custom_cmap = custom_cmap(base_color, light_color, dark_color)

for i in range(len(util_nums)):
    util_num = util_nums[i]
    if util_num == 0:
        infra_medians_df_toplot = owasa_infra_medians_df
        infra_counts_df_toplot = owasa_infra_counts_rolling_df
        infra_num_toplot = infra_num_owasa
    elif util_num == 1:
        infra_medians_df_toplot = durham_infra_medians_df
        infra_counts_df_toplot = durham_infra_counts_rolling_df
        infra_num_toplot = infra_num_durham
    elif util_num == 2:
        infra_medians_df_toplot = cary_infra_medians_df
        infra_counts_df_toplot = cary_infra_counts_rolling_df
        infra_num_toplot = infra_num_cary
    elif util_num == 3:
        infra_medians_df_toplot = raleigh_infra_medians_df
        infra_counts_df_toplot = raleigh_infra_counts_rolling_df
        infra_num_toplot = infra_num_raleigh
    elif util_num == 4:
        infra_medians_df_toplot = chatham_infra_medians_df
        infra_counts_df_toplot = chatham_infra_counts_rolling_df
        infra_num_toplot = infra_num_chatham
    else:
        infra_medians_df_toplot = pittsboro_infra_medians_df
        infra_counts_df_toplot = pittsboro_infra_counts_rolling_df
        infra_num_toplot = infra_num_pittsboro

    plot_infra_kde_likelihood(infra_medians_df_toplot, infra_counts_df_toplot, util_num, len(infra_num_toplot), 
                            robustness_allutils, kde_color[SOL_IDX])
    

