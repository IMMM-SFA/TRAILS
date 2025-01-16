import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_REAL = 500   # number of hydroclimatic realizations
NUM_RDM = 100  # number of DU SOWs
SOL_NUM = 167   # replace with 62 after this

util_list = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']

infra_dict = {9: ['Little River Reservoir','#FFD275'], 10: ['Richland Creek Quarry','#E8AE68'], 
              11: ['Teer Quarry', '#A57F60'], 12: ['Neuse River Intake', '#E3A587'],
              13: ['Harnett Intake', '#DB5A42'], 15: ['Stone Quarry Exp (L)', '#2A2B2A'], 
              16: ['Stone Quarry Exp (H)', '#5E4955'], 17: ['University Lake Exp', '#996888'], 
              18: ['Michie Exp (L)', '#C99DA3'], 19: ['Michie Exp (H)', '#C6DDF0'], 
              20: ['Falls Lake Realloc', '#F6E27F'], 21: ['Reclaimed Water (L)', '#E2C391'], 
              22: ['Reclaimed Water (H)', '#A8B7AB'], 23: ['Cane Creek Res Exp', '#9BBEC7'], 
              24: ['Haw Intake Exp (L)', '#32213A'], 25: ['Haw Intake Exp (H)', '#383B53'], 
              26:['Cary WTP Upgrade', '#66717E'], 27: ['Sanford WTP Upgrade (L)', '#D72483'], 
              28: ['Sanford WTP Upgrade (H)', '#792359'], 29:['Jordan Lake WTP Fixed (L)', '#C44536'], 
              30: ['Jordan Lake WTP Fixed (H)', '#772E25'], 33: ['Cary Sanford Intake', '#7192BE']}

infra_num_owasa = [15,16,17,23,29,30]
infra_num_durham = [11,18,19,21,22,29,30]
infra_num_cary = [13,26,33]
infra_num_raleigh = [9,10,12,20]
infra_num_chatham = [27,28,29,30]
infra_num_pittsboro = [24,25,27,28,30,29]

infra_num_util_dict = {0: infra_num_owasa, 1: infra_num_durham, 2: infra_num_cary, 
                       3: infra_num_raleigh, 4: infra_num_chatham, 5: infra_num_pittsboro}

infra_owasa = [infra_dict[key] for key in infra_num_owasa if key in infra_dict]
infra_durham = [infra_dict[key] for key in infra_num_durham if key in infra_dict]
infra_cary = [infra_dict[key] for key in infra_num_cary if key in infra_dict]
infra_raleigh = [infra_dict[key] for key in infra_num_raleigh if key in infra_dict]
infra_chatham = [infra_dict[key] for key in infra_num_chatham if key in infra_dict]
infra_pittsboro = [infra_dict[key] for key in infra_num_pittsboro if key in infra_dict]

def get_infra_byutil(util_num, infra_num_util_dict, SOL_NUM):
    infra_dict = {}
    util_infra_num = infra_num_util_dict[util_num]

    for infra_num in range(len(util_infra_num)):
        infra_id = util_infra_num[infra_num]
        infra_firstweeks = np.zeros(NUM_RDM)

        for rdm in range(NUM_RDM):
            pathways_filename = 'sol{}_objs_pathways/Pathways_s{}_RDM{}.out'.format(SOL_NUM, SOL_NUM, rdm)
        
            pathways = pd.read_csv(pathways_filename, index_col=None, delimiter='\t', header=0)
            
            pathways_util = pathways[pathways['utility'] == util_num]
            
            if pathways_util.empty:
                continue
            else:
                infra_occurrence_idx = pathways_util[pathways_util['infra.'] == infra_id].index
                if infra_occurrence_idx.empty:
                    continue
                else:
                    infra_firstweeks[rdm] = np.min(pathways_util.loc[infra_occurrence_idx, 'week'].values)
        
        infra_dict[infra_id] = infra_firstweeks
    return infra_dict

def plot_infra_byutil(util_num, infra_dict, util_infra_dict, infra_num_util_dict, SOL_NUM):
    util_infra_num = infra_num_util_dict[util_num]
    util_infra_df = pd.DataFrame.from_dict(util_infra_dict)

    sns.set_style('white')
    fig, axes = plt.subplots(nrows=len(util_infra_num), ncols=1, figsize=(14, 6), sharex=True)
    
    # Remove vertical space between subplots
    #plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(hspace=1.0)
    # Iterate through subplots
    for i, infra in enumerate(util_infra_df.columns):
        # Create KDE plot with color-coded fill representing robustness
        infra_name = infra_dict[infra][0]
        nonzero_weeks = util_infra_df[infra]
        #print(nonzero_weeks)
        sns.kdeplot(x=nonzero_weeks[nonzero_weeks != 0], fill=True, color='green', shade=True, ax=axes[i])
        
        # Set subplot title
        axes[i].set_ylabel('')
        axes[i].set_title(infra_name)

        # Hide tick labels
        axes[i].tick_params(axis='y', labelleft=False)
        
        # Remove legend from all but the last subplot
        #axes[i].get_legend().remove()
        axes[i].set_xlim([0, 2344])
    
    # Remove subplot borders
    for ax in range(len(axes)):
        axes[ax].spines['right'].set_visible(False)
        axes[ax].spines['left'].set_visible(False)
        axes[ax].spines['top'].set_visible(False)
        #if ax == 0:
            #axes[ax].spines['top'].set_visible(False)

    # Add a y-label to the leftmost subplot
    fig.text(0.04, 0.5, util_list[util_num], va='center', rotation='vertical', fontsize=14)
    # Decrease the space between the y-label and subplots
    fig.subplots_adjust(left=0.06)

    # set the x-axis tick values 
    weeks = np.arange(0, NUM_WEEKS+1)
    years = np.array(weeks) // 52

    plt.xticks(np.arange(min(weeks), max(weeks), step=5*52), labels=range(min(years), max(years)+1, 5))

    axes[len(util_infra_num)-1].set_xlabel('Year of first infrastructure project')

    # Show the plot
    plt.savefig('post_processing/infra_kde_figures/util{}_infra_kde_s{}.png'.format(util_num, SOL_NUM))
    plt.show()


def get_infra_firstweeks(util_num, SOL_NUM):
    # get the first week of each infrastructure project
    infra_firstweeks = np.zeros(NUM_RDM)

    for rdm in range(NUM_RDM):
        pathways_filename = 'sol{}_objs_pathways/Pathways_s{}_RDM{}.out'.format(SOL_NUM, SOL_NUM, rdm)
    
        pathways = pd.read_csv(pathways_filename, index_col=None, delimiter='\t', header=0)
        
        pathways_util = pathways[pathways['utility'] == util_num]
        
        if pathways_util.empty:
            continue
        else:
            first_occurrences = pathways_util[~pathways_util['Realization'].duplicated(keep='first')].index
            
            # get the median week of the first occurrence of an infrastructure project across all realizations
            infra_firstweeks[rdm] = np.median(pathways_util.loc[first_occurrences, 'week'].values)

    return infra_firstweeks
