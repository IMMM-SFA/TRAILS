"""
Created on Tues Apr 16 10:41:00 2024 by @lbl59 (Lillian Lau)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

objs_filename = 'refset_objs_utils.csv'
objs_df = pd.read_csv(objs_filename, header=0)
selected_rows = [92, 132, 140]

policy_names = ['Durham-focused', 'Raleigh-focused', 'Regionally robust']
regional_policy_num = 140

rel_df = objs_df[['REL_O', 'REL_D', 'REL_C', 'REL_R', 'REL_Ch', 'REL_P']]
rf_df = objs_df[['RF_O', 'RF_D', 'RF_C', 'RF_R', 'RF_Ch', 'RF_P']]
inpc_df = objs_df[['NPC_O', 'NPC_D', 'NPC_C', 'NPC_R', 'NPC_Ch', 'NPC_P']]
pfc_df = objs_df[['PFC_O', 'PFC_D', 'PFC_C', 'PFC_R', 'PFC_Ch', 'PFC_P']]
wcc_df = objs_df[['WCC_O', 'WCC_D', 'WCC_C', 'WCC_R', 'WCC_Ch', 'WCC_P']]
uc_df = objs_df[['UC_O', 'UC_D', 'UC_C', 'UC_R', 'UC_Ch', 'UC_P']]

rel_selected = ((rel_df.iloc[selected_rows,:] - rel_df.iloc[regional_policy_num,:])/rel_df.iloc[regional_policy_num,:])*100
rf_selected = ((rf_df.iloc[selected_rows,:] - rf_df.iloc[regional_policy_num,:])/rf_df.iloc[regional_policy_num,:])*(-100)
inpc_selected = ((inpc_df.iloc[selected_rows,:] - inpc_df.iloc[regional_policy_num,:])/inpc_df.iloc[regional_policy_num,:])*(-100)
pfc_selected = ((pfc_df.iloc[selected_rows,:] - pfc_df.iloc[regional_policy_num,:])/pfc_df.iloc[regional_policy_num,:])*(-100)
wcc_selected = ((wcc_df.iloc[selected_rows,:] - wcc_df.iloc[regional_policy_num,:])/wcc_df.iloc[regional_policy_num,:])*(-100)
uc_selected = ((uc_df.iloc[selected_rows,:] - uc_df.iloc[regional_policy_num,:])/uc_df.iloc[regional_policy_num,:])*(-100)

objs_df_dict = {1: rel_selected, 2: rf_selected, 3: inpc_selected,
                4: pfc_selected, 5: wcc_selected, 6: uc_selected}

# get du factor names
util_names =  ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_abbrevs = ['O', 'D', 'C', 'R', 'Ch', 'P']
objs_names = ['REL', 'RF', 'INPC', 'PFC', 'WCC', 'UC']
obj_names_full = ['Reliability (%)', 'Restriction Frequency (%)', 'Infrastructure NPC ($mil)',
                  'Peak Financial Cost', 'Worst-Case Cost', 'Unit Cost ($/kgal)']

num_utils = len(util_names)
figname = f'objs_enoki_all.pdf'
supertitle = '% change in objectives by utility'

#colors = ['#4357AD', '#48A9A6', '#31493C', '#D4B483', '#C1666B']
colors = ['#DC851F', '#48A9A6', '#355544']  

sns.set_style("white")
fig, axes = plt.subplots(3, 2, figsize=(10, 12))  # one subplot for each objective
axes = axes.flatten()
plt.subplots_adjust(wspace=0.35, hspace=0.35)

handles = []
labels = []
x_df = np.arange(0,len(util_names))
x_rf = np.arange(0,len(util_names))+0.4

x_pos = [x_df, x_rf]
for i in range(len(objs_names)):
    selected_obj_df = objs_df_dict[i+1].copy()

    ymin = selected_obj_df.min(axis=1).min()
    ymax = selected_obj_df.max(axis=1).max()
    
    selected_obj_name = obj_names_full[i]
    axes[i].plot([0, len(util_names)], [0, 0], color=colors[2], linestyle='--', linewidth=2)  # plot horizontal line

    for p in range(len(policy_names)-1):
        x_vals = x_pos[p]
        for x, height in zip(x_vals, selected_obj_df.iloc[p,:]):
            axes[i].plot([x, x], [0, height], color=colors[2], linewidth=2)  # plot vertical lines
            axes[i].plot(x, height, 'o', color=colors[p], markersize=12, 
                         markeredgecolor=colors[2], markeredgewidth=2)   # plot points
        
    axes[i].set_ylabel(r'$\longleftarrow$ % degradation', fontsize=14, fontdict={'fontname': 'Verdana'})
    
    axes[i].set_xticks(np.linspace(0.15,len(util_names)-0.8, len(util_names)))
    axes[i].set_xticklabels(['O', 'D', 'C', 'R', 'Ch', 'P'], fontsize=13, 
                             fontdict={'fontname': 'Verdana'})
    axes[i].xaxis.set_visible(True)

    # turn off top, bottom, and right spines
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    # turn on vertical gridlines
    axes[i].xaxis.grid(True)

    axes[i].set_title(obj_names_full[i], fontsize=16, y=1.1, pad = 1, 
                      fontdict={'fontname': 'Verdana'})

handles = [plt.Line2D([], [], color=colors[0], marker='o', markersize=12, linestyle='None'),
            plt.Line2D([], [], color=colors[1], marker='o', markersize=12, linestyle='None'),
            plt.Line2D([], [], color=colors[2], marker='None', linestyle='--', linewidth=3)]

labels = ['Durham-focused', 'Raleigh-focused', 'Regionally robust']
plt.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(-0.35, -0.3), 
           fontsize=14, frameon=False)

plt.savefig(figname, dpi=300, bbox_inches='tight')
