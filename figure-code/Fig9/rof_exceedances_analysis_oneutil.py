import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itsa_functions import find_input_ranges, plot_itsa_figure, itsa_index_alltimesteps, calc_cf_diff, normalize_inputs

print('Setting up SA analysis...')
sns.set_style('white')

NUM_WEEKS = 2344
NUM_REALS = 1000   # to change
NUM_RDMS = 1000

sol_nums = [92, 132, 140]
start_t_arr = [429, 215, 245]
end_t_arr = [1221, 487, 715]
window_size = 52

sol_selected = 2
util_num = 3
selected_du = 287
t = 2
s = 'p'
output_bins = 2
input_bins = 20

SOL_NUM = sol_nums[sol_selected]
start_t = start_t_arr[sol_selected]
end_t = end_t_arr[sol_selected]

dv_abbrevs_all = ['RT', 'TT', 'InfT']
dv_abbrevs_cary = ['RT', 'InfT']

output_nums = np.arange(len(dv_abbrevs_all))

output_names_all = ['Restrictions', 'Transfers', 'Infrastructure']
output_names_cary = ['Restrictions', 'Infrastructure']

dv_abbrev_dict = {'OWASA': dv_abbrevs_all, 'Durham': dv_abbrevs_all, 'Cary': dv_abbrevs_cary,
                  'Raleigh': dv_abbrevs_all, 'Chatham': dv_abbrevs_all, 'Pittsboro': dv_abbrevs_all}
output_num_dict = {0: 'Restrictions', 1: 'Transfers', 2: 'Infrastructure'}
rof_exceed_dict = {'Restrictions': 'rt_exceed', 'Transfers': 'tt_exceed', 'Infrastructure': 'inf_exceed'}
rof_abbrev_dict = {'Restrictions': 'st_rof', 'Transfers': 'st_rof', 'Infrastructure': 'lt_rof'}
trigger_dict = {'Restrictions': 'RT', 'Transfers': 'TT', 'Infrastructure': 'InfT'}
output_plot_dict = {'Restrictions': 0, 'Transfers': 1, 'Infrastructure': 2}

#du_abbrevs = dv_abbrevs_all
output_names = output_names_all

# get first-order and total-effect sensitivity indices for the utility
input_names = ['OWASA Storage', 'Durham Storage', 'Cary Storage',
            'Raleigh Storage', 'Chatham Storage', 'Pittsboro Storage',
            'OWASA Proj. Demand', 'Durham Proj. Demand', 'Cary Proj. Demand',
            'Raleigh Proj. Demand', 'Chatham Proj. Demand', 'Pittsboro Proj. Demand',
            'OWASA Capacity', 'Durham Capacity', 'Cary Capacity',
            'Raleigh Capacity', 'Chatham Capacity', 'Pittsboro Capacity']

decision_variables = pd.read_csv('refset_DVs_simplified.csv', index_col=0, header=0)
dv_selected = decision_variables.loc[SOL_NUM]

# list the utilities 
util_names = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_abbrev = ['O', 'D', 'C', 'R', 'Ch', 'P']

# Load the utilities' DU SOWs 
print('Loading the HDF files')
hdf_folder = f'../sol{SOL_NUM}_hdf_packed/Utilities_s{SOL_NUM}_RDM{selected_du}.h5'

du_deets_hdf = pd.HDFStore(hdf_folder, mode='r')

# iterate over dataframe keys in the HDF5 file 
print('Getting all realizations...')
dataframes = []
for key in du_deets_hdf.keys():
    dataframes.append(du_deets_hdf[key].copy())

du_deets_hdf.close()

# concatenate all dataframes vertically into one dataframe
du_deets_df = pd.concat(dataframes, ignore_index=True)
du_rof_df = pd.DataFrame()

# convert to numpy arrays then perform rolling mean and normalize
#storage_cols = [str(u) + 'st_vol' for u in np.arange(len(util_names))]  # storage for all utilities
inf_cols = [str(u) + 'st_vol' for u in np.arange(len(util_names))]  # infrastructure for all utilities
#rdem_cols = [str(u) + 'rest_demand' for u in np.arange(len(util_names))]  # demand for all utilities
dem_cols = [str(u) + 'proj_dem' for u in np.arange(len(util_names))]  # demand for all utilities
cap_cols = [str(u) + 'capacity' for u in np.arange(len(util_names))]  # capacity for all utilities
#unrest_dem = [str(u) + 'unrest_demand' for u in np.arange(len(util_names))]  # demand for all utilities

#obs_dem_cols = [str(u) + 'obs_ann_dem' for u in np.arange(len(util_names))]  # observed demand for all utilities
#proj_dem_cols = [str(u) + 'proj_dem' for u in np.arange(len(util_names))]  # projected demand for all utilities
#dem_cols = [str(util_num) + 'proj_dem' for u in np.arange(len(util_names))]  # demand for all utilities
#rev_cols = [str(u) + 'gross_rev' for u in np.arange(len(util_names))]  # net inflow for all utilities
#cf_cols= [str(u) + 'cont_fund' for u in np.arange(len(util_names))]  # debt service for all utilities

statevar_cols = inf_cols + dem_cols + cap_cols

print('Processing the state variables...')
# extract input (state variables) and output (ROF exceedances) dataframes
du_input = du_deets_df[statevar_cols].copy()
du_input_arr = du_input.values

# reshape the input and output arrays to be 3D
du_input_mat = np.zeros((NUM_WEEKS, NUM_REALS, len(statevar_cols)))

for i in range(NUM_REALS):
    du_input_mat[:,i:i+1,:] = du_input_arr[i*NUM_WEEKS:(i+1)*NUM_WEEKS,:].reshape(NUM_WEEKS, 1, len(statevar_cols))
    '''
    for u_num in range(len(util_names)):
        du_input_mat[:,i:i+1,12+u_num] = du_input_mat[:,i:i+1,12+u_num] / du_input_mat[:,i:i+1,18+u_num]
    '''
# delete the last six columns of the input matrix
# du_input_mat = np.delete(du_input_mat, np.s_[18:24], axis=2)

colors = ['#87CEEB', '#6495ED', '#4169E1', '#000080', '#47B5A8', '#008080', 
          '#FFDAB9', '#F28500', '#B7410E', '#662307', '#D08B5A', '#7B3F00',
          '#D3D3D3', '#708090', '#B2BEB5', '#4D4A53', '#C0C0C0', '#36454F']

# calculate restriction, transfer, and infrastructure trigger threshold exceedances for the utility
print('Calculating trigger threshold exceedances...')
#fig = plt.figure(figsize=(12, 3))
#gs = fig.add_gridspec(1, 3, hspace=0.6, wspace=0.3, height_ratios=[1], width_ratios=[0.3, 0.3, 0.3])
#ax1 = fig.add_subplot(gs[0, 0])
#ax2 = fig.add_subplot(gs[0, 1])
#ax3 = fig.add_subplot(gs[0, 2])
#axs = [ax1, ax2, ax3]

# plot the short-term ROF
fig = plt.figure(figsize=(6, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1,1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

srof_to_plot = f'{util_num}st_rof'
lrof_to_plot = f'{util_num}lt_rof'

rt_to_plot = f'{util_abbrev[util_num]}_RT'
tt_to_plot = f'{util_abbrev[util_num]}_TT'
inf_to_plot = f'{util_abbrev[util_num]}_InfT'

srof_arr = du_deets_df[srof_to_plot].values
lrof_arr = du_deets_df[lrof_to_plot].values

# reshape the srof and lrof arrays to be 3D
srof_mat = np.zeros([NUM_WEEKS, NUM_REALS])
lrof_mat = np.zeros([NUM_WEEKS, NUM_REALS])

for i in range(NUM_REALS):
    srof_mat[:,i:i+1] = srof_arr[i*NUM_WEEKS:(i+1)*NUM_WEEKS].reshape(NUM_WEEKS, 1)
    lrof_mat[:,i:i+1] = lrof_arr[i*NUM_WEEKS:(i+1)*NUM_WEEKS].reshape(NUM_WEEKS, 1)

srof_max = np.mean(srof_mat, axis=1)
lrof_max = np.mean(lrof_mat, axis=1)

print(f'mean of srof_max: {np.mean(srof_max)}') 
print(f'Size of srof_max: {len(srof_max)}')  # should be 2344
print(f'mean of lrof_max: {np.mean(lrof_max)}')
print(f'Size of lrof_max: {len(lrof_max)}')  # should be 2344

# reshape the short-term ROF dataframe
ax1.fill_between(np.arange(start_t, end_t+(52*5)), 0, srof_max[start_t:end_t+(52*5)], label=util_names[util_num], color='gainsboro', alpha=0.7)
ax1.axhline(dv_selected[rt_to_plot], color='#C1666B', linestyle='--', label=f'{util_names[util_num]} RT', linewidth=2)

if util_num != 2:
    ax1.axhline(dv_selected[tt_to_plot], color='#03312E', linestyle='--', label=f'{util_names[util_num]} TT', linewidth=2)
# remove right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# label the subplot
ax1.set_ylabel('Short-term ROF', fontsize=12)
ax1.legend(loc='upper right', fontsize=10, frameon=False)
ax1.set_title(f'{util_names[util_num]} Short-term ROF exceedances', fontsize=12)
ax1.set_ylim(0, 1)

ax2.fill_between(np.arange(start_t, end_t+(52*5)), 0, lrof_max[start_t:end_t+(52*5)], label=util_names[util_num], color='gainsboro', alpha=0.7)
#ax2.plot(np.arange(start_t, end_t), du_deets_df[lrof_to_plot].values[start_t:end_t], color='black', label=util_names[util_num], linewidth=1)
# fill between 0 and the trigger threshold
ax2.axhline(dv_selected[inf_to_plot], color='#F2C53D', linestyle='--', label=f'{util_names[util_num]}InfT', linewidth=2)
# remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xticklabels(['']*len(np.arange(start_t, end_t+(52*5))))
# label the subplot
num_years = np.arange(start_t, end_t+(52*5), 52*5)
num_years_labels = num_years / 52
# round up labels to nearest integer
num_years_labels = np.ceil(num_years_labels).astype(int)

ax2.set_xlabel('Years', fontsize=12)
ax2.set_xticks(num_years)
ax2.set_xticklabels(num_years_labels)
ax2.set_ylabel('Long-term ROF', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', fontsize=10, frameon=False)
ax2.set_title(f'{util_names[util_num]} Long-term ROF exceedances', fontsize=12)

plt.savefig(f'rof_exceedances/s{SOL_NUM}_u{util_num}_du{selected_du}_t{t}_{s}.jpg', dpi=300, bbox_inches='tight') 

'''
fig = plt.figure(figsize=(12, 3))
gs = fig.add_gridspec(1, 3, hspace=0.6, wspace=0.3, height_ratios=[1], width_ratios=[0.3, 0.3, 0.3])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
axs = [ax1, ax2, ax3]

for output_num in range(len(output_nums)):
    action_name = output_num_dict[output_num]
    axs_idx = output_plot_dict[action_name]
    dv_exceed = rof_exceed_dict[action_name]
    dv_rof = rof_abbrev_dict[action_name]
    trigger = trigger_dict[action_name]

    window_size = 52

    if action_name == 'Infrastructure':
        window_size = 78

    # Adjust the height ratios of the subplots
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    legend_labels = []

    for l in range(len(input_names)):
        input = plt.Rectangle((0,0),1,1,fc=colors[l], edgecolor='none')
        legend_labels.append(input)

    #for util_num in range(len(util_names)): 
    util_to_plot = util_num
    util_name = util_names[util_num]
    util_rof = f'{util_num}{dv_rof}'
    util_exceed = f'{util_num}{dv_exceed}'
    util_trigger = f'{util_abbrev[util_num]}_{trigger}'
    #du_abbrevs = dv_abbrev_dict[util_name]

    if util_exceed == '2tt_exceed':
        continue

    du_rof_df[util_exceed] = du_deets_df[util_rof] - dv_selected[util_trigger]
    
    # convert all negative values to 0
    du_rof_df[du_rof_df <= 0] = 0
    # convert all positive values to 1
    du_rof_df[du_rof_df > 0] = 1
        
    #output_col = f'{util_num}{rof_abbrev_dict[action_name]}'
    
    du_output_arr = du_rof_df[util_exceed].values

    du_output_mat = np.zeros((NUM_WEEKS, NUM_REALS))
    #du_output_rolling_mat = np.zeros((NUM_WEEKS - window_size + 1, NUM_REALS))

    for i in range(NUM_REALS):  
        #du_output_mat[:,i:i+1] = du_output_arr[i*NUM_WEEKS:(i+1)*NUM_WEEKS,:].reshape(NUM_WEEKS)
        du_output_mat[:,i:i+1] = du_output_arr[i*NUM_WEEKS:(i+1)*NUM_WEEKS].reshape(NUM_WEEKS,1)
        #du_output_rolling_mat[:,i] = np.convolve(du_output_mat[:,i], np.ones(window_size)/window_size, mode='valid')
        
        #for s in range(len(statevar_cols)):
            #du_input_rolling_mat[:,i, s] = np.convolve(du_input_mat[:,i,s], np.ones(window_size)/window_size, mode='valid')
    
    print('Calculating ITSA indices...')

    if util_num == 2 and action_name == 'Transfers':
        continue
    else:
        itsa_toplot = itsa_index_alltimesteps(du_input_mat, 
                                        du_output_mat, 
                                        input_names, action_name, 
                                        NUM_WEEKS - window_size + 1, util_num,
                                        SOL_NUM, selected_du, y_bins=output_bins, x_bins=input_bins)
            
        print('Plotting sensitivity indices...')
        plot_itsa_figure(util_num, itsa_toplot, input_names, action_name, 
                        window_size, axs[axs_idx], colors, start_t, end_t+(52*5))
        

        S1_restr, ST_restr =  delta_all_timesteps(input_bounds, 
                                                du_input_rolling,
                                                du_output_rolling[:,output_num], 
                                                input_names, output_names[output_num], 
                                                NUM_WEEKS - window_size + 1, util_num)


        # itsa_toplot = pd.DataFrame(itsa_restr_norm, columns=input_names)

        plot_tvsa_figure(S1_restr, ST_restr, NUM_WEEKS-window_size+1, input_names, output_names, colors,
                        util_names[util_num], output_names[output_num], window_size)
        
        plot_itsa_heatmap(itsa_toplot, NUM_WEEKS-window_size+1, 
                        input_names, custom_cmap, 
                        util_names[util_num], output_names[output_num], 
                        window_size, SOL_NUM, selected_du)

plt.figlegend(legend_labels,\
                input_names,\
                loc='lower center', ncol=3, fontsize=8, frameon=False,
                bbox_to_anchor=(0.5, -0.45), title='State variables')
plt.setp(legend_labels, alpha=0.8)

plt.suptitle(f"ITSA indices for {util_names[util_num]} in DU {selected_du}", 
                fontsize=10)
    
plt.savefig(f'figures/fillplot_s{SOL_NUM}_du{selected_du}_u{util_num}_{s}_t{t}_2bins.pdf', dpi=300, bbox_inches='tight')
'''
