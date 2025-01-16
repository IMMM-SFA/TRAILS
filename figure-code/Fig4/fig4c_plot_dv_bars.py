
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dvs_filename = 'refset_DVs_headers_abbrev.csv'
dvs_df = pd.read_csv(dvs_filename, header=0, index_col=0)
selected_rows = [92, 132, 140]
colors = ['#DC851F', '#48A9A6','#355544']
util_names =  ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_abbrevs = ['O', 'D', 'C', 'R', 'Ch', 'P']

# get dvs for each utility
dv_o = 1-dvs_df[['O_RT', 'O_TT', 'O_InfT']]
dv_o = dv_o.loc[selected_rows]
dv_d = 1-dvs_df[['D_RT', 'D_TT', 'D_InfT']]
dv_d = dv_d.loc[selected_rows]
dv_c = 1-dvs_df[['C_RT', 'C_InfT']]
dv_c = dv_c.loc[selected_rows]
dv_r = 1-dvs_df[['R_RT', 'R_TT', 'R_InfT']]
dv_r = dv_r.loc[selected_rows]
dv_ch = 1-dvs_df[['Ch_RT', 'Ch_TT', 'Ch_InfT']]
dv_ch = dv_ch.loc[selected_rows]
dv_p = 1-dvs_df[['P_RT', 'P_TT', 'P_InfT']]
dv_p = dv_p.loc[selected_rows]

# plot dv bar plots 
fig, axes = plt.subplots(1, len(util_names), figsize=(16, 3.5))
plt.subplots_adjust(wspace=0.1, hspace=0.5)

for util in range(len(util_names)):
    dv_to_plot = np.array([])
    x = np.arange(3)
    if util == 0:
        dv_to_plot = dv_o.T
    elif util == 1:
        dv_to_plot = dv_d.T
    elif util == 2:
        dv_to_plot = dv_c.T
    elif util == 3:
        dv_to_plot = dv_r.T
    elif util == 4:
        dv_to_plot = dv_ch.T
    elif util == 5:
        dv_to_plot = dv_p.T
    
    dv_to_plot.plot.barh(ax = axes[util], title=util_names[util], color=colors, legend=False)
    if util > 0:
        axes[util].set_yticklabels([])
    
    # Turn off top and bottom spines
    axes[util].spines['top'].set_visible(False)
    axes[util].spines['right'].set_visible(False)

    axes[util].set_xticks([0, 0.5, 1.0])
    axes[util].set_xticklabels(['1.0', '0.5', '0.0'])
    axes[util].set_xlim([0, 1.0])
    axes[util].set_xlabel(r'Increased use $\longrightarrow$')
plt.savefig('dv_bars_final.pdf')
#plt.show()


