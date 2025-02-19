import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from robustness_temporal_diagnostics_functions import boosted_trees_factor_ranking, \
    plot_timevarying_factor_contributions,\
    boosted_trees_factor_ranking_consequential,\
    plot_timevarying_gbr_shap,\
    boosted_trees_factor_ranking_with_shap_consequential

sns.set_style('white')

# set matplotlib font type
plt.rcParams['font.family'] = 'verdana'

NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_RDM = 1000  # number of DU SOWs

all_util_nums = np.arange(3,4)
sol_num = 92
util_list = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
#util_list = ['Regional']

'''
rdm_headers = ['Demand1', 'Demand2', 'Demand3', 'Bond Term', 
    'Bond Int', 'Discount', 'Rest Eff', 'Evap mult', 'JLWTP Permit', 
    'JLWTP Const', 'Inflow Amp', 'Inflow freq', 'Inflow phase']
'''
rdm_headers = ['5-year demand', '20-year demand', '45-year demand', 'Bond term', 
    'Bond interest', 'Discount rate', 'Restriction Efficiency', 'Evaporation intensity', 'Permitting time', 
    'Construction time', 'Inflow variation', 'Inflow return period', 'Inflow seasonality']

window_size = 52  # one year rolling window
robustness = np.zeros([1, NUM_WEEKS - window_size + 1], dtype=float)
du_factors = pd.read_csv('RDM_inputs_final_combined_headers.csv', index_col=None)
du_factor_names = du_factors.columns.values.tolist()

for util_num in all_util_nums:
    print(f'Importing satisficing, robustness, and period data for {util_list[util_num]} in solution {sol_num}...')

    # import satisficing data
    satisficing_filename = f'output_files/satisficing_sol{sol_num}_util{util_num}.csv'
    periods_filename = f'consequential_periods/periods_sol{sol_num}_u{util_num}_new.csv'
    crit_periods_filename = f'critical_periods/periods_sol{sol_num}.csv'
    satisficing_np = np.loadtxt(satisficing_filename, delimiter=',').astype(int)[:, window_size:]
    
    robustness_np = np.sum(satisficing_np, axis=0)/NUM_RDM
    periods_df = pd.read_csv(periods_filename, index_col=None, header=0)
    crit_periods_df = pd.read_csv(crit_periods_filename, index_col=None, header=0)
    
    period_selected = periods_df
    
    # calculate factor contributions over time 
    all_params = du_factors.values
    param_names = rdm_headers

    # check the length of the periods_df
    if period_selected.shape[0] == 0:
        print(f'No consequential periods detected for {util_list[util_num]}. Skipping this utility...')
        continue
    else:
        shap_list = boosted_trees_factor_ranking_with_shap_consequential(satisficing_np, all_params[:NUM_RDM,:], 
                                                                        param_names, period_selected, window_size, sol_num, 
                                                                        util_list, util_num, lag=0, period_name = 'conseq')