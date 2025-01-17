# Load all necessary files
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time 
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

'''
Change these values to desired simulation time, number of realizations, and number of DU SOWs
'''
NUM_YEARS = 45
NUM_WEEKS_PER_YEAR = 52
NUM_WEEKS = 2344
NUM_REAL = 10  # number of hydroclimatic realizations
NUM_RDM = 10  # number of DU SOWs

'''
FUNCTIONS TO CALCULATE TIME-VARYING PERFORMANCE METRICS
'''
def calculate_sol_reliability(util_num, window_size, sol_num, RDM, skip_weeks = 1):
    '''
    Calculates reliability of a solution over time in a given DU SOW using the ratio of 
    available storage to total storage in each week.
    
    Data comes from the utility file.
    
    :param st_vol_col: int, column number of currently-available storage volume in utility file
    :param capacity_col: int, column number of storage capacity in utility file
    :param util_num: int, utility ID number
    :param RDM: int, DU SOW number

    :return fail_sum: numpy array of dimensions (NUL_REALS x NUM_YEARS),
        total number of "failures" (weeks with storage < 20% of capacity) in a given DU SOW over time
        calculated using an annual rolling window
    :return reliability: numpy array of dimensions (1 x NUM_YEARS), 
        reliability of solution in a given DU SOW over time 
    '''

    fail_counts = np.zeros([NUM_REAL, NUM_WEEKS])

    # Load the HDF5 file
    hdf5_file = pd.HDFStore(f'../Step1/sol{sol_num}_hdf_packed/Utilities_s{sol_num}_RDM{RDM}.h5', 'r') # change this depending on where the HDF5 files are stored

    st_vol_col = str(util_num) + 'st_vol'
    capacity_col = str(util_num) + 'capacity'
    num_windows = int(np.floor((NUM_WEEKS - window_size)/skip_weeks) + 1)
    fail_sum = np.zeros([NUM_REAL, num_windows])

    # Populate fail_counts and fail_sum tensors
    for rel in range(0, NUM_REAL):
        key = f'r{rel}'
        cur_rel = hdf5_file[key]
        fail_idx = np.where(cur_rel[st_vol_col] / cur_rel[capacity_col] < 0.2)[0]
        fail_counts[rel, fail_idx] = 1
    
    hdf5_file.close()
    # Calculate fail_sum using sliding window approach
    for w in range(0, NUM_WEEKS - window_size + 1):
        fail_sum[:, w] = np.sum(fail_counts[:, w:w+window_size], axis=1)

    # calculate reliability over time
    reliability = 1 - (np.sum(fail_sum, axis=0)/NUM_REAL)
    return reliability

def calculate_restriction_frequency(util_num, window_size, sol_num, RDM, skip_weeks=1):
    '''
    Calculates restriction frequency of a solution over time in a given DU SOW using the restriction multiplier 
    over a certain window of time.

    Restriction frequency: Fraction of weeks in a given year that the restriction multiplier is less than 1.

    :param util_num: int, utility ID number
    :param window_size: int, size of rolling window to use for calculating average restriction frequency
    :param sol_num: int, solution number
    :param RDM: int, DU SOW number

    :return restriction_freq_windowed: numpy array of dimensions (NUM_REALS x NUM_WEEKS - window_size + 1), restriction frequency 
        in each realization calculated across a moving window
    :return restriction_freq_mean: numpy array of dimensions (1 x NUM_WEEKS - window_size + 1), avg restriction frequency
        across all realizations calculated across a moving window
    '''
    num_windows = int(np.floor((NUM_WEEKS - window_size)/skip_weeks) + 1)

    restricted_weeks = np.zeros([NUM_REAL, NUM_WEEKS])
    restriction_freq_windowed = np.zeros([NUM_REAL, num_windows])
    col_name = str(util_num) + 'rest_m'

    # Load the HDF5 file
    hdf5_file = pd.HDFStore(f'../Step1/sol{sol_num}_hdf_packed/Policies_s{sol_num}_RDM{RDM}.h5', 'r') # change this depending on where the HDF5 files are stored

    for rel in range(NUM_REAL):
        key = f'r{rel}'
        cur_rel = hdf5_file[key]

        mask = np.where(cur_rel[col_name] != 1)[0]
        restricted_weeks[rel, mask] = 1

    hdf5_file.close()

    for week in range(0, NUM_WEEKS - window_size + 1):
        restriction_freq_windowed[:, week] = np.sum(restricted_weeks[:, week:week + window_size], axis=1)  # calculate number of restricted weeks in the window
    
    # calculate rf over time
    restriction_freq_mean = np.sum(restriction_freq_windowed, axis=0)/NUM_REAL
    
    return restriction_freq_windowed, restriction_freq_mean

def calculate_peak_financial_costs(util_num, window_size, sol_num, RDM):
    '''
    Calculates avg financial costs of a solution within a given window of time over time across all realizations in a given DU SOW 
    :param sol_num: int, solution number
    :param RDM: int, DU SOW number
    :ds_payments_col: int, column number of debt service payments in utility file
    :cf_contrib_col: int, column number of the total contingency fund balance in utility file
    :gross_rev_col: int, column number of gross revenue in utility file
    '''
    ds_payments_col = str(util_num) + 'debt_serv'
    cf_contrib_col = str(util_num) + 'cont_fund'
    gross_rev_col = str(util_num) + 'gross_rev'
    insur_price_col = str(util_num) + 'ins_price'

    financial_costs = np.zeros([NUM_REAL, NUM_WEEKS-window_size+1], dtype=float)
    
    # Creates a table with years that failed in each realization.
    allreals_year_debt_payment = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_year_cont_fund_contrib = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_year_gross_revenue = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_year_insurance_contract_cost = np.zeros([NUM_REAL, NUM_WEEKS])

    # Load the HDF5 file
    hdf5_file = pd.HDFStore(f'../Step1/sol{sol_num}_hdf_packed/Utilities_s{sol_num}_RDM{RDM}.h5', 'r') # change this depending on where the HDF5 files are stored
    max_pfc = np.zeros(NUM_REAL)

    for rel in range(NUM_REAL):
        key = f'r{rel}'
        cur_rel_df = hdf5_file[key]

        allreals_year_debt_payment[rel,:] = cur_rel_df[ds_payments_col]   # 0debt_serv
        realizations_year_cont_fund_totals = cur_rel_df[cf_contrib_col]   # 0cont_fund
        allreals_year_cont_fund_contrib[rel, 1:] = np.diff(realizations_year_cont_fund_totals) # calculate contributions to the contingency fund
        allreals_year_insurance_contract_cost[rel,:] = cur_rel_df[insur_price_col]   # 0ins_price
        allreals_year_gross_revenue[rel,:] = cur_rel_df[gross_rev_col]  # 0gross_rev
        
    for w in range(0,NUM_WEEKS-window_size+1):
        realizations_year_debt_payment_sum = np.sum(allreals_year_debt_payment[:,w:w+window_size], axis=1)
        realizations_year_cont_fund_contrib_sum = np.sum(allreals_year_cont_fund_contrib[:,w:w+window_size], axis=1)
        realizations_year_insurance_contract_cost_sum = np.sum(allreals_year_insurance_contract_cost[:, w:w+window_size], axis=1)
        realizations_year_gross_revenue_sum = np.sum(allreals_year_gross_revenue[:, w:w+window_size], axis=1)

        # Calculate the sums efficiently
        financial_costs[:, w] = (realizations_year_debt_payment_sum + realizations_year_cont_fund_contrib_sum + realizations_year_insurance_contract_cost_sum) / realizations_year_gross_revenue_sum

    hdf5_file.close()

    # take max value within each window across all realizations
    max_pfc = np.max(financial_costs, axis=0)
    np.nan_to_num(max_pfc, copy=False, nan=0.0)
    
    return max_pfc


def calculate_worst_case_cost(util_num, window_size, sol_num, RDM):
    '''
    Calculates the worst case cost of a solution within a given window of time over time across all realizations in a given DU SOW
    :param sol_num: int, solution number
    :param RDM: int, DU SOW number
    :util_num: int, utility ID number
    :window_size: int, size of rolling window to use for calculating average worst case cost

    :return worst_case_cost: numpy array of dimensions (NUM_REALS x NUM_WEEKS - window_size + 1), the avg worst case cost in each realization
    
    '''
    dm_cost = str(util_num) + 'dm_cost'
    cf_value = str(util_num) + 'cont_fund'
    avg_revenue = str(util_num) + 'gross_rev'

    rolling_window_financial_costs = np.zeros([NUM_REAL, NUM_WEEKS - window_size + 1])
    financial_costs = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_dm_cost = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_cf_value = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_avg_revenue = np.zeros([NUM_REAL, NUM_WEEKS])

    # Load the HDF5 file
    hdf5_file = pd.HDFStore(f'../Step1/sol{sol_num}_hdf_packed/Utilities_s{sol_num}_RDM{RDM}.h5', 'r') # change this depending on where the HDF5 files are stored
    
    for rel in range(NUM_REAL):
        key = f'r{rel}'
        cur_rel_df = hdf5_file[key]

        allreals_dm_cost[rel, :] = cur_rel_df[dm_cost]   # 0dm_cost
        allreals_cf_value[rel, :] = cur_rel_df[cf_value]   # 0cont_fund
        allreals_avg_revenue[rel, :] = cur_rel_df[avg_revenue]  # 0gross_rev
    hdf5_file.close()
    
    for w in range(0,NUM_WEEKS-window_size+1):
        dm_cost = np.sum(allreals_dm_cost[:, w:w+window_size], axis=1)
        cf_value = np.sum(allreals_cf_value[:, w:w+window_size], axis=1)

        financial_costs_num = dm_cost - cf_value
        financial_costs_num[financial_costs_num < 0] = 0
        financial_costs_denom = np.sum(allreals_avg_revenue[:, w:w+window_size], axis=1)

        rolling_window_financial_costs[:, w] = np.divide(financial_costs_num, financial_costs_denom)
    
    worst_case_cost = np.percentile(rolling_window_financial_costs, 99, axis=0)
    np.nan_to_num(worst_case_cost, copy=False, nan=0.0)
    
    return worst_case_cost

def calculate_unit_cost(util_num, window_size, sol_num, RDM):
    '''
    Calculates the unit cost of the infrastructure choices built by a uility over time 
    in a given DU SOW using the unit cost column in the utility file.

    :param util_num: int, utility ID number
    :param window_size: int, size of rolling window to use for calculating average unit cost
    :param sol_num: int, solution number
    :param RDM: int, DU SOW number

    :return unit_cost: numpy array of dimensions (NUM_REALS x NUM_WEEKS - window_size + 1), the avg unit cost in each realization
    '''
    demand_col = str(util_num) + 'unrest_demand'
    init_demand_col = str(util_num) + 'obs_ann_dem'
    investment_cost = np.zeros([NUM_REAL, NUM_WEEKS])
    window_investment_cost = np.zeros([NUM_REAL, NUM_WEEKS - window_size + 1])

    allreals_debt_payment = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_demand = np.zeros([NUM_REAL, NUM_WEEKS])
    allreals_obs_ann_dem = np.zeros(NUM_REAL)

    # Load the HDF5 file
    hdf5_file = pd.HDFStore(f'../Step1/sol{sol_num}_hdf_packed/Utilities_s{sol_num}_RDM{RDM}.h5', 'r') # change this depending on where the HDF5 files are stored

    pv_ds_payments_col = str(util_num) + 'pv_debt_serv'
    demand_col = str(util_num) + 'unrest_demand'
    init_demand_col = str(util_num) + 'obs_ann_dem'

    investment_cost = np.zeros([NUM_REAL, NUM_WEEKS])
    window_investment_cost = np.zeros([NUM_REAL, NUM_WEEKS - window_size + 1])

    for rel in range(NUM_REAL):
        key = f'r{rel}'
        cur_rel_df  = hdf5_file[key]
        allreals_debt_payment[rel, :] = cur_rel_df[pv_ds_payments_col]   # 0pv_debt_serv
        allreals_demand[rel, :] = cur_rel_df[demand_col]
        allreals_obs_ann_dem[rel] = cur_rel_df.iloc[0, cur_rel_df.columns.get_loc(init_demand_col)]

    hdf5_file.close()

    # Calculate the sums efficiently
    for w in range(0,NUM_WEEKS-window_size+1):
        demand = np.sum(allreals_demand[:, w:w+window_size], axis=1)
        obs_ann_dem = allreals_obs_ann_dem
        investment_cost_num = np.sum(allreals_debt_payment[:, w:w+window_size], axis=1)

        investment_cost_denom = demand - (obs_ann_dem*window_size)
        investment_cost_denom[investment_cost_denom <= 0] = 1  # if the denominator < 0, set it to 1 (utility doesn't request more than historically needed)
        window_investment_cost[:, w] = np.divide(investment_cost_num, investment_cost_denom)
        
    unit_cost = np.mean(window_investment_cost, axis=0)*1000    
    np.nan_to_num(window_investment_cost, copy=False, nan=0.0)

    return unit_cost

'''
SET UP PARALLEL PROCESSING
Values will change depending on cluster/machine architecture.
'''
OMP_NUM_THREADS = 16

N_NODES = 1
N_TASKS_PER_NODE = 10
N_TASKS = N_NODES * N_TASKS_PER_NODE  # should be 10
window_size = 52
util_list = ['OWASA', 'Durham', 'Cary', 'Raleigh', 'Chatham', 'Pittsboro']
util_num = 5  # change index to desired utility, currently set at Pittsboro (5)
SOL_NUM = 140

# Calculate the number of rows each process will handle
rows_per_process = int(NUM_RDM / size)
start_row = rank * rows_per_process
end_row = start_row + rows_per_process
num_rows = end_row - start_row

# Create output matrices to store time-varying performance metrics
rel_allDU = np.zeros([num_rows, NUM_WEEKS - window_size + 2])
rf_allDU = np.zeros([num_rows, NUM_WEEKS - window_size + 2])
pfc_allDU = np.zeros([num_rows, NUM_WEEKS - window_size + 2])
wcc_allDU = np.zeros([num_rows, NUM_WEEKS - window_size + 2])
uc_allDU = np.zeros([num_rows, NUM_WEEKS - window_size + 2])

for du in range(num_rows):
    curr_du = (rank*rows_per_process) + du
    print("Curr DU: ", curr_du)

    # reliability
    rel_allDU[du, 1:] = calculate_sol_reliability(util_num, window_size, SOL_NUM, curr_du)
    rel_allDU[du, 0] = curr_du
    
    # restriction frequency
    rf_win, rf_mean = calculate_restriction_frequency(util_num, window_size, SOL_NUM, curr_du)
    rf_allDU[du, 1:] = rf_mean
    rf_allDU[du, 0] = int(curr_du)
    
    # peak financial costs
    pfc_allDU[du, 1:] = calculate_peak_financial_costs(util_num, window_size, SOL_NUM, curr_du)
    pfc_allDU[du, 0] = int(curr_du)
    
    # worst case cost
    wcc_allDU[du, 1:] = calculate_worst_case_cost(util_num, window_size, SOL_NUM, curr_du)
    wcc_allDU[du, 0] = int(curr_du)
   
    # unit cost
    uc_allDU[du, 1:] = calculate_unit_cost(util_num, window_size, SOL_NUM, curr_du)
    uc_allDU[du, 0] = int(curr_du)

all_rel_out = comm.gather(rel_allDU, root=0)
all_rf_out = comm.gather(rf_allDU, root=0)

all_pfc_out = comm.gather(pfc_allDU, root=0)
all_wcc_out = comm.gather(wcc_allDU, root=0)
all_uc_out = comm.gather(uc_allDU, root=0)

# set output file names
file_rel_outname = f"tv_objs/sol{SOL_NUM}_util{util_num}_REL.csv"
file_rf_outname = f"tv_objs/sol{SOL_NUM}_util{util_num}_RF.csv"
file_pfc_outname = f"tv_objs/sol{SOL_NUM}_util{util_num}_PFC.csv"
file_wcc_outname = f"tv_objs/sol{SOL_NUM}_util{util_num}_WCC.csv"
file_uc_outname = f"tv_objs/sol{SOL_NUM}_util{util_num}_UC.csv"

if rank == 0:

    # combine all outputs across all processes
    combined_rel_out = np.concatenate(all_rel_out, axis=0)
    combined_rf_out = np.concatenate(all_rf_out, axis=0)
    combined_pfc_out = np.concatenate(all_pfc_out, axis=0)
    combined_wcc_out = np.concatenate(all_wcc_out, axis=0)
    combined_uc_out = np.concatenate(all_uc_out, axis=0)

    # save outputs to csv
    np.savetxt(file_rel_outname, combined_rel_out, delimiter=',')
    np.savetxt(file_rf_outname, combined_rf_out, delimiter=',')
    np.savetxt(file_pfc_outname, combined_pfc_out, delimiter=',')
    np.savetxt(file_wcc_outname, combined_wcc_out, delimiter=',')
    np.savetxt(file_uc_outname, combined_uc_out, delimiter=',')

    print("Done!")
