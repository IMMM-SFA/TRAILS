import numpy as np 
import pandas as pd

def calc_satisficing_regional(sol_num):
    # Read the data
    rel_timeseries_filename = f'tv_objs/sol{sol_num}_regional_REL.csv'  
    rf_timeseries_filename = f'tv_objs/sol{sol_num}_regional_RF.csv'
    pfc_timeseries_filename = f'tv_objs/sol{sol_num}_regional_PFC.csv'
    wcc_timeseries_filename = f'tv_objs/sol{sol_num}_regional_WCC.csv'
    uc_timeseries_filename = f'tv_objs/sol{sol_num}_regional_UC.csv'

    rel_timeseries = np.loadtxt(rel_timeseries_filename, delimiter=',')[:,1:]
    rf_timeseries = np.loadtxt(rf_timeseries_filename, delimiter=',')[:,1:]
    pfc_timeseries = np.loadtxt(pfc_timeseries_filename, delimiter=',')[:,1:]
    wcc_timeseries = np.loadtxt(wcc_timeseries_filename, delimiter=',')[:,1:]
    uc_timeseries = np.loadtxt(uc_timeseries_filename, delimiter=',')[:,1:]

    num_du = rel_timeseries.shape[0]
    num_weeks = rel_timeseries.shape[1]

    satisficing = np.zeros((num_du, num_weeks), dtype=int)

    for i in range(num_du):
        
        rel_timeseries_i = rel_timeseries[i, :]
        rf_timeseries_i = rf_timeseries[i, :]
        pfc_timeseries_i = pfc_timeseries[i, :]
        wcc_timeseries_i = wcc_timeseries[i, :]
        uc_timeseries_i = uc_timeseries[i, :]

        rel_meets_criteria = np.zeros(num_weeks, dtype=int)
        rf_meets_criteria = np.zeros(num_weeks, dtype=int)
        pfc_meets_criteria = np.zeros(num_weeks, dtype=int)
        wcc_meets_criteria = np.zeros(num_weeks, dtype=int)
        uc_meets_criteria = np.zeros(num_weeks, dtype=int)

        rel_meets_criteria[rel_timeseries_i >= 0.98] = 1
        rf_meets_criteria[rf_timeseries_i <= 0.2] = 1
        pfc_meets_criteria[pfc_timeseries_i <= 0.8] = 1
        wcc_meets_criteria[wcc_timeseries_i <= 0.1] = 1
        uc_meets_criteria[uc_timeseries_i <= 5] = 1

        # for each week, if all criteria are met, then the system is satisficing
        satisticifing_i = rel_meets_criteria & rf_meets_criteria & pfc_meets_criteria & wcc_meets_criteria & uc_meets_criteria
        satisficing[i,:] = satisticifing_i

    print('Done calculating regional satisficing over time!')
    return satisficing

def calc_satisficing(sol_num, util_num):
    # Read the data
    rel_timeseries_filename = f'tv_objs/sol{sol_num}_util{util_num}_REL.csv'  
    rf_timeseries_filename = f'tv_objs/sol{sol_num}_util{util_num}_RF.csv'
    pfc_timeseries_filename = f'tv_objs/sol{sol_num}_util{util_num}_PFC.csv'
    wcc_timeseries_filename = f'tv_objs/sol{sol_num}_util{util_num}_WCC.csv'
    uc_timeseries_filename = f'tv_objs/sol{sol_num}_util{util_num}_UC.csv'

    rel_timeseries = np.loadtxt(rel_timeseries_filename, delimiter=',')[:,1:]
    rf_timeseries = np.loadtxt(rf_timeseries_filename, delimiter=',')[:,1:]
    pfc_timeseries = np.loadtxt(pfc_timeseries_filename, delimiter=',')[:,1:]
    wcc_timeseries = np.loadtxt(wcc_timeseries_filename, delimiter=',')[:,1:]
    uc_timeseries = np.loadtxt(uc_timeseries_filename, delimiter=',')[:,1:]

    num_du = rel_timeseries.shape[0]
    num_weeks = rel_timeseries.shape[1]

    satisficing = np.zeros((num_du, num_weeks), dtype=int)

    for i in range(num_du):
        rel_timeseries_i = rel_timeseries[i, :]
        rf_timeseries_i = rf_timeseries[i, :]
        pfc_timeseries_i = pfc_timeseries[i, :]
        wcc_timeseries_i = wcc_timeseries[i, :]
        uc_timeseries_i = uc_timeseries[i, :]
        
        rel_meets_criteria = np.zeros(num_weeks, dtype=int)
        rf_meets_criteria = np.zeros(num_weeks, dtype=int)
        pfc_meets_criteria = np.zeros(num_weeks, dtype=int)
        wcc_meets_criteria = np.zeros(num_weeks, dtype=int)
        uc_meets_criteria = np.zeros(num_weeks, dtype=int)

        rel_meets_criteria[rel_timeseries_i >= 0.98] = 1
        rf_meets_criteria[rf_timeseries_i <= 0.2] = 1
        pfc_meets_criteria[pfc_timeseries_i <= 0.8] = 1
        wcc_meets_criteria[wcc_timeseries_i <= 0.1] = 1
        uc_meets_criteria[uc_timeseries_i <= 5] = 1

        satisticifing_i = rel_meets_criteria & rf_meets_criteria & pfc_meets_criteria & wcc_meets_criteria & uc_meets_criteria
        satisficing[i,:] = satisticifing_i

    print('Done calculating satisficing over time!')
    return satisficing

sol_num = 92

for util_num in range(0,6):
    satisficing = calc_satisficing(sol_num, util_num)
    robustness = np.sum(satisficing, axis=0)/1000
    satisficing_filename = f'output/satisficing_sol{sol_num}_util{util_num}.csv'
    robustness_filename = f'output/robustness_sol{sol_num}_util{util_num}.csv'
    #satisficing_filename = f'robustness_analysis/satisficing_sol{sol_num}_regional.csv'
    np.savetxt(satisficing_filename, satisficing, delimiter=',', fmt='%d')
    np.savetxt(robustness_filename, robustness, delimiter=',', fmt='%.4f')

satisficing_regional_filename = f'output/satisficing_sol{sol_num}_regional.csv'
robustness_regional_filename = f'output/robustness_sol{sol_num}_regional.csv'

satisficing_regional = calc_satisficing_regional(sol_num)
robustness_regional = np.sum(satisficing_regional, axis=0)/1000

np.savetxt(satisficing_regional_filename, satisficing_regional, delimiter=',', fmt='%d')
np.savetxt(robustness_regional_filename, robustness_regional, delimiter=',', fmt='%.4f')
