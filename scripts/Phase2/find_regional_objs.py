import numpy as np 
import pandas as pd 

'''
Modify these as needed
'''
NUM_DU = 10
NUM_WEEKS = 2344
SOL_NUM = 140

def find_regional_obj_i(sol_num, obj_abbrev):
    # import data 
    obj_owasa = np.loadtxt(f'tv_objs/sol{sol_num}_util0_{obj_abbrev}.csv', delimiter=',') 
    obj_durham = np.loadtxt(f'tv_objs/sol{sol_num}_util1_{obj_abbrev}.csv', delimiter=',')
    obj_cary = np.loadtxt(f'tv_objs/sol{sol_num}_util2_{obj_abbrev}.csv', delimiter=',')
    obj_raleigh = np.loadtxt(f'tv_objs/sol{sol_num}_util3_{obj_abbrev}.csv', delimiter=',')
    obj_chatham = np.loadtxt(f'tv_objs/sol{sol_num}_util4_{obj_abbrev}.csv', delimiter=',')
    obj_pittsboro = np.loadtxt(f'tv_objs/sol{sol_num}_util5_{obj_abbrev}.csv', delimiter=',')
    obj_regional = np.zeros(obj_owasa.shape, dtype=float)

    print(obj_owasa.shape)
    # at each timestep, find the minimum of the objectives
    for d in range(obj_owasa.shape[0]):
        for t in range(obj_owasa.shape[1]):
            if obj_abbrev == 'REL':
                obj_regional[d,t] = min(obj_owasa[d,t], obj_durham[d,t], obj_cary[d,t], obj_raleigh[d,t], obj_chatham[d,t], obj_pittsboro[d,t])
            else:
                obj_regional[d,t] = max(obj_owasa[d,t], obj_durham[d,t], obj_cary[d,t], obj_raleigh[d,t], obj_chatham[d,t], obj_pittsboro[d,t])
    
    # save to csv
    np.savetxt(f'tv_objs/sol{sol_num}_regional_{obj_abbrev}.csv', obj_regional, delimiter=',')
    return obj_regional
            
find_regional_obj_i(SOL_NUM, 'REL')
find_regional_obj_i(SOL_NUM, 'RF')
find_regional_obj_i(SOL_NUM, 'WCC')
find_regional_obj_i(SOL_NUM, 'PFC')
find_regional_obj_i(SOL_NUM, 'UC')
