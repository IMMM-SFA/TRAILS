import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
import statsmodels.api as sm

'''
    Function to normalize data based on direction of preference and whether each objective is minimized or maximized.
    Output dataframe will have values ranging from 0 (which maps to bottom of figure) to 1 (which maps to top)
    Args:
        objs (pd.DataFrame): dataframe of objective values
        columns_axes (list): list of the subset of column names to be used as parallel axes
        ideal_direction (str): 'upwards' or 'downwards' - direction of preference for each axis
        minmaxs (list): list of 'max' or 'min' for each axis, indicating whether each axis should be maximized or minimized
    Returns:
        objs_reorg (pd.DataFrame): reorganized dataframe of objective values
        top_values (pd.Series): series of top values for each axis
        bottom_values (pd.Series): series of bottom values for each axis
'''
def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    ### if min/max directions not given for each axis, assume all should be maximized
    if minmaxs is None:
        minmaxs = ['max']*len(columns_axes)
         
    ### get subset of dataframe columns that will be shown as parallel axes
    objs_reorg = objs.copy()

    ### reorganize & normalize data to go from 0 (bottom of figure) to 1 (top of figure), 
    ### based on direction of preference for figure and individual axes
    if ideal_direction == 'downwards':
        top_values = objs_reorg.min(axis=0)
        bottom_values = objs_reorg.max(axis=0)
        for i, minmax in enumerate(minmaxs):
            if minmax == 'min':
                # changed here
                objs_reorg.iloc[:, i] = (objs.iloc[:, i].max(axis=0) - objs.iloc[:, i]) / \
                                        (objs.iloc[:, i].max(axis=0) - objs.iloc[:, i].min(axis=0))
                
            else:
                bottom_values[i], top_values[i] = top_values[i], bottom_values[i]
                objs_reorg.iloc[:, -1] = (objs.iloc[:, -1] - objs.iloc[:, -1].min(axis=0)) / \
                                         (objs.iloc[:, -1].max(axis=0) - objs.iloc[:, -1].min(axis=0))
    
    elif ideal_direction == 'upwards':
        #top_values = objs_reorg.max(axis=0)
        top_values = [1.0]*len(columns_axes)
        #bottom_values = objs_reorg.min(axis=0)
        bottom_values = [0.0]*len(columns_axes)
        
        for i, minmax in enumerate(minmaxs):
            if minmax == 'max':
                continue
                ''' 
                objs_reorg.iloc[:, i] = (objs.iloc[:, i] - objs.iloc[:, i].min(axis=0)) / \
                                        (objs.iloc[:, i].max(axis=0) - objs.iloc[:, i].min(axis=0))
                '''
            else:
                bottom_values[i], top_values[i] = top_values[i], bottom_values[i]
                objs_reorg.iloc[:, i] = (objs.iloc[:, i].max(axis=0) - objs.iloc[:, i]) / \
                                        (objs.iloc[:, i].max(axis=0) - objs.iloc[:, i].min(axis=0))
    
    objs_reorg = objs_reorg[columns_axes]
 
    return objs_reorg, top_values, bottom_values

'''
    Function to get color based on continuous color map or categorical map
    Args:
        value (float or str): value to be colored
        color_by_continuous (int): index of column to be colored by continuous color map
        color_palette_continuous (str): name of continuous color map
        color_by_categorical (int): index of column to be colored by categorical color map
        color_dict_categorical (dict): dictionary of categorical color map
    
    Returns:
        color (str): color to be used for given value
'''
def get_color(value, color_by_continuous, color_palette_continuous, 
              color_by_categorical, color_dict_categorical):
    
    if color_by_continuous is not None:
        #color = colormaps.get_cmap(color_palette_continuous)(value)
        color = cm.get_cmap(color_palette_continuous)(value)
    elif color_by_categorical is not None:
        color = color_dict_categorical[value]
    return color
	
'''
    Function to get zorder value for ordering lines on plot.
    Works by binning a given axis' values and mapping to discrete classes.

    Args:
        norm_value (float): normalized value of objective
        zorder_num_classes (int): number of classes to bin values into
        zorder_direction (str): 'ascending' or 'descending' - direction of preference for zorder
    Returns:
        zorder (int): zorder value for ordering lines on plot

    '''
def get_zorder(norm_value, zorder_num_classes, zorder_direction):
    
    xgrid = np.arange(0, 1.001, 1/zorder_num_classes)
    if zorder_direction == 'ascending':
        #return 4 + np.sum(norm_value > xgrid)
        return np.sum(norm_value > xgrid)
    elif zorder_direction == 'descending':
        #return 4 + np.sum(norm_value < xgrid)
        return 4 + np.sum(norm_value < xgrid)
'''
    Function to create a customizable parallel coordinates plot.

    Args:
        objs (pd.DataFrame): dataframe of objective values
        columns_axes (list): list of the subset of column names to be used as parallel axes
        axis_labels (list): list of axis labels
        ideal_direction (str): 'upwards' or 'downwards' - direction of preference for each axis
        minmaxs (list): list of 'max' or 'min' for each axis, indicating whether each axis should be maximized or minimized
        color_by_continuous (array): the array of values to be colored by continuous color map
        color_palette_continuous (str): name of continuous color map
        color_by_categorical (array): index of column to be colored by categorical color map
        color_palette_categorical (str): name of categorical color map
        colorbar_ticks_continuous (list): list of ticks for continuous color map
        color_dict_categorical (dict): dictionary of categorical color map
        zorder_by (int): index of column to be used for ordering lines on plot
        zorder_num_classes (int): number of classes to bin values into
        zorder_direction (str): 'ascending' or 'descending' - direction of preference for zorder
        alpha_base (float): transparency of lines
        brushing_dict (dict): dictionary of brushing criteria
        alpha_brush (float): transparency of brushed lines
        lw_base (float): line width
        fontsize (int): font size
        figtitle (str): title of figure
        figsize (tuple): figure size
        save_fig_filename (str): filename to save figure to
    
'''
def custom_parallel_coordinates(objs, columns_axes=None, axis_labels=None, 
                                ideal_direction='upwards', minmaxs=None, 
                                color_by_continuous=None, color_palette_continuous=None, 
                                color_by_categorical=None, color_palette_categorical=None,
                                colorbar_ticks_continuous=None, color_dict_categorical=None,
                                zorder_by=None, zorder_num_classes=10, zorder_direction='ascending', 
                                alpha_base=0.8, brushing_dict=None, alpha_brush=0.1, 
                                lw_base=1.5, fontsize=14, figtitle = None,
                                figsize=(11,6), save_fig_filename=None,
                                single_solution = False, single_solution_idx = 0, single_solution_color = 'red',
                                many_solutions = False, many_solutions_idx = None, many_solutions_color = None,
                                kde_plot=False):
    

    # get all column names of the original dataframe
    #all_columns = objs.columns

    ### verify that all inputs take supported values
    assert ideal_direction in ['upwards','downwards']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    
    assert color_by_continuous is None or color_by_categorical is None
    
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes
     
    ### create figure
    fig,ax = plt.subplots(1,1,figsize=figsize, gridspec_kw={'hspace':0.1, 'wspace':0.1})
 
    ### reorganize & normalize objective data that you want to plot
    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)
    objs_reorg = objs_reorg.fillna(1)

    ### apply any brushing criteria
    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.

        ### iteratively apply all brushing criteria to get satisficing set of solutions
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] >= threshold)
            elif operator == '==':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] == threshold)

            
    ### loop over all solutions/rows & plot on parallel axis plot
    satisficing_solutions= []
    satisficing_np = None
    if brushing_dict is not None:
        satisficing_solutions = [i for i,e in enumerate(satisfice) if e == True]
        satisficing_np = np.array(satisficing_solutions)
        np.savetxt('satisficing.csv', satisficing_np, delimiter=',')

    for i in range(objs_reorg.shape[0]):
        if color_by_continuous is not None:
            color = get_color(objs_reorg[columns_axes[color_by_continuous]].iloc[i], 
                              color_by_continuous, color_palette_continuous, 
                              color_by_categorical, color_dict_categorical)
        elif color_by_categorical is not None:
            color = get_color(objs[color_by_categorical].iloc[i], 
                              color_by_continuous, color_palette_categorical, 
                              color_by_categorical, color_dict_categorical)

        else:
            color = 'silver'
                         
        ### order lines according to ascending or descending values of one of the objectives
        if zorder_by is None:
            zorder = 4
        else:
            zorder = get_zorder(objs_reorg[columns_axes[zorder_by]].iloc[i], 
                                zorder_num_classes, zorder_direction)
             
        ### apply any brushing?
        if brushing_dict is not None:
            if satisfice.iloc[i]:
                alpha = alpha_base
                lw = lw_base
                #satisficing_solutions.append(i)
            else:
                alpha = alpha_brush
                lw = 1.5
                zorder = 2
        elif single_solution:
            if i == single_solution_idx:
                alpha = alpha_base
                lw = lw_base
                color = single_solution_color
            else:
                alpha = alpha_brush
                lw = 1.5
                zorder = 2
        else:
            alpha = alpha_base
            lw = lw_base
        ### loop over objective/column pairs & plot lines between parallel axes
        #for j in range(objs_reorg.shape[1]-1):
        for j in range(0,len(columns_axes)-1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j+1]]
            x = [j, j+1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)
             
    ### add top/bottom ranges
    for j in range(len(columns_axes)-1):
        if j < len(columns_axes):
            ax.annotate(str(tops[j]), [j, 1.01], ha='center', va='bottom', 
                        zorder=zorder, fontsize=fontsize)
            if j == len(columns_axes):
                ax.annotate(str(bottoms[j]) + '+', [j, -0.01], ha='center', va='top', 
                            zorder=zorder, fontsize=fontsize)    
            else:
                ax.annotate(str(bottoms[j]), [j, -0.01], ha='center', va='top', 
                            zorder=zorder, fontsize=fontsize)    
        ax.plot([j,j], [0,1], c='lightgrey', zorder=1, linewidth=1.5)
    
    if single_solution:
        for j in range(len(columns_axes)-1):
            ax.plot([j, j+1], [objs_reorg.iloc[single_solution_idx, j], objs_reorg.iloc[single_solution_idx, j+1]], 
                    c=single_solution_color, zorder=8, lw=3.5)
    
    if many_solutions:
        for i in range(len(many_solutions_idx)):
            idx = many_solutions_idx[i]
            for j in range(len(columns_axes)-1):
                ax.plot([j, j+1], [objs_reorg.iloc[idx, j], objs_reorg.iloc[idx, j+1]], 
                        c=many_solutions_color[i], zorder=8, lw=4)
                kde = sm.nonparametric.KDEUnivariate(objs_reorg.iloc[j, :])
                if j == len(columns_axes)-2:
                    ax.plot([j, j+1], [objs_reorg.iloc[idx, j], objs_reorg.iloc[idx, j+1]], 
                        c=many_solutions_color[i], zorder=8, lw=4, label=f'Solution {idx}')
                    # plot a kde plot on the y axis 
                    kde = sm.nonparametric.KDEUnivariate(objs_reorg.iloc[idx, :])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(many_solutions_idx), frameon=False)
        
    if kde_plot:
        bw = 0.025
        for j in range(len(columns_axes)):
            y = np.arange(0, 1, 0.01)
            x = []

            kde = sm.nonparametric.KDEUnivariate(objs_reorg.iloc[:, j])
            kde.fit(bw=bw)

            for yy in y:
                xx = kde.evaluate(yy) * 0.08
                if np.isnan(xx):
                    x.append(0)
                else:
                    x.append(xx[0])
            x = np.array(x)
            ax.fill_betweenx(y, x+j, j, x, color='lightseagreen', alpha=0.7, zorder=41, fc='turquoise', ec='k')
        #ax.fill_betweenx(y, len(columns_axes)+1, len(columns_axes)+2, x, color='lightseagreen', alpha=0.7, zorder=0, fc='turquoise', ec='k')

    ### other aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
     
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)
 
    if ideal_direction == 'upwards':
        ax.arrow(-0.15,0.1,0,0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'downwards':
        ax.arrow(-0.15,0.9,0,-0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3,0.5), ha='center', va='center',
                rotation=90, fontsize=fontsize)
 
    #ax.set_xlim(-0.4, len(columns_axes)-2)
    ax.set_ylim(-0.4, 1.1)
     
    for i,l in enumerate(axis_labels):
        ax.annotate(l, xy=(i,-0.1), ha='center', va='top', fontsize=fontsize)
    ax.patch.set_alpha(0)
     
 
    ### colorbar for continuous legend
    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=0, vmax=1)
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4, 
                          label='Robustness', pad=0.03, 
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous, 
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)  
    
    ### categorical legend
    '''
    elif color_by_categorical is not None:
        leg = []
        for label,color in color_dict_categorical.items():
            leg.append(Line2D([0], [0], color=color, lw=3, 
                              alpha=alpha_base, label=label))
        _ = ax.legend(handles=leg, loc='lower center', 
                      ncol=max(3, len(color_dict_categorical)),
                      bbox_to_anchor=[0.5,-0.07], frameon=False, fontsize=fontsize)
    '''
    
    if figtitle is not None:
        plt.title(figtitle)

    ### save figure
    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=300)

        