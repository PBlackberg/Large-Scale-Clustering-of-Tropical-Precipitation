'''
# ----------
#  Boxplot
# ----------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from scipy.stats import pearsonr

# -- Imported scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, plot_path):
    ''' import module from relative path '''
    if plot_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        relative_path = plot_path.replace(os.getcwd(), "").lstrip("/")
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS =    import_relative_module('user_specs',                                        'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                    'utils')
gC = import_relative_module('util_calc.correlations.gridbox_regression',            'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',   'utils')
mlR = import_relative_module('util_calc.multiple_linear_regression.mlr_calc',       'utils')


# == get metric ==
def get_metric(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                       # user settings
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'                          # 
    filename = (                                                                                                                    # base result_filename
            f'{metric_name}'                                                                                                        #
            f'_{dataset}'                                                                                                           #
            f'_{t_freq}'                                                                                                            #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                                   #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                                   #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                         #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                             #
            )       
    path = f'{folder}/{filename}.nc'
    metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        metric = metric[metric_var]
        if metric_var == 'tas_gradients_oni':
            metric = metric.ffill(dim='time')                                                                                       # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                                       # Backward fill  (fill first value for 3 month rolling mean)
    return metric

def pre_process_metric(xy_list):
    xy_list = [cA.detrend_data(da) for da in xy_list]    
    xy_list = [cA.get_monthly_anomalies(da) for da in xy_list]    
    xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
    xy_list = list(xr.align(*xy_list, join='inner'))
    xy_list = [da.data for da in xy_list]
    # [print(type(f)) for f in xy_list]
    # exit()
    return xy_list

def get_plot_metric(xy_list):
    x_list = xy_list[:-1]
    y = xy_list[-1]
    # print(f'Linear mdoel coefficients using {len(x_list)} predictor variables')
    vif = mlR.check_variance_inflation_factor(x_list, show = False)
    # print(f'VIF: {vif}')
    show_it = False
    # print(vif)
    # print(coeffs)
    # mlR.get_sequential_variance_decomposition(x_list, y, show = True)
    # r_partial_x2, p_partial_x2 = mlR.get_pearson_partial_correlation(x_list, y, show = show_it)
    # mlR.get_linear_model_components_2(x_list, y, show = True, standardized = True)
    r1, p1 = pearsonr(x_list[0], y)

    y_hat, coeffs, residual = mlR.get_linear_model_components(x_list, y, show = False, standardized = True)
    r2, p2 = pearsonr(y_hat, y)

    # print(r2)
    # print(r1)
    dr = r2 - r1
    # print(dr)
    # exit()
    dp = p2 - p1

    # print(coeffs)
    # print(r_partial_x2)
    # exit()    
    return dr, dp


# == plot ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(0.025,                                                                                              # x-start
            0.93,                                                                                              # y-start
            text,                        
            fontsize = 4,  
            transform=fig.transFigure,
            )

def move_col(ax, moveby):
    ax_position = ax.get_position()             
    _, bottom, width, height = ax_position.bounds                                                                                   # [left, bottom, width, height]
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds                                                                                     # [left, bottom, width, height]
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def plot_ax_title2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0,                                                                                              # x-start
            ax_position.y1 + 0.08,                                                                                               # y-start
            text,                        
            fontsize = 7,  
            # rotation = 'vertical', 
            transform=fig.transFigure,
            ha='center', va='center',
            )
    
def plot_ax_title3(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0,                                                                                              # x-start
            ax_position.y0 - 0.05,                                                                                              # y-start
            text,                        
            fontsize = 7,  
            transform=fig.transFigure,
            ha='left', va='center',
            # rotation = 'vertical', 
            )

def plot(y, y_p, 
         y_cmip_example, y_cmip_example_p, 
         y_obs, y_obs_p, 
         y_highres, y_highres_p):
    # -- create figure --    
    width, height = 2.5, 6                                                                                                                                         # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, scaleby = 0.5)
    scale_ax_y(ax, scaleby = 1)
    move_row(ax, 0)     
    move_col(ax, 0.375)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.set_xlim([0,1])
    # ax.set_ylim([-1,1])
    ax.set_ylim([-0.4,0.4])
    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    
    # -- boxplot lines --
    y_orig = y
    y = np.array([corr for corr, p in zip(y, y_p) if p < 0.05])                                                                                         # pick out statistically significant correlations 
    color = 'k'
    linewidth = 1
    alpha = 1 # 0.8
    box_mid = 0.5 #- 0.1
    box_left = 0.25 #- 0.1
    box_right = 0.75 #- 0.1
    box_top_width = 0.1
    # box
    prc_75 = np.percentile(y, 75)
    ax.plot([box_left, box_right],  [prc_75, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_50 = np.percentile(y, 50)
    ax.plot([box_left, box_right],  [prc_50, prc_50],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_25 = np.percentile(y, 25)
    ax.plot([box_left, box_right],  [prc_25, prc_25],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    ax.plot([box_left, box_left],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # verical line
    ax.plot([box_right, box_right],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # vertical line
    # max
    y_max = np.max(y)
    ax.plot([box_mid, box_mid],     [prc_75, y_max],    color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                           # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_max, y_max],     color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                           # hortizontal line
    # min
    y_min = np.min(y)
    ax.plot([box_mid, box_mid],     [y_min, prc_25], color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                              # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_min, y_min], color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                               # hortizontal line

    # text = f'N(p < 0.05) \n({len(y)} / {len(y_orig)})'
    text = f'({len(y)} / {len(y_orig)})'
    plot_ax_title3(fig, ax, text)

    # -- icons --
    if y_obs_p is not None:
        if y_obs_p < 0.05:
            color = 'g'
            x_position = box_mid - 0.2
            y_position = y_obs
            text = ax.text(x_position, y_position,                                                                                                              # data
                            '\u2605',                                                                                                                           # star                                   
                            color = 'w', ha='center', va='center',                                                                                              #
                            fontsize = 7, fontweight='bold')   
            text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = color)])                                                               # background effects

    if y_highres_p is not None:
        if y_highres_p < 0.05:
            color = 'purple'
            x_position = box_mid + 0.2
            y_position = y_highres
            text = ax.text(x_position, y_position,                                                                                                              # data
                            '\u25C6',                                                                                                                                # star                                   
                            color = 'w', ha='center', va='center',                                                                                              #
                            fontsize = 6, fontweight='bold')   
            text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = 'purple')])                                                            # background effects

    if y_cmip_example_p is not None:
        if y_cmip_example_p < 0.05:
            color = 'k'
            x_position = box_mid
            y_position = y_cmip_example
            text = ax.text(x_position, y_position,                                                                                                              # data
                            'P',                                                                                                                                # text                                   
                            color = 'w', ha='center', va='center',                                                                                              #
                            fontsize = 6, fontweight='bold')                                                                                                  # 
            text.set_path_effects([PathEffects.withStroke(linewidth=1.25, foreground='k')])                                                                     # background effects

    return fig, ax


# == main ==
def main():
    # == Specify metrics to use ==
    # -- x1-metric --
    x1_tfreq,    x1_group,  x1_name,    x1_var, x1_label,   x1_units =   'daily',   'doc_metrics',      'area_fraction',        'area_fraction',                            r'A$_f$',       r''   

    # -- x2-metric --    
    x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   

    # -- x3-metric --    
    x3_tfreq,   x3_group,   x3_name,    x3_var, x3_label,   x3_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  

    # -- x4-metric --    
    x4_tfreq,   x4_group,   x4_name,    x4_var, x4_label,   x4_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_{m}$',     r'km'  

    # -- y-metric --
    y_tfreq,   y_group,     y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'

    # -- normalization --
    n_tfreq,   n_group,     n_name,     n_var,  n_label,    n_units =   'daily',    'conv',              'conv_map',            'conv_map_mean',                            r'C',           r'%'    

    # == Get metrics ==
    # -- OBS: metrics --
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8    
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    time_period = '1998-01:2022-12'      
    x1 = get_metric(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var) # * domain_area
    x2 = get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var) # / length_scale
    x3 = get_metric(data_type_group, data_type, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, time_period, x3_var) # / length_scale
    x4 = get_metric(data_type_group, data_type, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, time_period, x4_var) # / length_scale
    y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var) # / domain_area
    xy_list = [x1, x2, x3, y]
    xy_list = pre_process_metric(xy_list)
    y_obs, y_obs_p = get_plot_metric(xy_list)
    # print(y_obs)
    # exit()

    # -- CMIP: metrics --
    y_cmip, y_cmip_p, model_list = [], [], []
    for model in cL.get_model_letters():
        data_type_group, data_type, dataset = 'models', 'cmip', model
        lon_area =  '0:360'                                                                                                                                                                             
        lat_area =  '-30:30'                                                                                                                                                                            
        res =       2.8    
        time_period =      '1970-01:1999-12'
        x1 = get_metric(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var) # * domain_area
        x2 = get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var) # / length_scale
        x3 = get_metric(data_type_group, data_type, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, time_period, x3_var) # / length_scale
        x4 = get_metric(data_type_group, data_type, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, time_period, x4_var) # / length_scale
        y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var) # / domain_area
        xy_list = [x1, x2, x3, y]
        xy_list = pre_process_metric(xy_list)
        r_partial_x2, p_partial_x2 = get_plot_metric(xy_list)
        y_cmip.append(r_partial_x2)
        y_cmip_p.append(p_partial_x2)
        model_list.append(model)
    y_cmip_example = y_cmip[model_list.index('ACCESS-ESM1-5')]
    y_cmip_example_p = y_cmip_p[model_list.index('ACCESS-ESM1-5')]

    # -- HIGHRES: metrics --
    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8    
    time_period = '2025-01:2049-12'    
    x1 = get_metric(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var) # * domain_area
    x2 = get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var) # / length_scale
    x3 = get_metric(data_type_group, data_type, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, time_period, x3_var) # / length_scale
    x4 = get_metric(data_type_group, data_type, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, time_period, x4_var) # / length_scale
    y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var) # / domain_area
    xy_list = [x1, x2, x3, y]
    xy_list = pre_process_metric(xy_list)
    y_highres, y_highres_p = get_plot_metric(xy_list)

    # == plot ==
    fig, ax = plot(y_cmip, y_cmip_p, 
                   y_cmip_example, y_cmip_example_p, 
                   y_obs, y_obs_p, 
                   y_highres, y_highres_p)
    # text = f'[OBS, CMIP]: x1: {x1_var} \nx1: {x2_var} \ny: {y_var}'
    # plot_ax_title(fig, ax, text)

    text = rf'r($\hat{{y}}$, {y_label}) - r({x1_label}, {y_label})'
    plot_ax_title2(fig, ax, text)


    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'x1_{x1_var}_x2_{x2_var}_y_{y_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()































