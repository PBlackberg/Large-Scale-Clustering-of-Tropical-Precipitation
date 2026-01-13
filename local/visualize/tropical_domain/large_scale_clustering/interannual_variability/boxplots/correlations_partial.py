'''
# -------------------------------
#  Boxplot: partial correlations 
# -------------------------------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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

# == get model subsets ==
def in_subset(model):
    ''' use a subset where there are some models missing variables '''
    datasets = (                                                                                                                    # Models ordered by change in temperature with warming    
        # 'INM-CM5-0',                                                                                                              # 1   # no cloud
        'IITM-ESM',                                                                                                                 # 2   
        'FGOALS-g3',                                                                                                                # 3    
        # 'INM-CM4-8',                                                                                                              # 4                                
        'MIROC6',                                                                                                                   # 5                                      
        'MPI-ESM1-2-LR',                                                                                                            # 6                         
        # 'KIOST-ESM',                                                                                                              # 7
        'BCC-CSM2-MR',                                                                                                              # 8           
        # 'GFDL-ESM4',                                                                                                              # 9         
        'MIROC-ES2L',                                                                                                               # 10 
        'NorESM2-LM',                                                                                                               # 11      
        # 'NorESM2-MM',                                                                                                             # 12                      
        'MRI-ESM2-0',                                                                                                               # 13                            
        'GFDL-CM4',                                                                                                                 # 14      
        'CMCC-CM2-SR5',                                                                                                             # 15                
        'CMCC-ESM2',                                                                                                                # 16                                    
        'NESM3',                                                                                                                    # 17     
        'ACCESS-ESM1-5',                                                                                                            # 18 
        'CNRM-ESM2-1',                                                                                                              # 19 
        # 'EC-Earth3',                                                                                                              # 20 
        'CNRM-CM6-1',                                                                                                               # 21
        # 'CNRM-CM6-1-HR',                                                                                                          # 22  # no clouds
        'KACE-1-0-G',                                                                                                               # 23            
        'IPSL-CM6A-LR',                                                                                                             # 24
        'ACCESS-CM2',                                                                                                               # 25 
        'TaiESM1',                                                                                                                  # 26                      
        'CESM2-WACCM',                                                                                                              # 27   
        'CanESM5',                                                                                                                  # 28  
        'UKESM1-0-LL',                                                                                                              # 29
        )             
    in_subset = False
    if model in datasets:
        in_subset = True
    return in_subset

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

    # if 'thres_precip_prctiles' in metric_var:
    #     metric_var = metric_var.split('_thres_precip_prctiles')[0]

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

def get_metric2(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'observations'
    data_type =         'GPCP'
    metric_group =      'doc_metrics'
    metric_name =       metric_name
    metric_var =        metric_var
    dataset =           dataset
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8
    
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    r_filename = (
            f'{metric_name}'   
            f'_{dataset}'                                                                                                   
            f'_{t_freq}'                                                                                                    
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
            )       
    folder_metric = f'{folder}/{r_filename}' # fiels are stored in years here
    year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
    paths = []
    for year in np.arange(int(year_start), int(year_end) + 1):
        path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
        paths.append(path)

    # -- find metric -- 
    doc_metric = xr.open_mfdataset(paths, combine='by_coords')[metric_var].resample(time='1MS').mean().load()
    return doc_metric


def pre_process_metric(xy_list):
    xy_list = [cA.detrend_data(da) for da in xy_list]    
    xy_list = [cA.get_monthly_anomalies(da) for da in xy_list]    
    xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
    xy_list = list(xr.align(*xy_list, join='inner'))
    xy_list = [da.data for da in xy_list]
    return xy_list

def get_plot_metric(xy_list):
    x_list = xy_list[:-1]
    y = xy_list[-1]
    vif = mlR.check_variance_inflation_factor(x_list, show = False)
    r_partial_x2, p_partial_x2 = mlR.get_pearson_partial_correlation(x_list, y, show = False)
    return r_partial_x2, p_partial_x2

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

def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.35,                                                                                                  # x-start
            ax_position.y1 + 0.075,                                                                                                 # y-start
            text,                        
            transform=fig.transFigure,
            ha='left', va='center',
            )
    
def plot_ax_title_sig(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0,                                                                                                         # x-start
            ax_position.y0 - 0.05,                                                                                                  # y-start
            text,                        
            # fontsize = 7,  
            transform=fig.transFigure,
            ha='left', va='center',
            # rotation = 'vertical', 
            )

def plot(y, y_p, 
         y_cmip_example, y_cmip_example_p, 
         y_obs, y_obs_p, 
         y_highres, y_highres_p, model_list):
    plt.rcParams['font.size'] = 8
    # -- create figure --    
    width, height = 2.5, 6                                                                                                                                      # max: 8.5 (single-column) 17.5 (double column) [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                                                         # convert to inches
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
    # ax.set_ylim([-0.35, 1])
    ax.set_ylim([-0.75, 1])


    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    
    # -- boxplot lines --
    y_orig = y
    y_sig = np.array([corr for corr, p in zip(y, y_p) if p < 0.05])                                                                                             # pick out statistically significant correlations 
    y = np.array([corr for corr, p in zip(y, y_p)]) # if p < 0.05])                                                                                             # all correlations
    color = 'k'
    linewidth = 1
    alpha = 1
    box_mid = 0.5 
    box_left = 0.15 
    box_right = 0.85 
    box_top_width = 0.1
    # box
    prc_75 = np.percentile(y, 75)
    ax.plot([box_left, box_right],  [prc_75, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_50 = np.percentile(y, 50)
    ax.plot([box_left, box_right],  [prc_50, prc_50],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_25 = np.percentile(y, 25)
    ax.plot([box_left, box_right],  [prc_25, prc_25],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    ax.plot([box_left, box_left],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                    # verical line
    ax.plot([box_right, box_right],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                  # vertical line
    # max
    y_max = np.max(y)
    ax.plot([box_mid, box_mid],     [prc_75, y_max],    color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                           # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_max, y_max],     color = color, linestyle = '-', linewidth = 1, alpha = alpha)           # hortizontal line
    # min
    y_min = np.min(y)
    ax.plot([box_mid, box_mid],     [y_min, prc_25], color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                              # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_min, y_min], color = color, linestyle = '-', linewidth = 1, alpha = alpha)               # hortizontal line

    # -- plot letters ontop --
    letters = cL.get_model_letters()                                                                                                                            # letter connection to model
    x = np.random.normal(0.5, 0.1, len(y))
    for i, model in enumerate(model_list):                                                                                                                      # plot models first, with the colormap
        color = (0.2, 0.2, 0.2)                                                                                                                                 # Get color from colormap
        # if letters[model] in ['D', 'A', 'B', 'R', 'O']:
        #     text = ax.text(x[i], y[i],                                                                                                                              # data
        #                     letters[model],                                                                                                                         # letter                                   
        #                     color = color, ha='center', va='center',                                                                                                #
        #                     fontsize = 8, fontweight = 'bold')                                                                                                      #
        #     text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])                                                                     # background effects
    text = f'({len(y_sig)} / {len(y_orig)})'
    plot_ax_title_sig(fig, ax, text)

    # -- observations --
    if y_obs_p is not None:
        if y_obs_p < 0.05:
            alpha = 1
        else:
            alpha = 0.25
        color = 'g'
        x_position = box_mid - 0.25
        y_position = y_obs
        text = ax.text(x_position, y_position,                                                                                                                  # data
                        '\u2605',                                                                                                                               # icon                                  
                        color = 'w', 
                        ha='center', 
                        va='center',                                                                                              
                        fontsize = 8, 
                        fontweight='bold',
                        alpha = alpha
                        )   
        text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = color)])                                                                      # background effects

    # -- high-res model --
    if y_highres_p is not None:
        if y_highres_p < 0.05:
            alpha = 1
        else:
            alpha = 0.25
        color = 'purple'
        x_position = box_mid + 0.25
        y_position = y_highres
        text = ax.text(x_position, y_position,                                                                                                                  # data
                        '\u25C6',                                                                                                                               # icon                                   
                        color = 'w', 
                        ha='center', 
                        va='center',                                                                                              
                        fontsize = 7, 
                        fontweight='bold',
                        alpha = alpha
                        )   
        text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = color)])                                                                   # background effects

    # -- cmip example --
    # if y_cmip_example_p is not None:
    #     if y_cmip_example_p < 0.05:
    #         color = 'k'
    #         x_position = box_mid
    #         y_position = y_cmip_example
    #         text = ax.text(x_position, y_position,                                                                                                            # data
    #                         'P',                                                                                                                              # text                                   
    #                         color = 'w', ha='center', va='center',                                                                                            #
    #                         fontsize = 6, fontweight='bold')                                                                                                  # 
    #         text.set_path_effects([PathEffects.withStroke(linewidth=1.25, foreground='k')])                                                                   # background effects

    return fig, ax

# == main ==
def main():
    # == Specify metrics to use ==
    # -- x1-metric --
    # x1_tfreq,    x1_group,    x1_name, x1_var,  x1_label,    x1_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_90',   r'A$_f$',       r'km$^2$'   
    x1_tfreq,    x1_group,    x1_name, x1_var,  x1_label,    x1_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_95',   r'C',       r'km$^2$'   
    # x1_tfreq,    x1_group,    x1_name, x1_var,  x1_label,    x1_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_97',   r'A$_f$',       r'km$^2$'   


    # -- x2-metric --
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',       r'A$_m$',       r'km$^2$'    
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',       r'A$_m$',       r'km$^2$'    
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',       r'A$_m$',       r'km$^2$'    
    
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_90',      r'C$_z$',       r'km'  
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_95',      r'P$_z$',       r'km'  
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_97',      r'C$_z$',       r'km'  

    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_90',      r'C$_m$',       r'km'  
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_95',      r'P$_{eq}$',       r'km'  
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_97',      r'C$_m$',       r'km'  

    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_90',      r'C$_{heq}$',       r'km'  
    x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_95',      r'P$_{heq}$',       r'km'  
    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_97',      r'C$_{heq}$',       r'km'  

    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =   'monthly',    'tas',              'tas_gradients',       'tas_gradients_oni',                    r'ONI',     r'K'   

    # x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =   'monthly',    'slp',              'slp_gradients',        'slp_gradients_SOI',                        r'SOI',         r'K'   


    # -- y-metric --
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',       r'A$_m$',       r'km$^2$'    
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',       r'A$_m$',       r'km$^2$'    
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',       r'A$_m$',       r'km$^2$'    

    # y_tfreq,    y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',    'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',    r'RH',      r'%'  
    y_tfreq,    y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',    'clouds',             'clouds_timeseries',    'clouds_timeseries_low_mean',   r'LCF$_d$', r'%'  
    # y_tfreq,    y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',    'clouds',           'cloud_timeseries',     'cloud_timeseries_high_mean',   r'HCF$_a$', r'%'  
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'monthly',    'slp',                    'slp_gradients',        'slp_gradients_SOI',            r'SOI',         r'K'   
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =   'monthly',    'tas',              'tas_gradients',       'tas_gradients_oni',                    r'ONI',     r'K'   

    # == Get metrics ==
    # -- OBS: metrics --
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8    
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    time_period = '1998-01:2022-12'      
    x1 = get_metric2(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var)

    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    x2 = - get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var)

    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
    xy_list = [x1, x2, y]
    xy_list = pre_process_metric(xy_list)
    y_obs, y_obs_p = get_plot_metric(xy_list)

    # -- CMIP: metrics --
    # y_tfreq,    y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',    'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',           r'LCF$_d$', r'%'  
    # y_tfreq,    y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',    'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean',           r'HCF$_a$', r'%'  
    y_cmip, y_cmip_p, model_list = [], [], []
    for model in cL.get_model_letters():
        if y_group == 'clouds' and not in_subset(model):
            continue
        data_type_group, data_type, dataset = 'models', 'cmip', model
        lon_area =  '0:360'                                                                                                                                                                             
        lat_area =  '-30:30'                                                                                                                                                                            
        res =       2.8    
        time_period =      '1970-01:1999-12'
        x1 = get_metric(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var)
        x2 = - get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var)
        y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
        xy_list = [x1, x2, y]
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
    x1 = get_metric(data_type_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var)
    x2 = - get_metric(data_type_group, data_type, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var)
    y = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
    xy_list = [x1, x2, y]
    xy_list = pre_process_metric(xy_list)
    y_highres, y_highres_p = get_plot_metric(xy_list)
    # y_highres, y_highres_p = 0, 0

    # == plot ==
    fig, ax = plot(y_cmip, y_cmip_p, 
                   y_cmip_example, y_cmip_example_p, 
                   y_obs, y_obs_p, 
                   y_highres, y_highres_p, model_list)
    # text = f'[OBS, CMIP]: x1: {x1_var} \nx1: {x2_var} \ny: {y_var}'
    # plot_ax_title(fig, ax, text)

    # text = f'r({y_label} | {x1_label}, {x2_label})'

    text = f'r({x2_label}, {y_label} | {x1_label})'
    plot_ax_title(fig, ax, text)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'a_plot'
    path = f'{folder}/{plot_name}.svg'
    # print(path)
    # exit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    # ds = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/observations/ERA5/clouds/clouds_timeseries/ERA5/cloud_timeseries_ERA5_monthly_0-360_-30-30_128x64_1998-01_2010-12.nc')
    # print(ds)
    # exit()
    main()














