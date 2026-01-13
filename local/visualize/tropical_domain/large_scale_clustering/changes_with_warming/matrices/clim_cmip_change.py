'''
# -----------------
#   plot_figure
# -----------------

'''

# == imports ==
# -- Packages --
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

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
mS = import_relative_module('user_specs',                                                       'utils')
cT = import_relative_module('util_calc.correlations.self_correlation_matrix',                   'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',               'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')


# == helper funcs ==
def in_subset(model):
    datasets = (                                                                                                                    # Models ordered by change in temperature with warming    
        # 'INM-CM5-0',                                                                                                                # 1   # no cloud
        'IITM-ESM',                                                                                                                 # 2   
        'FGOALS-g3',                                                                                                                # 3    
        # 'INM-CM4-8',                                                                                                                # 4                                
        'MIROC6',                                                                                                                   # 5                                      
        'MPI-ESM1-2-LR',                                                                                                            # 6                         
        # 'KIOST-ESM',                                                                                                              # 7
        'BCC-CSM2-MR',                                                                                                              # 8           
        # 'GFDL-ESM4',                                                                                                                # 9         
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
        # 'EC-Earth3',                                                                                                                # 20 
        'CNRM-CM6-1',                                                                                                               # 21
        # 'CNRM-CM6-1-HR',                                                                                                            # 22  # no clouds
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

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]                 # convert the variables to a data array
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models


# == get correlation matrix ==
def get_correlation_matrix(ds):
    ''' Correlates row_i with all the other rows in the dataset (each row is a separate metric) '''
    # -- identify metrics --
    metric_names = list(ds.data_vars.keys())
    num_metrics = len(metric_names)
    data_array = ds.to_array().values

    # -- create empty numpy array to fill --
    correlation_matrix = np.zeros((num_metrics, num_metrics))
    p_value_matrix = np.zeros((num_metrics, num_metrics))

    # -- fill metrix with correlations --
    for i in range(num_metrics):
        for j in range(num_metrics):
            if np.all(np.isnan(data_array[i, :])) or np.all(np.isnan(data_array[j, :])):    # some timeseries are all NaN (like cloud metric for model without cloud variable)
                correlation_matrix[i, j] = np.nan
                p_value_matrix[i, j] = np.nan
            else:
                valid_indices = ~np.isnan(data_array[i, :]) & ~np.isnan(data_array[j, :])   # some metrics are calculated from a limited timeperiod (1998 - 2017 for obs clouds)
                corr, p_val = pearsonr(data_array[i, valid_indices], data_array[j, valid_indices])
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_val

    # -- give dimensions and coords --
    coords = {'metric1': metric_names, 'metric2': metric_names}  
    da_correlation_matrix  = xr.DataArray(correlation_matrix,  dims = ('metric1', 'metric2'), coords = coords)
    da_p_value_matrix      = xr.DataArray(p_value_matrix,      dims = ('metric1', 'metric2'), coords = coords)        
    return da_correlation_matrix, da_p_value_matrix



# == load metrics ==
def get_metric_tas(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    
    # -- find path --
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                        # base result_filename
            f'{metric_name}'                                                                                            #
            f'_{dataset}'                                                                                               #
            f'_{t_freq}'                                                                                                #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 #
            )       
    path = f'{folder}/{filename}.nc'
    
    # -- find metric -- 
    doc_metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(doc_metric)
        print('exiting')
        exit()
    else:

        # -- get metric variation -- 
        doc_metric = doc_metric[metric_var]
    return doc_metric

def get_dtas(models, p_id1, p_id2):
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8            
    ds_dtas = xr.Dataset()
    for model in models:
        data_tyoe_group, data_type, dataset = 'models', 'cmip', model
        if model == 'NOAA':
            data_tyoe_group, data_type = 'observations', 'NOAA'
        if model == 'IFS_9_FESOM_5':
            data_tyoe_group, data_type = 'models', 'IFS'
        x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_map',    'tas_map_mean',    r'T',   r'[$^o$C]'                               # x1
        map_value = get_metric_tas(data_tyoe_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id1, x1_var).mean(dim = ('lat', 'lon'))
        x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_map',    'tas_map_mean',    r'T',   r'[$^o$C]'                               # x1
        map_value_warm = get_metric_tas(data_tyoe_group, data_type, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id2, x1_var).mean(dim = ('lat', 'lon'))
        ds_dtas[model] = map_value_warm - map_value
    return ds_dtas

def get_metric(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var, time_period2, d_tas = None, scenario = None):
    # if metric_var in ['clouds_timeseries_low_mean', 'clouds_timeseries_high_mean'] and not in_subset(dataset):
    if metric_group in ['clouds'] and not in_subset(dataset):
        return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[range(360)]).mean(dim = 'time')
    else:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
        
        # -- historical --
        folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
        filename = (                                                                                                        # base result_filename
                f'{metric_name}'                                                                                            #
                f'_{dataset}'                                                                                               #
                f'_{t_freq}'                                                                                                #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
                f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 #
                )       
        path = f'{folder}/{filename}.nc'

        if 'area_fraction' in metric_var:
            metric_hist = xr.open_dataset(path)[metric_var].mean(dim = 'time')  # .std(dim = 'time')
        else:
            metric_hist = xr.open_dataset(path)[metric_var].mean(dim = 'time')

        # -- warm --
        folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
        filename = (                                                                                                        # base result_filename
                f'{metric_name}'                                                                                            #
                f'_{dataset}'                                                                                               #
                f'_{t_freq}'                                                                                                #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
                f'_{time_period2.split(":")[0]}_{time_period2.split(":")[1]}'                                                 #
                )       
        path = f'{folder}/{filename}.nc'

        if 'area_fraction' in metric_var:
            # print('executes')
            metric_warm = xr.open_dataset(path)[metric_var].mean(dim = 'time') # .std(dim = 'time')
        else:
            metric_warm = xr.open_dataset(path)[metric_var].mean(dim = 'time')

        if 'area_fraction' in metric_var and scenario == 'historical':
            metric = metric_hist
        else:
            metric = metric_warm - metric_hist
            
            # -- dtas --
            if d_tas is not None:
                metric = metric / d_tas[dataset]

        return metric


# def get_metric_month(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var, time_period2, d_tas = None, scenario = None):
#     if metric_var in ['clouds_timeseries_low_mean', 'clouds_timeseries_high_mean'] and not in_subset(dataset):
#         return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[range(360)]).mean(dim = 'time')
#     else:
#         folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
        
#         # -- historical --
#         folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
#         r_filename = (
#                 f'{metric_name}'   
#                 f'_{dataset}'                                                                                                   
#                 f'_{t_freq}'                                                                                                    
#                 f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
#                 f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
#                 f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
#                 f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
#                 )       
#         folder_metric = f'{folder}/{r_filename}' # fiels are stored in years here
#         year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
#         paths = []
#         for year in np.arange(int(year_start), int(year_end) + 1):
#             path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
#             paths.append(path)

#         if 'area_fraction' in metric_var:
#             metric_hist = xr.open_mfdataset(paths)[metric_var].std(dim = 'time')
#         else:
#             metric_hist = xr.open_mfdataset(paths)[metric_var].mean(dim = 'time')

#         # -- warm --
#         folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
#         r_filename = (
#                 f'{metric_name}'   
#                 f'_{dataset}'                                                                                                   
#                 f'_{t_freq}'                                                                                                    
#                 f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
#                 f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
#                 f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
#                 f'_{time_period2.split(":")[0]}_{time_period2.split(":")[1]}'                                                     
#                 )       
#         folder_metric = f'{folder}/{r_filename}' # fiels are stored in years here
#         year_start, year_end = time_period2.split(":")[0].split('-')[0], time_period2.split(":")[1].split('-')[0]
#         paths = []
#         for year in np.arange(int(year_start), int(year_end) + 1):
#             path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
#             paths.append(path)

#         if 'area_fraction' in metric_var:
#             metric_warm = xr.open_dataset(path)[metric_var].std(dim = 'time')
#         else:
#             metric_warm = xr.open_dataset(path)[metric_var].mean(dim = 'time')

#         if 'area_fraction' in metric_var and scenario == 'historical':
#             metric = metric_hist
#         else:
#             metric = metric_warm - metric_hist
            
#             # -- dtas --
#             if d_tas is not None:
#                 metric = metric / d_tas[dataset]

#         return metric
    

# == plot funcs ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds                                                                                       # [left, bottom, width, height]
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

def cbar_ax_right(fig, ax, h):
    ax_position = ax.get_position()
    cbar_ax = fig.add_axes([ax_position.x1 + 0.0125,                                                                                # left
                            ax_position.y0 + (ax_position.height - ax_position.height * 0.9) / 2,                                   # bottom
                            ax_position.width * 0.025,                                                                              # width
                            ax_position.height * 0.9                                                                                # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize = 7)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    return cbar_ax

def cbar_ax_below(fig, ax, h):
    ax_position = ax.get_position()
    w = 0.8
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                 # left
                            ax_position.y0 - 0.025,                                                                            # bottom
                            ax_position.width * w,                                                                            # width
                            ax_position.height * 0.02                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize = 7)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # cbar.ax.tick_params(labelsize = 7)
    # formatter = ticker.ScalarFormatter(useMathText = True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # cbar.ax.yaxis.set_major_formatter(formatter)
    # cbar.ax.yaxis.get_offset_text().set_size(7)
    # cbar.ax.yaxis.set_offset_position('left')
    return cbar_ax

def lighten(color, amount=0.5):
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)


# == plot ==
def plot(da_corr_matrix, da_p_value_matrix):
    # print(da_p_value_matrix_obs)
    # exit()
    labels = da_corr_matrix['metric1'].data
    da_corr_matrix = da_corr_matrix.data
    plt.rcParams['font.size'] = 7

    # -- create figure --                                                                                                                  
    width, height = 14, 15                                                                                                                              # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                                                 # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax(ax, 1.14)
    move_row(ax, -0.025)     
    move_col(ax, -0.035)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # -- plot colormap --        
    h = ax.imshow(da_corr_matrix, cmap = 'RdBu_r', vmin = - 1, vmax = 1)    

    # -- cmip stats --
    text_size = 8
    for i in range(len(labels)):                                                                                                                            # metric i
        for j in range(len(labels)):                                                                                                                        # metric 2
            corr_value = da_corr_matrix[i, j]
            text = f'{da_corr_matrix[i, j]:.2f}' if da_p_value_matrix[i, j] < 0.05 else f'({da_corr_matrix[i, j]:.2f})'
                                            
            color = 'black'
            ax.text(j, i, text, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'white', fontsize = text_size - 2)                    # plot text in color to show contrast with colormap

            if i == 0:
                ax.text(j - 0.2, - 0.65, labels[j], ha='left', va='bottom', fontsize = text_size, rotation = 30)                                            # top label
                ax.text(j, - 0.5, '-', ha='center', va='bottom', fontweight = 'bold', fontsize = text_size + 0.5, rotation = 90)                            # dash from plot
        ax.text(- 0.8, i, labels[i], ha='right', va='center', fontsize = text_size)                                                                         # side label
        ax.text(- 0.5, i, '-', ha='right', va='center', fontweight = 'bold', fontsize = text_size + 0.5)                                                    # dash from plot

    cbar_ax = cbar_ax_below(fig, ax, h)
    ax_position = ax.get_position()
    ax.text(ax_position.x1 - (ax_position.x1 - ax_position.x0) / 2,
            ax_position.y0 - 0.08, 
            r'Change with warming r(metric$_{row}$, metric$_{col}$)', 
            rotation = 'horizontal', 
            ha = 'center', 
            va = 'center', 
            fontsize = 7, 
            transform=fig.transFigure)
    
    # -- title --
    # ax_position = ax.get_position()
    # ax.text(ax_position.x0 + 0.1,                                                                                                                     # x-start
    #         ax_position.y1 + 0.145,                                                                                                                   # y-start
    #         title,                        
    #         fontsize = 10,  
    #         transform=fig.transFigure,
    #         )
    return fig


# == main ==
def main():
    # == settings common to all ==
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8      
    p_id1, p_id2 =      '1970-01:1999-12', '2070-01:2099-12'      

    # -- normalize by change in temperature mean --
    d_tas = get_dtas(cL.get_model_letters(), p_id1, p_id2)
    # print(d_tas)

    # d_tas = xr.Dataset()
    # for model in cL.get_model_letters():
    #     data_type_group, data_tyoe, dataset = 'models', 'cmip', model
    #     x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_ecs',                                               r'ECS',                 r'[K]'
    #     d_tas[model] =      get_metric_tas(data_type_group, data_tyoe, dataset, x_tfreq,  x_group,  x_name, lon_area, lat_area, res, p_id1, x_var).mean(dim = 'time')
    # # print(ds)
    # # exit()

    # -- add metrics to dataset --
    list_clim = []
    for model in cL.get_model_letters():
        data_type_group, data_tyoe, dataset = 'models', 'cmip', model
        ds = xr.Dataset()
        # -- Clustering --    
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =  'daily',    'doc_metrics',      'area_fraction',        'area_fraction',                            r'$\sigma$(A$_f)$',     r'[]'   
        # ds[f'{x_label} {x_units}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas, 'historical')

        # # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_90',                           r'd$\sigma$(A$_f)$',    r'[K$^{-1}$]'    
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_95',                           r'd$\sigma$(A$_f)$',    r'[K$^{-1}$]'    
        # # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_97',                           r'd$\sigma$(A$_f)$',    r'[K$^{-1}$]'    
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',                               r'dA$_m$',              r'[km$^2$ K$^{-1}$]'   
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',                               r'dA$_m$',              r'[km$^2$ K$^{-1}$]'   
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',                               r'dA$_m$',              r'[km$^2$ K$^{-1}$]'   
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =  'daily',    'doc_metrics',      'number_index',         'number_index',                             r'dN',                  r'[K$^{-1}$]'    
        # ds[f'{x_label} {x_units}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_90',     r'dC$_z$',              r'[km K$^{-1}$]'  
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_95',     r'dP$_{z}$',              r'[km K$^{-1}$]'  
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_97',     r'dC$_z$',              r'[km K$^{-1}$]'  
        ds[f'{x_label}'] =      - get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)
        
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_90',          r'dC$_m$',              r'[km K$^{-1}$]'  
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_95',          r'dP$_{eq}$',              r'[km K$^{-1}$]'  
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_97',          r'dC$_m$',              r'[km K$^{-1}$]'  
        ds[f'{x_label}'] =      - get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)
        
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_90',            r'dC$_{heq}$',          r'[km K$^{-1}$]'   
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_95',            r'dP$_{heq}$',          r'[km K$^{-1}$]'   
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_97',            r'dC$_{heq}$',          r'[km K$^{-1}$]'   
        ds[f'{x_label}'] =      - get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # -- rainfall --
        x_tfreq, x_group, x_name,  x_var,   x_label, x_units =              'monthly',  'precip',              'precip_timeseries',        'precip_timeseries_mean',                                    r'pr',              r'[]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # -- temperature / ENSO --
        # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_mean',                                              r'dT$_s$',                 r'[K]'
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2)

        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =    'monthly',  'tas',              'tas_gradients',        'tas_gradients_pacific_zonal_clim',                                 r'dT$_z$',              r'[K K$^{-1}$]'    
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'wap',              'wap_timeseries',       'wap_timeseries_mean',                                              r'dA$_a$',              r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq, x_group, x_name,  x_var,   x_label, x_units =              'monthly',  'ta',               'ta_timeseries',        'ta_timeseries_mean',                                               r'dT$_{500}$',                   r'[K]'    
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # -- Radiative feedbacks --
        # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'rel_humid',        'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                        r'dRH$_{700}$',                 r'[% K$^{-1}$]'  
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                        r'dRH$_{500}$',                 r'[% K$^{-1}$]'  
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'rel_humid_upper',  'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                        r'dRH$_{250}$',                 r'[% K$^{-1}$]'  
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',                                       r'dLCF$_d$',            r'[% K$^{-1}$]'  
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean',                                      r'dHCF$_a$',            r'[% K$^{-1}$]'  
        # ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        # -- climate sensitivity --
        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_ecs',                                               r'ECS',                 r'[K]'
        ds[f'{x_label}'] =      get_metric_tas(data_type_group, data_tyoe, dataset, x_tfreq,  x_group,  x_name, lon_area, lat_area, res, p_id1, x_var).mean(dim = 'time')

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'daily',    'doc_metrics',      'gini',                   'gini',                                                           r'GINI',                r''  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)


        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean_all',                                       r'dLCF$_{all}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',                                       r'dLCF$_d$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean_tm',                                       r'dLCF$_{tm}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean_se',                                       r'dLCF$_{se}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean_all',                                      r'dHCF$_{all}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean',                                      r'dHCF$_a$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean_tm',                                      r'dHCF$_{tm}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)

        x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =         'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean_se',                                      r'dHCF$_{se}$',            r'[% K$^{-1}$]'  
        ds[f'{x_label}'] =      get_metric(data_type_group, data_tyoe, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id1, x_var,     p_id2, d_tas)







        ds = ds.assign_coords(model = model)
        if 'height' in ds:
            ds = ds.drop_vars('height')
        list_clim.append(ds)
    ds_clim = xr.concat(list_clim, dim = 'model')
    da_corr_matrix, da_p_value_matrix = get_correlation_matrix(ds_clim) 
    # print(da_p_value_matrix)    
    # print(da_corr_matrix)
    # exit()

    # = other potential metrics ==
    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units = 'monthly',  'olr',              'olr_timeseries',       'olr_timeseries_mean',                      r'dOLR',                r'[Wm$^{-2}$ K$^{-1}$]'  
    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units = 'daily',    'doc_metrics',      'f_pr10',               'f_pr10',                                   r'dF$_{pr10}$',         r'[% K$^{-1}$]'  


    # == Plot ==
    fig = plot(da_corr_matrix, da_p_value_matrix)


    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    path = f'{folder_scratch}/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}.png' 
    
    # print(path)
    # exit()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran ==
if __name__ == '__main__':
    # ds = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/models/cmip/doc_metrics/area_fraction/ACCESS-CM2/area_fraction_ACCESS-CM2_daily_0-360_-30-30_128x64_1970-01_1999-12.nc')
    # print(ds['area_fraction'].isel(time = 0).data)
    # exit()
    # path = '/Users/cbla0002/Desktop/work/metrics/models/cmip/doc_metrics/f_pr10/ACCESS-CM2/f_pr10_ACCESS-CM2_daily_0-360_-30-30_128x64_1970-01_1999-12.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()
    main()




# == extra ==
# def get_dtas(models, p_id1, p_id2):
#     lon_area =  '0:360'                                                                                                                                                                             # area
#     lat_area =  '-30:30'                                                                                                                                                                            #
#     res =       2.8            
#     ds_dtas = xr.Dataset()
#     for model in models:
#         data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
#         x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_timeseries',    'tas_timeseries_ecs',    r'T',   r'[$^o$C]'
#         ds_dtas[model] = get_metric_tas(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id1, x1_var).mean(dim = 'time') # map_value_warm - map_value
#     return ds_dtas


