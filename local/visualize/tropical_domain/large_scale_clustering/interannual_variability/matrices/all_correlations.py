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
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',               'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')
mlR = import_relative_module('util_calc.multiple_linear_regression.gridbox_mlr',                'utils')


# == helper funcs ==
def in_subset(model):
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
        # 'CNRM-CM6-1-HR',                                                                                                          # 22
        'KACE-1-0-G',                                                                                                               # 23            
        'IPSL-CM6A-LR',                                                                                                             # 24
        'ACCESS-CM2',                                                                                                               # 25 
        'TaiESM1',                                                                                                                  # 26                      
        'CESM2-WACCM',                                                                                                              # 27   
        'CanESM5',                                                                                                                  # 28  
        'UKESM1-0-LL',                                                                                                              # 29
        'IFS_9_FESOM_5',
        'ISCCP',
        )             
    in_subset = False
    if model in datasets:
        in_subset = True
    return in_subset

def pre_process(metric):
    # metric = cA.detrend_month_anom(metric)                                                                          
    metric = cA.get_monthly_anomalies(metric)
    return metric

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models

def get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix):
    # -- mean and spread --
    nb_models   = len(ds_corr_matrix.data_vars)
    da_corr     = ds_to_da(ds_corr_matrix)
    corr_mean   = da_corr.mean(dim = 'variable')
    corr_spread = da_corr.std(dim='variable')
    # -- nb of significant --
    da_p        = ds_to_da(ds_p_value_matrix)
    p_threshold = 0.05
    da_p_sig = xr.where(da_p < p_threshold, int(1), 0)
    p_count = da_p_sig.sum(dim = 'variable').data
    return corr_mean, corr_spread, p_count, nb_models

def get_change_with_warming():
    ''


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

def get_model_ensemble_stats(ds):
    # -- get metric correlations for each model --
    ds_corr_matrix, ds_p_value_matrix = xr.Dataset(), xr.Dataset()
    # print(ds)
    # print(ds.model.values)
    # exit()
    for model in ds.model.values:
        ds_metric = ds.sel(model = model)
        ds_corr_matrix[model], ds_p_value_matrix[model] = get_correlation_matrix(ds_metric)

    # -- get ensemble stats --
    da_corr     = ds_to_da(ds_corr_matrix)
    corr_mean   = da_corr.mean(dim = 'variable')
    corr_spread = da_corr.std(dim='variable')
    nb_models   = xr.where(np.isnan(da_corr), 0, int(1)).sum(dim = 'variable').data
    da_p        = ds_to_da(ds_p_value_matrix)
    p_threshold = 0.05
    da_p_sig = xr.where(da_p < p_threshold, int(1), 0)
    p_count = da_p_sig.sum(dim = 'variable').data
    return corr_mean, corr_spread, p_count, nb_models


# == load metrics ==
def get_metric(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var, just_return_metric = False):
    if metric_group in ['clouds'] and not in_subset(dataset):
        print(f'model: {dataset} doesnt have cloud variable, creating metric of NaN')
        return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[np.arange(360)])
    else:
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
        metric = xr.open_dataset(path)
        if not metric_var:
            print('choose a metric variation')
            print(metric)
            print('exiting')
            exit()
        else:
            # -- get metric variation -- 
            metric = metric[metric_var]
        # if just_return_metric:
        #     return metric

        # -- check NaN --
        try:
            cA.detrend_month_anom(metric)  
            cA.get_monthly_anomalies(metric)
        except:                                                                                                             # it won't work with nan, so..
            metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
            print(f'forward / backwards filling {metric_name}')

        # -- get anomalies --
        metric = pre_process(metric)

        # -- check length --
        if  metric.sizes['time'] < 300:
            metric = metric.pad(time=(0, 300 - metric.sizes['time']), constant_values=np.nan)                               # some obs products don't have the full time period, so pad the end
            print(f'padding {dataset} {metric_name} with nan to match length of 300 of time-dim')
            metric = metric.assign_coords(time=np.arange(300))     
        elif metric.sizes['time'] > 300:
            metric = metric.assign_coords(time=np.arange(360))                                                                  # make sure they have common time axis (to put in the same dataset)
        else:
            metric = metric.assign_coords(time=np.arange(300))    
    return metric


def get_metric_residual(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    if metric_group in ['clouds'] and not in_subset(dataset):
        print(f'model: {dataset} doesnt have cloud variable, creating metric of NaN')
        return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[np.arange(360)])
    else:
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
        metric = xr.open_dataset(path)
        if not metric_var:
            print('choose a metric variation')
            print(metric)
            print('exiting')
            exit()
        else:
            # -- get metric variation -- 
            metric = metric[metric_var]

        # -- check NaN --
        try:
            cA.detrend_month_anom(metric)  
            cA.get_monthly_anomalies(metric)
        except:                                                                                                             # it won't work with nan, so..
            metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
            print(f'forward / backwards filling {metric_name}')

        # -- get anomalies --
        metric = pre_process(metric)

        # -- check length --
        if  metric.sizes['time'] < 300:
            metric = metric.pad(time=(0, 300 - metric.sizes['time']), constant_values=np.nan)                               # some obs products don't have the full time period, so pad the end
            print(f'padding {dataset} {metric_name} with nan to match length of 300 of time-dim')
            metric = metric.assign_coords(time=np.arange(300))     
        elif metric.sizes['time'] > 300:
            metric = metric.assign_coords(time=np.arange(360))                                                                  # make sure they have common time axis (to put in the same dataset)
        else:
            metric = metric.assign_coords(time=np.arange(300))    
        
        # -- find residuals of --
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_95',                           r'A$_f$',           r'[km$^2$]'  
        try:
            af = get_metric_month(data_type_group, data_type, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, resolution, time_period, x_var, just_return_metric = True)
        except:
            af = get_metric(data_type_group, data_type, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, resolution, time_period, x_var, just_return_metric = True)
        xy_list = [af, metric]
        xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
        xy_list = list(xr.align(*xy_list, join='inner'))
        xy_list_numpy = [da.data for da in xy_list]
        y_hat, coeffs, residual, p_values = mlR.get_linear_model_components(x_list = [xy_list_numpy[0]], y = xy_list_numpy[1], show = False, standardized = False)
        metric = xr.DataArray(residual, dims=("time",), coords={"time": xy_list[0].time})

        # print(residual)
        # exit()
    return metric



def get_metric_month(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var, just_return_metric = False):
    ''' Some metrics are saved in monthly frequency '''
    if metric_group in ['clouds'] and not in_subset(dataset):
        print(f'model: {dataset} doesnt have cloud variable, creating metric of NaN')
        return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[np.arange(360)])
    else:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings

        # -- find path --
        folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
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
        metric = xr.open_mfdataset(paths, combine='by_coords')[metric_var].resample(time='1MS').mean().load()
        # if just_return_metric:
        #     return metric
        
        # -- check NaN --
        try:
            cA.detrend_month_anom(metric)  
            cA.get_monthly_anomalies(metric)
        except:                                                                                                             # it won't work with nan, so..
            metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
            print(f'forward / backwards filling {metric_name}')

        # -- get anomalies --
        metric = pre_process(metric)

        # -- check length --
        if  metric.sizes['time'] < 300:
            metric = metric.pad(time=(0, 300 - metric.sizes['time']), constant_values=np.nan)                               # some obs products don't have the full time period, so pad the end
            print(f'padding {dataset} {metric_name} with nan to match length of 300 of time-dim')
            metric = metric.assign_coords(time=np.arange(300))                      
        elif metric.sizes['time'] > 300:
            metric = metric.assign_coords(time=np.arange(360))       
        else:
            metric = metric.assign_coords(time=np.arange(300))    
    
    return metric


def get_metric_month_residual(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    ''' Some metrics are saved in monthly frequency '''
    if metric_group in ['clouds'] and not in_subset(dataset):
        print(f'model: {dataset} doesnt have cloud variable, creating metric of NaN')
        return xr.DataArray(np.nan * np.ones(360), dims=["time"], coords=[np.arange(360)])
    else:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings

        # -- find path --
        folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
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
        metric = xr.open_mfdataset(paths, combine='by_coords')[metric_var].resample(time='1MS').mean().load()
                
        # -- check NaN --
        try:
            cA.detrend_month_anom(metric)  
            cA.get_monthly_anomalies(metric)
        except:                                                                                                             # it won't work with nan, so..
            metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
            print(f'forward / backwards filling {metric_name}')

        # -- get anomalies --
        metric = pre_process(metric)

        # -- check length --
        if  metric.sizes['time'] < 300:
            metric = metric.pad(time=(0, 300 - metric.sizes['time']), constant_values=np.nan)                               # some obs products don't have the full time period, so pad the end
            print(f'padding {dataset} {metric_name} with nan to match length of 300 of time-dim')
            metric = metric.assign_coords(time=np.arange(300))                      
        elif metric.sizes['time'] > 300:
            metric = metric.assign_coords(time=np.arange(360))       
        else:
            metric = metric.assign_coords(time=np.arange(300))    

        # -- find residuals --
        x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_95',                           r'A$_f$',           r'[km$^2$]'  
        af = get_metric_month(data_type_group, data_type, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, resolution, time_period, x_var, just_return_metric = True)
        xy_list = [af, metric]
        xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
        xy_list = list(xr.align(*xy_list, join='inner'))
        xy_list_numpy = [da.data for da in xy_list]
        y_hat, coeffs, residual, p_values = mlR.get_linear_model_components(x_list = [xy_list_numpy[0]], y = xy_list_numpy[1], show = False, standardized = False)
        metric = xr.DataArray(residual, dims=("time",), coords={"time": xy_list[0].time})
        # print(metric)
        # exit()

    return metric


def get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var):
    model_metric = {}
    for model in cL.get_model_letters():

        data_type_group, data_type, dataset = 'models', 'cmip', model
        model_metric[model] = get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)
    da = xr.concat(list(model_metric.values()), dim=xr.DataArray(list(model_metric.keys()), dims="model"),
                   coords="minimal",
                   compat="override"
                   )
    return da


def get_model_ensemble_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var):
    model_metric = {}
    for model in cL.get_model_letters():
        data_type_group, data_type, dataset = 'models', 'cmip', model
        model_metric[model] = get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)
        if "height" in model_metric[model].coords:
            model_metric[model] = model_metric[model].reset_coords("height", drop=True)
    da = xr.concat(list(model_metric.values()), dim=xr.DataArray(list(model_metric.keys()), dims="model"),                
                   coords="minimal",
                   compat="override"
                   )
    return da



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

def lighten(color, amount=0.5):
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)

def cbar_ax_below(fig, ax, h):
    ax_position = ax.get_position()
    w = 0.6
    cbar_ax = fig.add_axes([ax_position.x0, # + (ax_position.width - ax_position.width * w) / 2,                                 # left
                            ax_position.y0 - 0.02,                                                                            # bottom
                            ax_position.width * w,                                                                            # width
                            ax_position.height * 0.01                                                                            # height
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

# == plot ==
def plot(da_corr_matrix_obs, da_p_value_matrix_obs, da_corr_matrix_ifs, da_p_value_matrix_ifs, corr_mean, corr_spread, p_count, nb_models):
    # print(da_p_value_matrix_obs)
    # exit()
    labels = corr_mean['metric1'].data
    corr_mean, corr_spread = corr_mean.data, corr_spread.data
    plt.rcParams['font.size'] = 7

    # -- create figure --                                                                                                                  
    width, height = 18, 19                                                                                                                              # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                                                 # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax(ax, 1.175)  
    move_row(ax, - 0.06)    
    move_col(ax, -0.045)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # -- plot colormap --        
    h = ax.imshow(corr_mean, cmap = 'RdBu_r', vmin = - 1, vmax = 1)
    ax.set_aspect(1.15)     

    # -- cmip stats --
    text_size = 7
    for i in range(len(labels)):                                                                                                                            # metric i
        for j in range(len(labels)):                                                                                                                        # metric 2
            corr_value = corr_mean[i, j]
            spread_value = corr_spread[i, j]
            count_value = f'{p_count[i, j]} / {nb_models[i, j]}'

            text_obs = f'{da_corr_matrix_obs[i, j].data:.2f}' if da_p_value_matrix_obs[i, j] < 0.05 else f'({da_corr_matrix_obs[i, j].data:.2f})'
            text_ifs = f'{da_corr_matrix_ifs[i, j].data:.2f}' if da_p_value_matrix_ifs[i, j] < 0.05 else f'({da_corr_matrix_obs[i, j].data:.2f})'
            text_cmip = f"{corr_value:.2f}"
            text_cmip1 = f"± {spread_value:.2f}"
            text_cmip2 = f"({count_value})"
                                            
            color = 'green'
            ax.text(j, i - 0.38, text_obs, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'lightgreen', fontsize = text_size - 2)    # plot text in color to show contrast with colormap

            color = 'purple'
            ax.text(j, i - 0.18, text_ifs, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'pink', fontsize = text_size - 2)          # plot text in color to show contrast with colormap

            color = 'black'
            ax.text(j, i + 0,  text_cmip, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'w', fontsize = text_size - 2)            # plot text in color to show contrast with colormap
            ax.text(j, i + 0.18, text_cmip1, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'w', fontsize = text_size - 2)            # plot text in color to show contrast with colormap
            ax.text(j, i + 0.38,  text_cmip2, ha='center', va='center', color = color if abs(corr_value) < 0.5 else 'w', fontsize = text_size - 2)            # plot text in color to show contrast with colormap

            if i == 0:
                ax.text(j - 0.2, - 0.65, labels[j], ha='left', va='bottom', fontsize = text_size, rotation = 30)                                            # top label
                ax.text(j, - 0.5, '-', ha='center', va='bottom', fontweight = 'bold', fontsize = text_size + 0.5, rotation = 90)                            # dash from plot
        ax.text(- 0.8, i, labels[i], ha='right', va='center', fontsize = text_size)                                                                         # side label
        ax.text(- 0.5, i, '-', ha='right', va='center', fontweight = 'bold', fontsize = text_size + 0.5)                                                    # dash from plot
        # ax.text(i - 0.45, i, '-', ha='right', va='center', fontweight = 'bold', fontsize=fontsize + 0.5)                                                  # dash from plot

    cbar_ax = cbar_ax_below(fig, ax, h)
    ax_position = ax.get_position()
    ax.text(ax_position.x1 - (ax_position.x1 - ax_position.x0) / 5,
            ax_position.y0 - 0.016, 
            r'interannual variability r(metric$_{row}$, metric$_{col}$)', 
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
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8      
    p_id =      '1998-01:2022-12' 
    p_id_ifs =  '2025-01:2049-12'    
    p_id_cmip = '1970-01:1999-12'    

    # == Get metrics == 
    ds, ds_ifs, ds_cmip = xr.Dataset(), xr.Dataset(), xr.Dataset()

    # -- clustering --
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',        'mean_area_thres_precip_prctiles_90',                           r'A$_m$',           r'[km$^2$]'  
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',        'mean_area_thres_precip_prctiles_95',                           r'A$_m$',           r'[km$^2$]'  
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',        'mean_area_thres_precip_prctiles_97',                           r'A$_m$',           r'[km$^2$]'  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  get_metric_month(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)
    # print(ds[f'{x_label}'])
    # exit()

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)
    # print(ds_ifs[f'{x_label}'])
    # exit()

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_90',                           r'A$_f$',           r'[km$^2$]'  
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_95',                           r'C',           r'[km$^2$]'  
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'area_fraction',        'area_fraction_thres_precip_prctiles_97',                           r'A$_f$',           r'[km$^2$]'  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  get_metric_month(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)
    # print(ds[f'{x_label}'])
    # exit()

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)
    # print(ds_ifs[f'{x_label}'])
    # exit()

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',                               r'A$_m$|A$_f$',           r'[km$^2$]'   
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',                               r'A$_m$|C',           r'[km$^2$]'   
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',                               r'A$_m$|A$_f$',           r'[km$^2$]'   
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  get_metric_month_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)

    # exit()

    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_90',     r'C$_z$|A$_f$',           r'[km]'  
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_95',     r'P$_{z}$|C',           r'[km]'  
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_97',     r'C$_z$|A$_f$',           r'[km]'  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             - get_model_ensemble_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_90',          r'C$_m$|A$_f$',           r'[km]'  
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_95',          r'P$_{eq}$|C',           r'[km]'  
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line_thres_precip_prctiles_97',          r'C$_m$|A$_f$',           r'[km]'  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             - get_model_ensemble_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_90',            r'C$_{heq}$|A$_f$',       r'[km]'   
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_95',            r'P$_{heq}$|C',       r'[km]'   
    # x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro_thres_precip_prctiles_97',            r'C$_{heq}$|A$_f$',       r'[km]'   
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label} {x_units}'] =                                                        - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              - get_metric_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             - get_model_ensemble_residual(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'daily',    'doc_metrics',      'gini',                   'gini',                                                           r'GINI',                r''  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)
    dummy = ds_cmip[f'{x_label}']


    # -- rainfall --
    x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'daily',  'precip',              'precip_timeseries',        'precip_timeseries_mean',                                      r'pr',              r'[]'  
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'monthly',  'precip',              'precip_timeseries',        'precip_timeseries_mean',                                    r'pr',              r'[]'  
    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # -- temperature / ENSO --
    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_mean',                                              r'Ts',               r'[K]'
    data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    x_tfreq,   x_group,   x_name,    x_var,     x_label,   x_units =                    'monthly',  'tas',              'tas_gradients',        'tas_gradients_oni',                                                r'ONI',             r'[K]'    
    data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'monthly',  'slp',              'slp_gradients',        'slp_gradients_SOI',                                                r'SOI',             r'[]'  
    data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # -- ascent area --
    x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'wap',              'wap_timeseries',       'wap_timeseries_mean',                                              r'A$_a$',           r'[%]'  
    data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)
    dummy = ds[f'{x_label}']

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)

    # -- Radiative feedbacks --
    x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'monthly',  'ta2',               'ta2_timeseries',        'ta2_timeseries_mean',                                            r'T$_{500hpa}$',       r'[K]'    
    data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'monthly',  'ta',               'ta_timeseries',        'ta_timeseries_mean',                                               r'T$_{500hpa}$',       r'[K]'    
    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'rel_humid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                            r'RH$_{700hpa}$',              r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    # ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)

    x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                        r'RH$_{500hpa}$',              r'[%]'  
    data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)

    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'rel_humid_upper',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                                            r'RH$_{250hpa}$',              r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    # ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_low_mean',                                      r'LCF$_d$',         r'[%]'  
    data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_high_mean',                                     r'HCF$_a$',         r'[%]'  
    data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)



    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'daily',    'doc_metrics',      'pr_variance',                   'pr_variance',                                                           r'pr$_{var}$',                r''  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              ds[f'{x_label}'] # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             dummy # get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)





    # == extra clouds (high) ==





    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_high_mean_all',                                     r'HCF$_{all}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_high_mean',                                     r'HCF$_a$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    # ds[f'{x_label}'] =                                                                  dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_high_mean_tm',                                     r'HCF$_{tm}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_high_mean_se',                                     r'HCF$_{se}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)




    # # == extra clouds (low) ==
    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_low_mean_all',                                     r'HCF$_{all}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_low_mean',                                     r'HCF$_a$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    # ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_low_mean_tm',                                     r'HCF$_{tm}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # x_tfreq,  x_group,  x_name,   x_var,    x_label,  x_units =                         'monthly',  'clouds',           'clouds_timeseries',     'clouds_timeseries_low_mean_se',                                     r'HCF$_{se}$',         r'[%]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  dummy #get_metric_month(data_type_group, data_type, dataset, x_tfreq,  x_group,  x_name,   lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              dummy # get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)



    # x_tfreq, x_group, x_name,  x_var,   x_label, x_units =                              'daily',  'precip',              'precip_prctiles',        'precip_prctiles_95',                                            r'pr$_{95}$',             r'[]'  
    # data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # ds[f'{x_label}'] =                                                                  get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id, x_var)

    # data_type_group, data_type, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # ds_ifs[f'{x_label}'] =                                                              get_metric(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_ifs, x_var)

    # data_type_group, data_type, dataset = 'models', 'cmip', 'model'
    # ds_cmip[f'{x_label}'] =                                                             get_model_ensemble(data_type_group, data_type, dataset, x_tfreq,   x_group,   x_name,    lon_area, lat_area, res, p_id_cmip, x_var)


    # -- other potential metrics --
    # x_tfreq,  x_group,    x_name,     xvar,   x_label,    x_units =                   'daily',    'doc_metrics',      'f_pr10',               'f_pr10',                                                           r'F$_{pr10}$',      r'[%]'  
    # x_tfreq,  x_group,    x_name,     x_var,  x_label,    x_units =                   'daily',    'doc_metrics',      'number_index',         'number_index',                                                     r'N',               r'[]'    
    # data_type_group, data_type, dataset = 'observations', 'CERES', 'CERES'
    # x_tfreq,  x_group,    x_name,     x_var,  x_label,    x_units =                   'monthly',  'olr',              'olr_timeseries',       'olr_timeseries_mean',                                              r'OLR',             r'[Wm$^{-2}$]'  

    # print(ds)
    # print(ds_ifs)
    # print(ds_cmip)
    # exit()


    # -- add metrics to dataset and create correlation matrix --
    da_corr_matrix_obs, da_p_value_matrix_obs = get_correlation_matrix(ds) 
    da_corr_matrix_ifs, da_p_value_matrix_ifs = get_correlation_matrix(ds_ifs) 
    corr_mean, corr_spread, p_count, nb_models = get_model_ensemble_stats(ds_cmip)


    # == Plot ==
    fig = plot(da_corr_matrix_obs, da_p_value_matrix_obs, da_corr_matrix_ifs, da_p_value_matrix_ifs, corr_mean, corr_spread, p_count, nb_models)


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
    # ds = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/observations/GPCP/precip/precip_prctiles/GPCP/precip_prctiles_GPCP_daily_0-360_-30-30_128x64_1998-01_2022-12.nc')
    # print(ds)
    # exit()
    main()

