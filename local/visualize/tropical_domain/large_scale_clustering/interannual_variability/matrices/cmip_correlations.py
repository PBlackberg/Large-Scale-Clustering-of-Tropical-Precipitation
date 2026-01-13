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
pF = import_relative_module('util_plot.get_subplot.corr_table_mean_subplot_new',                'utils')
cT = import_relative_module('util_calc.correlations.self_correlation_matrix',                   'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',               'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')

# == model subset ==
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
        )             
    in_subset = False
    if model in datasets:
        in_subset = True
    return in_subset

# == get metrrics ==
def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    if metric_var in ['clouds_timeseries_low_mean', 'clouds_timeseries_high_mean'] and not in_subset(dataset):
        return xr.DataArray(np.nan * np.ones(10), dims=["time"], coords=[range(10)])
    else:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                   # user settings
        # -- find path --
        folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
        filename = (                                                                                                                # base result_filename
                f'{metric_name}'                                                                                                    #
                f'_{dataset}'                                                                                                       #
                f'_{t_freq}'                                                                                                        #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                               #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                               #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                                     #
                f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                         #
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
        # -- anomalies --
        try:
            metric = cA.detrend_month_anom(metric)                                                                          # correlate anomalies
        except:                                                                                                             #
            metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
            metric = cA.detrend_month_anom(metric)                                                                          #
        metric = metric.assign_coords(time=np.arange(360))                                                                  # make sure they have common time axis (to put in the same dataset)
        return metric

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models

def get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix):
    # -- mean and spread --
    da_corr     = ds_to_da(ds_corr_matrix)
    corr_mean   = da_corr.mean(dim = 'variable')
    corr_spread = da_corr.std(dim='variable')
    # q25 = da_corr.quantile(0.25, dim='variable')
    # q75 = da_corr.quantile(0.75, dim='variable')
    # -- nb of models --
    nb_models   = xr.where(np.isnan(da_corr), 0, int(1)).sum(dim = 'variable').data
    # -- nb of significant --
    da_p        = ds_to_da(ds_p_value_matrix)
    p_threshold = 0.05
    da_p_sig = xr.where(da_p < p_threshold, int(1), 0)
    p_count = da_p_sig.sum(dim = 'variable').data
    return corr_mean, corr_spread, p_count, nb_models

# == plot ==
def main():
    # -- settings common to all, that can be changed --
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8      

    # -- create figure --
    width, height = 17, 16                                                                                                                                    
    width, height = [f / 2.54 for f in [width, height]]     
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
    
    # -- specify metrics --
    # -- DOC --    
    x1_tfreq,   x1_group,   x1_name,    x1_var,     x1_label,   x1_units =  'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',           r'[km$^2$]'   
    x2_tfreq,   x2_group,   x2_name,    x2_var,     x2_label,   x2_units =  'daily',    'doc_metrics',      'area_fraction',        'area_fraction',                            r'A$_f$',           r'[]'    
    x3_tfreq,   x3_group,   x3_name,    x3_var,     x3_label,   x3_units =  'daily',    'doc_metrics',      'number_index',         'number_index',                             r'N',               r'[]'    

    # -- spatial preference --
    x4_tfreq,   x4_group,   x4_name,    x4_var,     x4_label,   x4_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',           r'[km]'  
    x5_tfreq,   x5_group,   x5_name,    x5_var,     x5_label,   x5_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',           r'[km]'  
    x6_tfreq,   x6_group,   x6_name,    x6_var,     x6_label,   x6_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',       r'[km]'   

    # -- mechanisms --
    x7_tfreq,   x7_group,   x7_name,    x7_var,     x7_label,   x7_units =  'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_mean',                      r'T',               r'[K]'
    x8_tfreq,   x8_group,   x8_name,    x8_var,     x8_label,   x8_units =  'monthly',  'tas',              'tas_gradients',        'tas_gradients_oni',                        r'ONI',             r'[K]'    

    # -- effects --
    x9_tfreq,   x9_group,   x9_name,    x9_var,     x9_label,   x9_units =  'monthly',  'olr',              'olr_timeseries',       'olr_timeseries_mean',                      r'OLR',             r'[Wm$^{-2}$]'  
    x10_tfreq,  x10_group,  x10_name,   x10_var,    x10_label,  x10_units = 'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',               r'LCF$_d$',         r'[%]'  
    x11_tfreq,  x11_group,  x11_name,   x11_var,    x11_label,  x11_units = 'monthly',  'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                r'RH',              r'[%]'  
    x12_tfreq,  x12_group,  x12_name,   x12_var,    x12_label,  x12_units = 'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean',              r'HCF$_a$',         r'[%]'  

    # -- extra --
    x13_tfreq,  x13_group,  x13_name,   x13_var,    x13_label,  x13_units = 'daily',    'doc_metrics',      'f_pr10',               'f_pr10',                                   r'F$_{pr10}$',      r'[%]'  
    x14_tfreq,  x14_group,  x14_name,   x14_var,    x14_label,  x14_units = 'monthly',  'wap',              'wap_timeseries',       'wap_timeseries_mean',                      r'A$_a$',           r'[%]'  

    # -- add metrics to dataset and create correlation matrix --
    ds_corr_matrix, ds_p_value_matrix = xr.Dataset(), xr.Dataset()
    for model in cL.get_model_letters():
        ds = xr.Dataset()
        p_id =      '1970-01:1999-12'    
        data_type_group, data_type, dataset = 'models', 'cmip', model
        ds[f'{x1_label} {x1_units}'] =      get_metric(data_type_group, data_type, dataset, x1_tfreq,   x1_group,   x1_name,    lon_area, lat_area, res, p_id, x1_var)
        ds[f'{x2_label} {x2_units}'] =      get_metric(data_type_group, data_type, dataset, x2_tfreq,   x2_group,   x2_name,    lon_area, lat_area, res, p_id, x2_var)
        ds[f'{x3_label} {x3_units}'] =      get_metric(data_type_group, data_type, dataset, x3_tfreq,   x3_group,   x3_name,    lon_area, lat_area, res, p_id, x3_var)
        ds[f'{x4_label} {x4_units}'] =      get_metric(data_type_group, data_type, dataset, x4_tfreq,   x4_group,   x4_name,    lon_area, lat_area, res, p_id, x4_var)
        ds[f'{x5_label} {x5_units}'] =      get_metric(data_type_group, data_type, dataset, x5_tfreq,   x5_group,   x5_name,    lon_area, lat_area, res, p_id, x5_var)
        ds[f'{x6_label} {x6_units}'] =      get_metric(data_type_group, data_type, dataset, x6_tfreq,   x6_group,   x6_name,    lon_area, lat_area, res, p_id, x6_var)
        ds[f'{x7_label} {x7_units}'] =      get_metric(data_type_group, data_type, dataset, x7_tfreq,   x7_group,   x7_name,    lon_area, lat_area, res, p_id, x7_var)
        ds[f'{x8_label} {x8_units}'] =      get_metric(data_type_group, data_type, dataset, x8_tfreq,   x8_group,   x8_name,    lon_area, lat_area, res, p_id, x8_var)
        ds[f'{x9_label} {x9_units}'] =      get_metric(data_type_group, data_type, dataset, x9_tfreq,   x9_group,   x9_name,    lon_area, lat_area, res, p_id, x9_var)
        ds[f'{x10_label} {x10_units}'] =    get_metric(data_type_group, data_type, dataset, x10_tfreq,  x10_group,  x10_name,   lon_area, lat_area, res, p_id, x10_var)
        ds[f'{x11_label} {x11_units}'] =    get_metric(data_type_group, data_type, dataset, x11_tfreq,  x11_group,  x11_name,   lon_area, lat_area, res, p_id, x11_var)
        ds[f'{x12_label} {x12_units}'] =    get_metric(data_type_group, data_type, dataset, x12_tfreq,  x12_group,  x12_name,   lon_area, lat_area, res, p_id, x12_var)
        ds[f'{x13_label} {x13_units}'] =    get_metric(data_type_group, data_type, dataset, x13_tfreq,  x13_group,  x13_name,   lon_area, lat_area, res, p_id, x13_var)
        ds[f'{x14_label} {x14_units}'] =    get_metric(data_type_group, data_type, dataset, x14_tfreq,  x14_group,  x14_name,   lon_area, lat_area, res, p_id, x14_var)
        da_corr_matrix, da_p_value_matrix = cT.get_correlation_matrix(ds) 
        ds_corr_matrix[model] =     da_corr_matrix
        ds_p_value_matrix[model] =  da_p_value_matrix
    
    # -- model mean, spread, and significance --
    da_corr_mean, da_corr_spread, p_count, nb_models = get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix)

    # -- plot metrics --
    ds = xr.Dataset()
    ds.attrs.update({
        'scale': 1.15, 'move_row': -0.1, 'move_col': -0.015,
        'cmap': 'RdBu',
        'hide_labels': False,
        'text_size': 8,
        'hide_cbar': False, 'cbar_pad': 0.025, 'cbar_ypad': 0.1, 'cbar_width': 0.005, 'cbar_height': 0.8, 
        'cbar_numsize': 8, 'cbar_label_pad': 0.1, 'cbar_label': r'r(metric$_{row}$, metric$_{col}$)', 'cbar_fontsize': 9,
        'axtitle_fontsize': 8, 'axtitle_label': '', 'axtitle_ypad': 0, 'axtitle_xpad': 0,
        })
    pF.plot(da_corr_mean, da_corr_spread, p_count, nb_models, fig, ax, ds)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    path = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}.svg' 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran ==
if __name__ == '__main__':
    main()





