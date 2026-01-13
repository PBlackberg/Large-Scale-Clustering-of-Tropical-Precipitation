'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import xarray as xr
import numpy as np
from pathlib import Path

# -- util- and local scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, file_path):
    ''' import module from relative path '''
    if file_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        cwd = os.getcwd()
        relative_path = os.path.relpath(file_path, cwd) # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                                           'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',   'utils')


# == metric funcs ==
def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (
            f'{metric_name}'                                                                                            
            f'_{dataset}'                                                                                               
            f'_{t_freq}'                                                                                                
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 
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

def elNino_mask(da, oni, timescale = 'daily'):
    if timescale == 'daily':
        oni = oni.reindex(time = da.time, method='ffill').bfill('time')
    threshold = 0.5
    mask = oni > threshold
    return da.where(mask), mask

# -- get conv_threshold --
def get_conv_threshold(dataset, years, da, fixed_area = False):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- specify metric --
    data_tyoe_group =   'models'
    data_type =         'cmip'
    metric_group =      'precip'
    metric_name =       'precip_prctiles'
    metric_var =        'precip_prctiles_95'
    dataset =           dataset
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8
    time_period =       '1970-01:1999-12' if 1970 <= int(years[0]) <= 1999 else '2070-01:2099-12' 
    
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                             
            f'{metric_name}'   
            f'_{dataset}'                                                                                                 
            f'_{t_freq}'                                                                                                  
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                         
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                         
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                               
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                   
            )       
    path = f'{folder}/{filename}.nc'
    # -- find metric -- 
    if not fixed_area:
        threshold = xr.open_dataset(path)[metric_var].mean(dim = 'time')
        da_threshold = threshold.broadcast_like(da.isel(lat = 0, lon = 0))
    else:
        threshold = xr.open_dataset(path)[metric_var]
        da_threshold = threshold.sel(time = da.time, method='nearest')
    return da_threshold


# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name
    ds = xr.Dataset()

    # -- check data --
    da, lon_area, lat_area, dataset, years = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )


    # == basic metric ==
    # -- conv as exceeding precipitation threshsold --
    conv_threshold = get_conv_threshold(dataset, years, da)
    conv_regions_daily = (da > conv_threshold) * 1
    conv_regions = (conv_regions_daily.resample(time='1MS').mean()) * 100
    conv_regions_anom = cA.detrend_month_anom(conv_regions)

    # -- add to xr.dataset --
    ds[f'{metric_name}_mean'] =                    conv_regions.mean(dim = 'time')
    ds[f'{metric_name}_variability_month'] =       conv_regions.std(dim='time')    
    ds[f'{metric_name}_variability_month_anom'] =  conv_regions_anom .std(dim='time')    


    # == for difference map between el nino and not el nino ==
    p_id =       '1970-01:1999-12' if 1970 <= int(years[0]) <= 1999 else '2070-01:2099-12' 
    data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', dataset
    res =       2.8    
    # -- nino index --
    x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_gradients',    'tas_gradients_oni',    r'C',   r'[%]'
    oni_index = get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var)
    # -- conv during el nino --
    conv_regions_daily_nino, mask = elNino_mask(conv_regions_daily, oni_index, timescale = 'daily')
    conv_regions_daily_nino = conv_regions_daily_nino.dropna(dim='time', how='all')
    freq_occur = (conv_regions_daily.fillna(0).sum(dim= 'time') / len(conv_regions_daily.time.data)) * 100
    freq_occur_nino = (conv_regions_daily_nino.fillna(0).sum(dim= 'time') / len(conv_regions_daily_nino.time.data)) * 100
    conv_diff_nino_not_nino = freq_occur_nino - freq_occur
    # -- mean area during el nino --
    x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units = 'daily',    'doc_metrics',  'mean_area',            'mean_area',                        r'A$_m$',   r'[km$^2$]' 
    mean_area = get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var)
    mean_area_nino = mean_area.where(mask)
    mean_area_nino_diff = mean_area_nino.mean(dim = 'time') - mean_area.mean(dim = 'time')
    conv_diff_nino_not_nino.attrs['mean_area_nino_diff'] = mean_area_nino_diff.data
    # -- add to xr.dataset --
    ds[f'{metric_name}_nino_vs_not_nino'] = conv_diff_nino_not_nino


    # == fixed area version ==
    conv_threshold = get_conv_threshold(dataset, years, da, fixed_area = True)
    conv_regions = (da > conv_threshold) * 1
    conv_regions = (conv_regions.resample(time='1MS').mean()) * 100
    conv_regions_anom = cA.detrend_month_anom(conv_regions)
    ds[f'{metric_name}_mean_fixed_area'] =                    conv_regions.mean(dim = 'time')
    ds[f'{metric_name}_variability_month_fixed_area'] =       conv_regions.std(dim='time')    
    ds[f'{metric_name}_variability_month_anom_fixed_area'] =  conv_regions_anom .std(dim='time')

    return ds




