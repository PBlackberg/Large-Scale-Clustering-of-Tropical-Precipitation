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
pf_M = import_relative_module('helper_funcs.plot_func_map',                         __file__)


# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name
    ds = xr.Dataset()

    # -- check data --
    da, lon_area, lat_area, dataset, years, wap = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    da.coords['time'] = wap.coords['time']
    w500 =      wap.sel(plev = 500e2)
    ascent =    xr.where(w500 < 0, 1, 0)  
    descent =   xr.where(w500 > 0, 1, 0)  
    da_low =    da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev').fillna(0)   # fill nan with zero just in case
    da_high =   da.sel(plev = slice(400e2, 0)).max(dim = 'plev').fillna(0)
    da_low_no_overlap = (da_low - da_high) * descent
    da_low_no_overlap = da_low_no_overlap.where(da_low_no_overlap > 0, 0)
    da_low =            da_low  * descent
    da_high =           da_high * ascent

    # -- visualize --         
    plot = False
    if plot:
        for i, day in enumerate(da['time']):
            folder = f'{os.path.dirname(__file__)}/plots/snapshots'
            filename = f'snapshot_{i}.png'
            path = f'{folder}/{filename}'
            da_plot_here = da_high
            title = str(da_plot_here.isel(time = i).time.data)[0:10]
            pf_M.plot(da_plot_here.isel(time = i), path, ds_contour = xr.Dataset({'var': da_plot_here.mean(dim = 'time')}), title = title)
            break
    #         exit()
    # exit()

    # -- metric --
    da_low_anom = cA.detrend_month_anom(da_low)
    ds[f'{metric_name}_low_mean'] =                                 da_low.mean(dim = 'time')
    ds[f'{metric_name}_low_variability_month'] =                    da_low.std(dim = 'time')    
    ds[f'{metric_name}_low_variability_month_anom'] =               da_low_anom.std(dim = 'time')     

    da_low_no_overlap_anom = cA.detrend_month_anom(da_low_no_overlap)
    ds[f'{metric_name}_low_mean_no_overlap'] =                      da_low_no_overlap.mean(dim = 'time')
    ds[f'{metric_name}_low_variability_month_no_overlap'] =         da_low_no_overlap.std(dim = 'time')    
    ds[f'{metric_name}_low_variability_month_anom_no_overlap'] =    da_low_no_overlap_anom.std(dim = 'time')     

    da_high_anom = cA.detrend_month_anom(da_high)
    ds[f'{metric_name}_high_mean'] =                                da_high.mean(dim = 'time')
    ds[f'{metric_name}_high_variability_month'] =                   da_high.std(dim = 'time')    
    ds[f'{metric_name}_high_variability_month_anom'] =              da_high_anom.std(dim = 'time')     

    # print(ds)
    # exit()

    # # -- add sub-reigon --
    # lat_min, lat_max, lon_min, lon_max = -20, -10, 240, 260
    # da_low_se = da_low.where((da_low.lat >= lat_min) & (da_low.lat <= lat_max) & 
    #                 (da_low.lon >= lon_min) & (da_low.lon <= lon_max), np.nan)
    # ds[f'{metric_name}_low_mean_se'] = da_low_se.mean(dim = 'time')

    # da_low_no_overlap_se = da_low_no_overlap.where((da_low_no_overlap.lat >= lat_min) & (da_low_no_overlap.lat <= lat_max) & 
    #                 (da_low_no_overlap.lon >= lon_min) & (da_low_no_overlap.lon <= lon_max), np.nan)
    # ds[f'{metric_name}_low_no_overlap_mean_se'] = da_low_no_overlap_se.mean(dim = 'time')

    # da_high_se = da_high.where((da_high.lat >= lat_min) & (da_high.lat <= lat_max) & 
    #                 (da_high.lon >= lon_min) & (da_high.lon <= lon_max), np.nan)
    # ds[f'{metric_name}_high_mean_se'] = da_high_se.mean(dim = 'time')

    return ds




