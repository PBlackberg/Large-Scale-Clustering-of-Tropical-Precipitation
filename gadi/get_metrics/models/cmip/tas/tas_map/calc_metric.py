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


# == metric funcs ==

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

    # print(da)
    # exit()
    plot = False
    if plot:
        for i, day in enumerate(da['time']):
            folder = f'{os.path.dirname(__file__)}/plots/snapshots'
            filename = f'snapshot_{i}.png'
            path = f'{folder}/{filename}'
            da_plot_here = da
            title = str(da_plot_here.isel(time = i).time.data)[0:10]
            pf_M.plot(da_plot_here.isel(time = i), path, ds_contour = xr.Dataset({'var': da_plot_here.mean(dim = 'time')}), title = title)
            # exit()
    # exit()



    da_anom = cA.detrend_month_anom(da)                                                         # detrended monthly anomalies

    # -- get metric --
    ds[f'{metric_name}_mean'] =                    da.mean(dim = 'time')
    ds[f'{metric_name}_variability_month'] =       da.std(dim='time')    
    ds[f'{metric_name}_variability_month_anom'] =  da_anom .std(dim='time')    

    return ds




