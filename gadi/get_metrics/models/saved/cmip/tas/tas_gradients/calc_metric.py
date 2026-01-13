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
cG = import_relative_module('util_calc.gradients.map_gradients',                    'utils')


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

    # -- do rolling mean anomalies --
    da_anom = cA.detrend_month_anom(da)
    da_anom_rolling = da_anom.rolling(time=3, center=True).mean()

    # -- calculate gradients --
    ds[f'{metric_name}_oni'] =                      cG.oni_calc(da)
    ds[f'{metric_name}_pacific_zonal'] =            cG.zonal_gradient_calc(da_anom_rolling)
    ds[f'{metric_name}_pacific_meridional'] =       cG.meridional_gradient_calc(da_anom_rolling)

    ds[f'{metric_name}_pacific_zonal_clim'] =       cG.zonal_gradient_calc(da.mean(dim = 'time')).broadcast_like(da.isel(lat = 0, lon = 0).resample(time='1MS').mean())
    ds[f'{metric_name}_pacific_meridional_clim'] =  cG.meridional_gradient_calc(da.mean(dim = 'time')).broadcast_like(da.isel(lat = 0, lon = 0).resample(time='1MS').mean())

    return ds




