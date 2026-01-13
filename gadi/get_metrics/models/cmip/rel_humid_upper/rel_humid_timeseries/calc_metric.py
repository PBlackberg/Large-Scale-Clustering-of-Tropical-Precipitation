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
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')


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
    da = da.sel(plev = 250e2, method='nearest')           # lower troposphere

    # -- timeseries --
    da_area = cW.get_area_matrix(da.lat, da.lon)
    ds[f'{metric_name}_mean'] = (da * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()
    ds[f'{metric_name}_std'] =  da.std(dim = ('lat', 'lon'))

    # -- add reigon that is correlated --
    lat_min, lat_max, lon_min, lon_max = -20, -5, 235, 270
    da_se = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), np.nan)
    ds[f'{metric_name}_mean_se'] = (da_se.where(~np.isnan(da_se), 0) * da_area).sum(dim = ('lat', 'lon')) / (xr.where(~np.isnan(da_se), 1, 0) * da_area).sum()


    return ds




