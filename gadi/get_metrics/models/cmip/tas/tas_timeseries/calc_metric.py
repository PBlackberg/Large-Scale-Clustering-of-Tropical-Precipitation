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
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')


# == metric funcs ==
# -- estimated climate sensitivity --
def get_ecs(model):
    ecs = {
        'INM-CM4-8':        1.83, 
        'INM-CM5-0':        1.92, 
        'IITM-ESM':         2.37, 
        'NorESM2-MM':       2.49, 
        'NorESM2-LM':       2.56, 
        'MIROC6':           2.6,
        'GFDL-ESM4':        2.65, 
        'MIROC-ES2L':       2.66, 
        'FGOALS-g3':        2.87, 
        'MPI-ESM1-2-LR':    3.02,
        'BCC-CSM2-MR':      3.02, 
        'MRI-ESM2-0':       3.13,  
        'KIOST-ESM':        3.36, 
        'CMCC-CM2-SR5':     3.56, 
        'CMCC-ESM2':        3.58, 
        'CMCC-ESM2':        3.58, 
        'ACCESS-ESM1-5':    3.88, 
        'GFDL-CM4':         3.89, 
        'EC-Earth3':        4.26, 
        'CNRM-CM6-1-HR':    4.34, 
        'TaiESM1':          4.36, 
        'ACCESS-CM2':       4.66, 
        'CESM2-WACCM':      4.68, 
        'IPSL-CM6A-LR':     4.70,   
        'NESM3':            4.72, 
        'KACE-1-0-G':       4.75, 
        'CNRM-ESM2-1':      4.79, 
        'CNRM-CM6-1':       4.90, 
        'CESM2':            5.15, 
        'UKESM1-0-LL':      5.36, 
        'CanESM5':          5.64
        }
    return ecs[model]

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

    # -- timeseries --
    da_area = cW.get_area_matrix(da.lat, da.lon)
    ds[f'{metric_name}_mean'] = (da * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()
    ds[f'{metric_name}_std'] =  da.std(dim = ('lat', 'lon'))
    ds[f'{metric_name}_ecs'] = xr.DataArray([get_ecs(dataset)] * len(da.time), dims=('time'), coords={'time': da['time'].data})
    # print(ds[f'{metric_name}_ecs'])
    # exit()
    return ds




