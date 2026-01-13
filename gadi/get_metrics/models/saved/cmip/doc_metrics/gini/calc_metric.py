'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import skimage.measure as skm

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
mS = import_relative_module('user_specs',                                                   'utils')
doc = import_relative_module('util_calc.doc_metrics.area_fraction.area_fraction',   'utils')
doc2 = import_relative_module('util_calc.doc_metrics.I_org.I_org_calc',                     'utils')
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',                   'utils')
cB = import_relative_module('util_calc.connect_boundaries.connect_lon_boundary',            'utils')


# == helper funcs ==
def get_metric_threshold(time_period):
    ''' 
    convective threshold: 16.107049368020803 mm/day (95th percentile)
    /Users/cbla0002/Desktop/work/metrics/observations/GPCP/precip/precip_prctiles/GPCP/precip_prctiles_GPCP_daily_0-360_-30-30_128x64_1998-01_2022-12.nc
    '''
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False) # user settings
    # -- specify metric --
    data_type_group =   'observations'
    data_type =         'GPCP'
    metric_group =      'precip'
    metric_name =       'precip_prctiles'
    dataset =           'GPCP'
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8
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
    path = f'{folder}/{r_filename}.nc' # fiels are stored in years here
    # -- get metric --
    thresholds = xr.open_dataset(path).load()
    # threshold = threshold.mean(dim = 'time').data                                        
    # print(threshold)
    # exit()
    # try:
    #     threshold = xr.open_dataset(path, combine='by_coords')[metric_var].load()
    #     threshold = threshold.mean(dim = 'time').data                                        
    #     print(threshold)
    # except:
    #     print('get_metric_threshold function in calc_metric.py didnt work')
    #     print(f'prob couldnt open metric file, check files with structure: {path}')
    #     print('try regenerating if it does not exist')
    #     print('exiting')
    #     exit()
    return thresholds


def gini(x):
    x = np.asarray(x).ravel()          # flatten (lat, lon) → 1D
    x = x[np.isfinite(x)]              # drop NaNs if any
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n  # 0–1

# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = f'{Path(__file__).resolve().parents[0].name}'
    ds = xr.Dataset()

    # == create metric ==
    # -- check data and loaded metrics --
    da, lon_area, lat_area, dataset, years = data_objects
    # print(da)
    # exit()
        
    # -- fill xr.dataset with metric --
    ds[f'{metric_name}'] = gini(da.data)

    # == give time coordinate ==
    ds = ds.expand_dims(dim = 'time')
    ds = ds.assign_coords(time=[da.time.data])

    # print(ds)
    # print(ds[f'{metric_name}'].data)
    # exit()


    return ds




