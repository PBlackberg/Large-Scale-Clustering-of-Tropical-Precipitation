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
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')
doc = import_relative_module('util_calc.doc_metrics.area_fraction.area_fraction',   'utils')


# == metric funcs ==
# -- get conv_threshold --
def get_conv_threshold(dataset, years, da, fixed_area = False):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)       # user settings
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
    threshold = xr.open_dataset(path)

    # # -- find metric -- 
    # if not fixed_area:
    #     threshold = xr.open_dataset(path)[metric_var].mean(dim = 'time')
    #     da_threshold = threshold.broadcast_like(da.isel(lat = 0, lon = 0))
    # else:
    #     threshold = xr.open_dataset(path)[metric_var]
    #     da_threshold = threshold.sel(time = da.time, method='nearest')
    return threshold


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
    conv_thresholds = get_conv_threshold(dataset, years, da)
    # print(conv_thresholds)
    # exit()

    # -- for area weighting --
    da_area = cW.get_area_matrix(da.lat, da.lon)

    # -- threshold variations --
    quantile_thresholds = [0.90, 0.95, 0.97] #, 0.97, 0.99] #0.9, 
    for quant in quantile_thresholds:
        quant_str = f'precip_prctiles_{int(quant * 100)}'
        conv_regions = (da > conv_thresholds[quant_str].mean(dim = 'time').data) * 1
        # print((conv_regions * da_area).sum())
        # print(type((conv_regions * da_area).sum()))
        # exit()

        # -- fill xr.dataset with metric --
        ds[f'{metric_name}_thres_{quant_str}'] = (conv_regions * da_area).sum(dim = ('lat', 'lon'))

    # == fixed area version ==
    # conv_threshold = get_conv_threshold(dataset, years, da, fixed_area = True)
    # conv_regions = (da > conv_threshold) * 1
    # ds[f'{metric_name}_fixed_area'] = doc.area_fraction(conv_regions, da_area)

    # print(ds)
    # exit()
    return ds



