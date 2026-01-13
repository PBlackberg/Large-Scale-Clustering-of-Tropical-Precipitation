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
mS = import_relative_module('user_specs',                                           'utils')
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')
cB = import_relative_module('util_calc.connect_boundaries.connect_lon_boundary',    'utils')
doc = import_relative_module('util_calc.doc_metrics.mean_area.mean_area',           'utils')


# == metric funcs ==
# -- get conv_threshold --
def get_conv_threshold(dataset, years, da, fixed_area = False):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
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
    filename = (                                                                                                             # base result_filename
            f'{metric_name}'   
            f'_{dataset}'                                                                                                 #
            f'_{t_freq}'                                                                                                  #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                         #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                         #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                               #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                   #
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

    # == create metric ==
    # -- check data --
    da, lon_area, lat_area, dataset, years = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )

    # -- convective regions --
    conv_threshold = get_conv_threshold(dataset, years, da)                     # convectionas exceeding precipitation rate
    conv_regions = (da > conv_threshold) * 1
    
    # -- convective objects --
    labels_np = skm.label(conv_regions, background = 0, connectivity = 2)       # returns numpy array
    labels_np = cB.connect_boundary(labels_np)                                  # connect objects across boundary
    labels = np.unique(labels_np)[1:]                                           # first unique value (zero) is background
    
    # -- metric --
    ds[f'{metric_name}'] = xr.DataArray(len(labels))                            # number of connected componenst (objects)

    # == fixed area version ==
    conv_threshold = get_conv_threshold(dataset, years, da, fixed_area=True)    # fixed area
    conv_regions = (da > conv_threshold) * 1
    labels_np = skm.label(conv_regions, background = 0, connectivity = 2)       # returns numpy array
    labels_np = cB.connect_boundary(labels_np)                                  # connect objects across boundary
    labels = np.unique(labels_np)[1:]                                           # first unique value (zero) is background
    ds[f'{metric_name}_fixed_area'] = xr.DataArray(len(labels))
    
    
    # == give time coordinate ==
    ds = ds.expand_dims(dim = 'time')
    ds = ds.assign_coords(time=[da.time.data])

    # print(ds)
    # exit()
    return ds

