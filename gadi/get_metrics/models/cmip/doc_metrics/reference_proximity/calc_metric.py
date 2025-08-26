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
ref = import_relative_module('util_calc.doc_metrics.itcz_features.reference_lines', 'utils')
dM = import_relative_module('util_calc.distance_matrix.distance_matrix',            'utils')
pf_M = import_relative_module('helper_funcs.plot_func_map',                         __file__)


# == metric funcs ==
# -- get conv_threshold --
def get_conv_threshold(dataset, years, da, fixed_area = False):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                               # user settings
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
    da, lon_area, lat_area, dataset, years, hus = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    hus = hus.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )

    # -- conv as exceeding precipitation threshsold --
    conv_threshold = get_conv_threshold(dataset, years, da)
    conv_regions = (da > conv_threshold) * 1
    conv_regions = conv_regions.where(conv_regions != 0, np.nan)                                                                            # To not spatially averaging over zeros 
    hus = hus.resample(time = 'MS').mean()                                                                                                  # daily coordinate at the start of the month

    # -- references --
    if conv_regions.lon.max().data >= 350 and conv_regions.lon.min().data <= 10:
        lon_centre = 180
    else:
        lon_centre = (conv_regions.lon.max().data - conv_regions.lon.min().data) / 2
    ds_line_meridional, da_line_meridional =                        ref.meridional_line(conv_regions.isel(time = 0),        lon_centre)     # zonal contraction
    ds_line_zonal, da_line_zonal =                                  ref.zonal_line(conv_regions.isel(time = 0),             lat_centre = 0) # meridional contraction
    ds_line_eq_hydro, da_line_eq_hydro =                            ref.hyrdo_eq_line(conv_regions.isel(time = 0),          hus)            # hydrological equator (function of time in months)
    ds_line_eq_hydro_median, da_line_eq_hydro_median, median_lat =  ref.hydro_eq_median_line(conv_regions.isel(time = 0),   hus)            # median hydrological position (function of time in months)

    # -- replicate the monthly values to a daily data array, with the nearest month (do da after distance calc) -- 
    ds_line_eq_hydro =                      ds_line_eq_hydro.reindex(time=conv_regions['time'], method='nearest')
    ds_line_eq_hydro_median, median_lat =  [da.reindex(time=conv_regions['time'], method='nearest') for da in [ds_line_eq_hydro_median, median_lat]]

    # -- distance to reference --
    distance_matrix = dM.create_distance_matrix(conv_regions.lat, conv_regions.lon)
    hydro_eq_distance = xr.concat([dM.find_distance(da_line_eq_hydro.isel(time = month), distance_matrix).assign_coords(time = time_month) for month, time_month in enumerate(da_line_eq_hydro['time'])], dim = 'time')
    hydro_eq_median_distance = xr.concat([dM.find_distance(da_line_eq_hydro_median.isel(time = month), distance_matrix).assign_coords(time = time_month) for month, time_month in enumerate(da_line_eq_hydro_median['time'])], dim = 'time')

    # -- replicate the monthly values to a daily data array --
    hydro_eq_distance, hydro_eq_median_distance = [da.reindex(time=conv_regions['time'], method='nearest') for da in [hydro_eq_distance, hydro_eq_median_distance]]

    ds[f'{metric_name}_meridional_line'] =      (dM.find_distance(da_line_meridional,   distance_matrix) * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_zonal_line'] =           (dM.find_distance(da_line_zonal,        distance_matrix) * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro'] =             (hydro_eq_distance * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro_median'] =      (hydro_eq_median_distance  * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro_median_pos'] =  median_lat

    # -- visualize --
    plot = False
    if plot:
        da_plot_here = hydro_eq_distance
        for i, day in enumerate(hydro_eq_distance['time'].sel(time = '1970')):
            folder = f'{os.path.dirname(__file__)}/plots/snapshots'
            filename = f'snapshot_{i}.png'
            path = f'{folder}/{filename}'
            title = str(da_plot_here.isel(time = i).time.data)[0:10]
            pf_M.plot(da_plot_here.isel(time = i), path, lines = [ds_line_eq_hydro.isel(time = i)], ds_ontop = xr.Dataset({'var': conv_regions.isel(time = i)}), ds_contour = xr.Dataset({'var': conv_regions.fillna(0).mean(dim = 'time')}), title = title)
            # exit()    
    # exit()

    # == fixed area version ==
    conv_threshold = get_conv_threshold(dataset, years, da, fixed_area = True)
    conv_regions = (da > conv_threshold) * 1
    conv_regions = conv_regions.resample(time = 'MS').mean()
    ds[f'{metric_name}_meridional_line_fixed_area'] =           (dM.find_distance(da_line_meridional,   distance_matrix) * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_zonal_line_fixed_area'] =                (dM.find_distance(da_line_zonal,        distance_matrix) * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro_fixed_area'] =                  (hydro_eq_distance * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro_median_fixed_area'] =           (hydro_eq_median_distance  * conv_regions).mean(dim = ('lat', 'lon'))
    ds[f'{metric_name}_eq_hydro_median_pos_fixed_area'] =       median_lat

    return ds

