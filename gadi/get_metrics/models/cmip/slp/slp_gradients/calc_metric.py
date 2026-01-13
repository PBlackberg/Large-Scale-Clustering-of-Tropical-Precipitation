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
pf_M = import_relative_module('helper_funcs.plot_func_map',                         __file__)


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

    da = da.resample(time='1MS').mean()
    da_darwin = da.sel(lat = -12.5, lon = 130.9, method='nearest')
    da_tahiti = da.sel(lat = -17.5, lon = 210.4, method='nearest')
    dp = da_tahiti - da_darwin

    dp_anom = dp.groupby('time.month') - dp.groupby('time.month').mean('time')
    dp_standardized = dp_anom.groupby('time.month') / dp_anom.groupby('time.month').std('time', ddof=1)
    dp_standardized_rolling = dp_standardized.rolling(time=3, center=True).mean()
    dp_standardized_rolling = dp_standardized_rolling.ffill(dim='time')
    dp_standardized_rolling = dp_standardized_rolling.bfill(dim='time')         
    # exit()

    # -- calculate gradients --
    ds[f'{metric_name}_SOI'] = dp_standardized_rolling

    # print(ds)
    # print(np.shape(ds[f'{metric_name}_SOI'].data))
    # exit()

    plot = False
    if plot:
        for i, day in enumerate(da['time']):
            folder = f'{os.path.dirname(__file__)}/plots/snapshots'
            filename = f'snapshot_{i}.png'
            path = f'{folder}/{filename}'
            spatial_mean = da.mean(dim = ('lat', 'lon'))
            da_plot = da - spatial_mean                         # anomalies from the spatial-mean
            da_plot_here =  da_plot / da_plot.std()              # standard deviation of anomalies
            title = str(da_plot_here.isel(time = i).time.data)[0:10]
            lat_coords = np.array([-12.5, -17.5])
            lon_coords = np.array([130.9, 210.4])
            pf_M.plot(da_plot_here.isel(time = i), path, ds_contour = xr.Dataset({'var': da_plot_here.mean(dim = 'time')}), title = title, lon_coords = lon_coords, lat_coords = lat_coords)
            exit()
    # exit()


    return ds




