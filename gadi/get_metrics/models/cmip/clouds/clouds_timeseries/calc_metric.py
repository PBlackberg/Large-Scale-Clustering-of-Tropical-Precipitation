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
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')


# == metric funcs ==


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
    # print(wap)
    # [print(f) for f in wap.plev.data]
    # print(wap.sel(plev = 500e2))
    # exit()

    da.coords['time'] = wap.coords['time']
    w500 =              wap.sel(plev = 500e2, method = 'nearest')

    # print(w500)
    # exit()

    tm_ascent =         xr.where(w500.mean(dim = 'time') < 0, 1, 0)  
    tm_descent =        xr.where(w500.mean(dim = 'time') > 0, 1, 0)  
    ascent =            xr.where(w500 < 0, 1, 0)  
    descent =           xr.where(w500 > 0, 1, 0)  
    da_low =            da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev').fillna(0)   # fill nan with zero just in case
    da_high =           da.sel(plev = slice(400e2, 0)).max(dim = 'plev').fillna(0)

    # -- metric --
    # -- low clouds --
    da_area = cW.get_area_matrix(da.lat, da.lon)
    ds[f'{metric_name}_low_mean'] = (da_low * descent * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()                                          # this is timevarying descent area, but is tropically averaged, to not favour small areas with high low cloud fraction
    ds[f'{metric_name}_low_mean_all'] = (da_low * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()                                                # all
    ds[f'{metric_name}_low_mean_tm'] = (da_low * tm_descent * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()                                    # this is time-mean descent
    
    lat_min, lat_max, lon_min, lon_max = -20, -10, 240, 260
    da_low_se = da_low.where((da_low.lat >= lat_min) & (da_low.lat <= lat_max) & (da_low.lon >= lon_min) & (da_low.lon <= lon_max), np.nan) * descent        # pick climate sensitivity region
    ds[f'{metric_name}_low_mean_se'] = (da_low_se.where(~np.isnan(da_low_se), 0) * da_area).sum(dim = ('lat', 'lon')) / (xr.where(~np.isnan(da_low_se), 1, 0) * da_area).sum()


    # -- high clouds --
    ds[f'{metric_name}_high_mean'] = (da_high * ascent * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()
    ds[f'{metric_name}_high_mean_all'] = (da_high * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()                                                # all
    ds[f'{metric_name}_high_mean_tm'] = (da_high * tm_ascent * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()

    da_high_se = da_high.where((da_high.lat >= lat_min) & (da_high.lat <= lat_max) & (da_high.lon >= lon_min) & (da_high.lon <= lon_max), np.nan) * ascent
    ds[f'{metric_name}_high_mean_se'] = (da_high_se.where(~np.isnan(da_high_se), 0) * da_area).sum(dim = ('lat', 'lon')) / (xr.where(~np.isnan(da_high_se), 1, 0) * da_area).sum()  # there are no NaN, because they are filled before, so I think normal weighting is fine here.


    # print(ds)
    # exit()
    return ds




    # ds[f'{metric_name}_high_std'] =  ((da_high * da_area) / da_area.sum()).std(dim = ('lat', 'lon'))
    # ds[f'{metric_name}_low_std'] =  ((da_low * da_area) / da_area.sum()).std(dim = ('lat', 'lon'))

    # da_low_no_overlap = (da_low - da_high) * descent
    # da_low_no_overlap = da_low_no_overlap.where(da_low_no_overlap > 0, 0)

    # da_low_no_overlap_se = da_low_no_overlap.where((da_low_no_overlap.lat >= lat_min) & (da_low_no_overlap.lat <= lat_max) & 
    #                 (da_low_no_overlap.lon >= lon_min) & (da_low_no_overlap.lon <= lon_max), np.nan)
    # ds[f'{metric_name}_low_no_overlap_mean_se'] = (da_low_no_overlap_se.where(~np.isnan(da_low_no_overlap_se), 0) * da_area).sum(dim = ('lat', 'lon')) / (xr.where(~np.isnan(da_low_no_overlap_se), 1, 0) * da_area).sum()

    # ds[f'{metric_name}_low_no_overlap_mean'] = (da_low_no_overlap * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()
    # ds[f'{metric_name}_low_no_overlap_std'] =  ((da_low_no_overlap * da_area) / da_area.sum()).std(dim = ('lat', 'lon'))