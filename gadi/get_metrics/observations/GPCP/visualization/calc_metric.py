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
pf_M = import_relative_module('helper_funcs.plot_func_map',                         __file__)


# == metric funcs ==
# -- get conv_threshold --
def get_conv_threshold(dataset, years, da, time_period, fixed_area = False):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- specify metric --
    data_tyoe_group =   'observations'
    data_type =         'GPCP'
    metric_group =      'precip'
    metric_name =       'precip_prctiles'
    metric_var =        'precip_prctiles_95'
    dataset =           dataset
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8

    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                        # base result_filename
            f'{metric_name}'                                                                                            #
            f'_{dataset}'                                                                                               #
            f'_{t_freq}'                                                                                                #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 #
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


# -- metric --
def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                        # base result_filename
            f'{metric_name}'                                                                                            #
            f'_{dataset}'                                                                                               #
            f'_{t_freq}'                                                                                                #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 #
            )       
    path = f'{folder}/{filename}.nc'
    # -- find metric -- 
    doc_metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(doc_metric)
        print('exiting')
        exit()
    else:
        # -- get metric variation -- 
        doc_metric = doc_metric[metric_var]
        doc_metric = doc_metric.sel(time = slice('1998-01', '2017-06'))
        month = 2
    return doc_metric.sel(time = doc_metric['time.month'] == month) 

# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name

    # -- check data --
    da, lon_area, lat_area, dataset, years, hur, cl_low = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    hur = hur.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    cl_low = cl_low.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )

    # -- conv as exceeding precipitation threshsold --
    conv_threshold = get_conv_threshold(dataset, years, da, time_period = '1998-01:2022-12')
    conv_regions = (da > conv_threshold) * 1
    conv_regions = conv_regions.where(conv_regions != 0, np.nan)                                            
    hur = hur.resample(time = 'MS').mean()                                                                          # coordinate at the start of the month
    cl_low = cl_low.resample(time = 'MS').mean()            

    # -- lines for plot --
    ds_line_zonal, da_line_zonal = ref.zonal_line(conv_regions.isel(time = 0), lat_centre = 0)                      # meridional line
    ds_line_zonal.attrs.update({'color': 'k', 'dashes': (1.5, 1), 'linewidth': 1})                                  # plot settings
    lon_centre = 180                                                                                                # zonal line
    ds_line_meridional, da_line_meridional = ref.meridional_line(conv_regions.isel(time = 0), lon_centre)           #
    ds_line_meridional.attrs.update({'color': 'k', 'dashes': (1.5, 1), 'linewidth': 1})                             # plot settings

    # -- replicate the monthly values to a daily data array, with the nearest month (do da after distance calc) -- 
    da = da.sel(time = slice('1998-01', '2017-06'))
    conv_regions = conv_regions.sel(time = slice('1998-01', '2017-06'))
    hur = hur.sel(time = slice('1998-01', '2017-06'))
    cl_low = cl_low.sel(time = slice('1998-01', '2017-06'))
    hur, cl_low = [da.reindex(time=conv_regions['time'], method='nearest') for da in [hur, cl_low]]

    # -- pick out a month, to show variations resulting in deseasonalized monthly anomalies --
    month = 2
    da, conv_regions, hur, cl_low = [f.sel(time = f['time.month'] == month) for f in [da, conv_regions, hur, cl_low]]

    # -- plots for animation --
    plot = False
    if plot:
        for i, day in enumerate(conv_regions['time']):
            folder = f'{os.path.dirname(__file__)}/plots'
            filename = f'conv_itcz_cl_{i}.png'
            path = f'{folder}/{filename}'
            title = f'time:{str(day.data)[0:10]}'
            pf_M.plot(da =          da.sel(time = day).where(da.sel(time = day) > 1, np.nan), 
                       temp_path =  path, 
                       lines =      [ds_line_meridional, ds_line_zonal], 
                       ds_ontop =   xr.Dataset({'var': conv_regions.sel(time = day)}), 
                       ds_ontop2 =  xr.Dataset({'var': cl_low.sel(time = day)}), 
                       ds_contour = xr.Dataset({'var': hur.sel(time = day)}), 
                       title =      title
                       )
            if i > 10:
                break
    # exit()

    # -- plots for specific scenes --
    # -- constrain area fraction --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'observations', 'GPCP', dataset, 'daily', 'doc_metrics', 'area_fraction', lon_area, lat_area, 2.8, '1998-01:2022-12', 'area_fraction'
    area_fraction = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)

    # -- get mean area --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'observations', 'GPCP', dataset, 'daily', 'doc_metrics', 'mean_area', lon_area, lat_area, 2.8, '1998-01:2022-12', 'mean_area'
    mean_area = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)

    # -- get elnino condition --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'observations', 'NOAA', 'NOAA', 'monthly', 'tas', 'tas_gradients', lon_area, lat_area, 2.8, '1998-01:2022-12', 'tas_gradients_oni'
    oni = get_metric(data_tyoe_group, data_type, 'NOAA', t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)
    oni = oni.ffill(dim='time')
    oni = oni.bfill(dim='time')   
    oni =  [da.reindex(time=conv_regions['time'], method='nearest') for da in [oni]][0]

    # -- get number index --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'observations', 'GPCP', 'GPCP', 'daily', 'doc_metrics', 'number_index', lon_area, lat_area, 2.8, '1998-01:2022-12', 'number_index'
    number_index = get_metric(data_tyoe_group, data_type, 'GPCP', t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)
    
    # -- find constrained min, median, and max --
    lower_bound = 0.04
    upper_bound = 0.06
    doc_min =       mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).min()
    doc_median =    mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).median()
    doc_max =       mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).max()

    # -- find index of min, median, and max --
    doc_min_index =     np.abs(mean_area - doc_min).argmin()
    doc_median_index =  np.abs(mean_area - doc_median).argmin()
    doc_max_index =     np.abs(mean_area - doc_max).argmin()

    # -- example plots --
    label1 = r'A$_f$'
    label2 = r'A$_m$'
    units = r'km$^{2}$'
    plot = True
    if plot:
        for i, day in enumerate([doc_min_index.data, doc_median_index.data, doc_max_index.data]):
            folder = f'{os.path.dirname(__file__)}/plots2'
            filename = f'conv_itcz_cl_{i}.svg'
            path = f'{folder}/{filename}'
            pf_M.plot(da =         da.isel(time = day).where(da.isel(time = day) > 1, np.nan), 
                       temp_path =  path, 
                       lines =      [ds_line_meridional, ds_line_zonal], 
                       ds_ontop =   xr.Dataset({'var': conv_regions.isel(time = day)}), 
                       ds_ontop2 =   xr.Dataset({'var': cl_low.isel(time = day)}), 
                       ds_contour = xr.Dataset({'var': hur.isel(time = day)}), 
                       title = f'   {label1}: {area_fraction.isel(time = day).data * 100:.1f}%, {label2}: {mean_area.isel(time = day).data:.1e} {units}, N: {number_index.isel(time = day).data}, ONI: {oni.isel(time = day).data:.2f} K'
                       )
            # exit()
    exit()

    return ds






