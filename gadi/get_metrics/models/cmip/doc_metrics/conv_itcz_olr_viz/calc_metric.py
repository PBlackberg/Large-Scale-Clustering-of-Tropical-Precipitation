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
pf_M2 = import_relative_module('helper_funcs.plot_func_map2',                         __file__)


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


# -- metric --
def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
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
    return doc_metric



# == calculate metric ==
def calculate_metric(data_objects):
    # -- create empty metric --
    metric_name = Path(__file__).resolve().parents[0].name

    # -- check data --
    da, lon_area, lat_area, dataset, years, hus, olr = data_objects

    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    hus = hus.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    olr = olr.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )
    
    # -- select a particular month -- 
    month = 2
    da =    da.sel(time=da.time.dt.month == month)
    # hus =   hus.sel(time=hus.time.dt.month == month)
    # olr =   olr.sel(time=olr.time.dt.month == month)
    # [print(f) for f in [da, hus, olr]]
    # exit()

    # -- conv as exceeding precipitation threshsold --
    conv_threshold = get_conv_threshold(dataset, years, da)
    conv_regions = (da > conv_threshold) * 1
    conv_regions = conv_regions.where(conv_regions != 0, np.nan)                                                                            # for not later spatially averaging over zeros 
    hus = hus.resample(time = 'MS').mean()                                                                                                  # daily coordinate at the start of the month
    olr = olr.resample(time = 'MS').mean()            

    # -- reference line --
    ds_line_eq_hydro, da_line_eq_hydro = ref.hyrdo_eq_line(conv_regions.isel(time = 0), hus)                                                # hydrological equator (function of time in months)

    # -- replicate the monthly values to a daily data array, with the nearest month (do da after distance calc) -- 
    ds_line_eq_hydro, hus, olr =  [da.reindex(time=conv_regions['time'], method='nearest') for da in [ds_line_eq_hydro, hus, olr]]

    # print(hus.quantile(0.1))
    # exit()

    # -- plots for animation --
    plot = False
    if plot:
        # folder = '/home/565/cb4968/gadi/get_metrics/models/cmip/doc_metrics/reference_proximity'
        # for i, day in enumerate(conv_regions['time'].sel(time = '1970')):
        for i, day in enumerate(conv_regions['time']):
            folder = f'{os.path.dirname(__file__)}/plots'
            filename = f'conv_itcz_olr_{i}.png'
            path = f'{folder}/{filename}'
            # pf_M.plot(hydro_eq_median_distance.isel(time = i), path)
            pf_M.plot(hus.isel(time = i) , path, lines = [ds_line_eq_hydro.isel(time = i)], ds_ontop = xr.Dataset({'var': conv_regions.isel(time = i)}), ds_contour = xr.Dataset({'var': olr.isel(time = i)}))
            # if i == 60:
            #     exit()
            # exit()
    # exit()
    # -- plots for specific scenes --
    # -- area fraction --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'models', 'cmip', dataset, 'daily', 'doc_metrics', 'area_fraction', lon_area, lat_area, 2.8, '1970-01:1999-12', 'area_fraction'
    area_fraction = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)

    # -- mean area --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'models', 'cmip', dataset, 'daily', 'doc_metrics', 'mean_area', lon_area, lat_area, 2.8, '1970-01:1999-12', 'mean_area'
    mean_area = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)

    # -- elnino condition --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'models', 'cmip', dataset, 'monthly', 'tas', 'tas_gradients', lon_area, lat_area, 2.8, '1970-01:1999-12', 'tas_gradients_oni'
    oni = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)
    oni = oni.ffill(dim='time')
    oni = oni.bfill(dim='time')   
    # print(oni)
    # exit()
    oni =  [da.reindex(time=conv_regions['time'], method='nearest') for da in [oni]][0]
    # print(oni.max())
    # print(oni.min())
    # exit()

    # -- number index --
    data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var = 'models', 'cmip', dataset, 'daily', 'doc_metrics', 'number_index', lon_area, lat_area, 2.8, '1970-01:1999-12', 'number_index'
    number_index = get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var)
    
    area_fraction, mean_area, number_index =  [da.reindex(time=conv_regions['time'], method='nearest') for da in [area_fraction, mean_area, number_index]]

    # -- find min, median, and max --
    lower_bound = 0
    upper_bound = 1
    doc_min =       mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).min()
    doc_median =    mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).median()
    doc_max =       mean_area.where((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound), np.nan).max()
    # [print(f) for f in [doc_min, doc_median, doc_max]]

    # -- find index of min, median, and max --
    doc_min_index =     np.abs(mean_area - doc_min).argmin()
    doc_median_index =  np.abs(mean_area - doc_median).argmin()
    doc_max_index =     np.abs(mean_area - doc_max).argmin()
    # [print(f) for f in [doc_min_index, doc_median_index, doc_max_index]]
    # exit()

    # -- plots example --
    label1 = r'A$_f$'
    label2 = r'A$_m$'
    units = r'km$^{2}$'
    plot = False
    if plot:
        for i, day in enumerate([doc_min_index.data, doc_median_index.data, doc_max_index.data]):
            folder = f'{os.path.dirname(__file__)}/plots2'
            filename = f'conv_itcz_olr_{i}.png'
            path = f'{folder}/{filename}'
            pf_M2.plot(da =         hus.isel(time = day) , 
                       temp_path =  path, 
                       lines =      [ds_line_eq_hydro.isel(time = day)], 
                       ds_ontop =   xr.Dataset({'var': conv_regions.isel(time = day)}), 
                       ds_contour = xr.Dataset({'var': olr.isel(time = day)}), 
                       title = f'model: {dataset}, month: {month}            {label1}: {area_fraction.isel(time = day).data * 100:.1f}%,         {label2}: {mean_area.isel(time = day).data:.1e} {units},         N: {number_index.isel(time = day).data}         ONI: {oni.isel(time = day).data:.2f} K'
                       )
            # exit()
    # exit()
    # print(ds_line_eq_hydro.isel(time = doc_min_index.data))
    # exit()


    # -- put data arrays in dataset --
    ds = xr.Dataset()

    day = doc_min_index.data
    da = olr.isel(time = day)
    da.attrs =  {   'area_fraction':    f'{area_fraction.isel(time =    day).data * 100:.1f}%',
                    'mean_area':        f'{mean_area.isel(time =        day).data:.1e}',
                    'N':                f'{number_index.isel(time =     day).data}',
                    'ONI':              f'{oni.isel(time =              day).data:.2f} K'
                 }
    ds['olr0'] =            da
    ds['hus0'] =            hus.isel(time =                 day)
    ds['conv_regions0'] =   conv_regions.isel(time =        day)
    ds['line0'] =           ds_line_eq_hydro.isel(time =    day)['var']


    day = doc_median_index.data
    da = olr.isel(time = day)
    da.attrs =  {   'area_fraction':    f'{area_fraction.isel(time =    day).data * 100:.1f}%',
                    'mean_area':        f'{mean_area.isel(time =        day).data:.1e}',
                    'N':                f'{number_index.isel(time =     day).data}',
                    'ONI':              f'{oni.isel(time =              day).data:.2f} K'
                 }
    ds['olr1'] =            da
    ds['hus1'] =            hus.isel(time =                 day)
    ds['conv_regions1'] =   conv_regions.isel(time =        day)
    ds['line1'] =           ds_line_eq_hydro.isel(time =    day)['var']


    day = doc_max_index.data
    da = olr.isel(time = day)
    da.attrs =  {   'area_fraction':    f'{area_fraction.isel(time =    day).data * 100:.1f}%',
                    'mean_area':        f'{mean_area.isel(time =        day).data:.1e}',
                    'N':                f'{number_index.isel(time =     day).data}',
                    'ONI':              f'{oni.isel(time =              day).data:.2f} K'
                 }
    ds['olr2'] =            da
    ds['hus2'] =            hus.isel(time =                 day)
    ds['conv_regions2'] =   conv_regions.isel(time =        day)
    ds['line2'] =           ds_line_eq_hydro.isel(time =    day)['var']

    # print(ds)
    # exit()

    # -- plots test --
    plot = False
    if plot:
        folder = f'{os.path.dirname(__file__)}/plots3'
        day = '0'
        filename = f'conv_itcz_olr_{day}.png'
        path = f'{folder}/{filename}'
        pf_M2.plot(da =         ds[f'hus{day}'], 
                    temp_path =  path, 
                    lines =      [xr.Dataset({'var': ds[f'line{day}']})], 
                    ds_ontop =   xr.Dataset({'var': ds[f'conv_regions{day}']}), 
                    ds_contour = xr.Dataset({'var': ds[f'olr{day}']}), 
                    title = f"model: {dataset},            {label1}: {ds[f'olr{day}'].attrs['area_fraction']},         {label2}: {ds[f'olr{day}'].attrs['mean_area']} {units},         N: {ds[f'olr{day}'].attrs['N']}          ONI: {ds[f'olr{day}'].attrs['ONI']}"
                    )
        # exit()
        day = '1'
        filename = f'conv_itcz_olr_{day}.png'
        path = f'{folder}/{filename}'
        pf_M2.plot(da =         ds[f'hus{day}'], 
                    temp_path =  path, 
                    lines =      [xr.Dataset({'var': ds[f'line{day}']})], 
                    ds_ontop =   xr.Dataset({'var': ds[f'conv_regions{day}']}), 
                    ds_contour = xr.Dataset({'var': ds[f'olr{day}']}), 
                    title = f"model: {dataset},            {label1}: {ds[f'olr{day}'].attrs['area_fraction']},         {label2}: {ds[f'olr{day}'].attrs['mean_area']} {units},         N: {ds[f'olr{day}'].attrs['N']}          ONI: {ds[f'olr{day}'].attrs['ONI']}"
                    )

        day = '2'
        filename = f'conv_itcz_olr_{day}.png'
        path = f'{folder}/{filename}'
        pf_M2.plot(da =         ds[f'hus{day}'], 
                    temp_path =  path, 
                    lines =      [xr.Dataset({'var': ds[f'line{day}']})], 
                    ds_ontop =   xr.Dataset({'var': ds[f'conv_regions{day}']}), 
                    ds_contour = xr.Dataset({'var': ds[f'olr{day}']}), 
                    title = f"model: {dataset},            {label1}: {ds[f'olr{day}'].attrs['area_fraction']},         {label2}: {ds[f'olr{day}'].attrs['mean_area']} {units},         N: {ds[f'olr{day}'].attrs['N']}          ONI: {ds[f'olr{day}'].attrs['ONI']}"
                    )


            # exit()
    # print(ds)
    # exit()

    return ds





if __name__ == '__main__':
    ''
    # ds_line_centre, da_line_centre =                                ref.centre_point(conv_regions.isel(time = 0), lon_centre)             # centre point

    # lower_bound = 0.04
    # upper_bound = 0.06
    # area_fraction_med = ((area_fraction.data >= lower_bound) & (area_fraction.data <= upper_bound))

    # # lower_bound = mean_area.quantile(0).data
    # # upper_bound = mean_area.quantile(0.2).data
    # # doc_low = ((mean_area.data >= lower_bound) & (mean_area.data <= upper_bound))

    # # lower_bound = mean_area.quantile(0.4).data
    # # upper_bound = mean_area.quantile(0.6).data
    # # doc_med = ((mean_area.data >= lower_bound) & (mean_area.data <= upper_bound))

    # # lower_bound = mean_area.quantile(0.8).data
    # # upper_bound = mean_area.quantile(1).data
    # # doc_high = ((mean_area.data >= lower_bound) & (mean_area.data <= upper_bound))

    # # doc_low_constrained = (area_fraction_med * doc_low) * mean_area
    # # doc_med_constrained = (area_fraction_med * doc_med) * mean_area
    # # doc_high_constrained = (area_fraction_med * doc_high) * mean_area