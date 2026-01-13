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
gC = import_relative_module('util_calc.correlations.gridbox_correlation',           'utils')


# == metric funcs ==
# -- doc metric --
def get_doc_metric(dataset, years, da, metric_name, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'models'
    data_type =         'cmip'
    metric_group =      'doc_metrics'
    metric_name =       metric_name
    metric_var =        metric_var
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
            f'_{dataset}'                                                                                                 
            f'_{t_freq}'                                                                                                  
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                         
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                         
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                               
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                   
            )       
    path = f'{folder}/{filename}.nc'
    # -- find metric -- 
    doc_metric = xr.open_dataset(path)[metric_var].resample(time='1MS').mean()
    doc_metric = doc_metric.sel(time = da.time, method='nearest')                                                           # pick out the relevant section
    return doc_metric


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
    da.coords['time'] = wap.coords['time']
    w500 =      wap.sel(plev = 500e2)
    ascent =    xr.where(w500 < 0, 1, 0)  
    descent =   xr.where(w500 > 0, 1, 0)  
    da_low =    da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev').fillna(0)   # fill nan with zero just in case
    da_high =   da.sel(plev = slice(400e2, 0)).max(dim = 'plev').fillna(0)
    da_low_no_overlap = (da_low - da_high) * descent
    da_low_no_overlap = da_low_no_overlap.where(da_low_no_overlap > 0, 0)
    da_low =            da_low  * descent
    da_high =           da_high * ascent

    # ==
    #
    # variable area fraction method
    #
    # ==
    # == low clouds ==
    da_low_anom = cA.detrend_month_anom(da_low)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area',
        'area_fraction':    'area_fraction',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_low_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_low_anom)
        # -- put into dataset --
        ds[f'{metric_name}_low_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_low_vs_{doc_var}_sig'] =  significance_mask


    # == low clouds, no overlap ==
    da_low_no_overlap_anom = cA.detrend_month_anom(da_low_no_overlap)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area',
        'area_fraction':    'area_fraction',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_low_no_overlap_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_low_no_overlap_anom)
        # -- put into dataset --
        ds[f'{metric_name}_low_no_overlap_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_low_no_overlap_vs_{doc_var}_sig'] =  significance_mask


    # == high clouds ==
    da_high_anom = cA.detrend_month_anom(da_high)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area',
        'area_fraction':    'area_fraction',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_high_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_high_anom)
        # -- put into dataset --
        ds[f'{metric_name}_high_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_high_vs_{doc_var}_sig'] =  significance_mask


    # ==
    #
    # constant area fraction method
    #
    # ==
    # == low clouds ==
    da_low_anom = cA.detrend_month_anom(da_low)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area_fixed_area',
        'area_fraction':    'area_fraction_fixed_area',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_low_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_low_anom)
        # -- put into dataset --
        ds[f'{metric_name}_low_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_low_vs_{doc_var}_sig'] =  significance_mask


    # == low clouds, no overlap ==
    da_low_no_overlap_anom = cA.detrend_month_anom(da_low_no_overlap)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area_fixed_area',
        'area_fraction':    'area_fraction_fixed_area',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_low_no_overlap_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_low_no_overlap_anom)
        # -- put into dataset --
        ds[f'{metric_name}_low_no_overlap_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_low_no_overlap_vs_{doc_var}_sig'] =  significance_mask


    # == high clouds ==
    da_high_anom = cA.detrend_month_anom(da_high)
    # -- doc metric --  
    doc_names = {
        'mean_area':        'mean_area_fixed_area',
        'area_fraction':    'area_fraction_fixed_area',
        }
    for doc_name, doc_var in doc_names.items():
        doc_metric = get_doc_metric(dataset, years, da_high_anom, 
                                    metric_name = doc_name, 
                                    metric_var = doc_var
                                    )
        doc_metric_anom = cA.detrend_month_anom(doc_metric)
        # -- correlate tropical doc with conv --  
        corr_da, significance_mask = gC.calculate_correlation_and_significance(da_1d = doc_metric_anom, da_3d = da_high_anom)
        # -- put into dataset --
        ds[f'{metric_name}_high_vs_{doc_var}_corr'] = corr_da
        ds[f'{metric_name}_high_vs_{doc_var}_sig'] =  significance_mask


    # print(ds)
    # exit()

    return ds




