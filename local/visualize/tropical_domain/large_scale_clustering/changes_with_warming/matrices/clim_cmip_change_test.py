'''
# -----------------
#   plot_figure
# -----------------

'''

# == imports ==
# -- Packages --
import string
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

# -- Imported scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, plot_path):
    ''' import module from relative path '''
    if plot_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        relative_path = plot_path.replace(os.getcwd(), "").lstrip("/")
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                                                       'utils')
pF = import_relative_module('util_plot.get_subplot.corr_table_subplot',                         'utils')
cT = import_relative_module('util_calc.correlations.self_correlation_matrix',                   'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',               'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')

# == model subset ==
def in_subset(model):
    datasets = (                                                                                                                    # Models ordered by change in temperature with warming    
        # 'INM-CM5-0',                                                                                                                # 1   # no cloud
        'IITM-ESM',                                                                                                                 # 2   
        'FGOALS-g3',                                                                                                                # 3    
        # 'INM-CM4-8',                                                                                                                # 4                                
        'MIROC6',                                                                                                                   # 5                                      
        'MPI-ESM1-2-LR',                                                                                                            # 6                         
        # 'KIOST-ESM',                                                                                                              # 7
        'BCC-CSM2-MR',                                                                                                              # 8           
        # 'GFDL-ESM4',                                                                                                                # 9         
        'MIROC-ES2L',                                                                                                               # 10 
        'NorESM2-LM',                                                                                                               # 11      
        # 'NorESM2-MM',                                                                                                             # 12                      
        'MRI-ESM2-0',                                                                                                               # 13                            
        'GFDL-CM4',                                                                                                                 # 14      
        'CMCC-CM2-SR5',                                                                                                             # 15                
        'CMCC-ESM2',                                                                                                                # 16                                    
        'NESM3',                                                                                                                    # 17     
        'ACCESS-ESM1-5',                                                                                                            # 18 
        'CNRM-ESM2-1',                                                                                                              # 19 
        # 'EC-Earth3',                                                                                                                # 20 
        'CNRM-CM6-1',                                                                                                               # 21
        # 'CNRM-CM6-1-HR',                                                                                                            # 22  # no clouds
        'KACE-1-0-G',                                                                                                               # 23            
        'IPSL-CM6A-LR',                                                                                                             # 24
        'ACCESS-CM2',                                                                                                               # 25 
        'TaiESM1',                                                                                                                  # 26                      
        'CESM2-WACCM',                                                                                                              # 27   
        'CanESM5',                                                                                                                  # 28  
        'UKESM1-0-LL',                                                                                                              # 29
        )             
    in_subset = False
    if model in datasets:
        in_subset = True
    return in_subset


# == get metrrics ==
def get_metric_tas(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
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
    return doc_metric

# def get_dtas(models, p_id1, p_id2):
#     lon_area =  '0:360'                                                                                                                                                                             # area
#     lat_area =  '-30:30'                                                                                                                                                                            #
#     res =       2.8            
#     ds_dtas = xr.Dataset()
#     for model in models:
#         data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
#         if model == 'NOAA':
#             data_tyoe_group, data_tyoe = 'observations', 'NOAA'
#         if model == 'IFS_9_FESOM_5':
#             data_tyoe_group, data_tyoe = 'models', 'IFS'
#         x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_map',    'tas_map_mean',    r'T',   r'[$^o$C]'                               # x1
#         map_value =     get_metric_tas(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id1, x1_var).mean(dim = ('lat', 'lon'))
#         x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_map',    'tas_map_mean',    r'T',   r'[$^o$C]'                               # x1
#         map_value_warm =     get_metric_tas(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id2, x1_var).mean(dim = ('lat', 'lon'))
#         ds_dtas[model] = map_value_warm - map_value
#     return ds_dtas

def get_dtas(models, p_id1, p_id2):
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8            
    ds_dtas = xr.Dataset()
    for model in models:
        data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
        x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =  'monthly',  'tas',          'tas_timeseries',    'tas_timeseries_ecs',    r'T',   r'[$^o$C]'
        ds_dtas[model] = get_metric_tas(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id1, x1_var).mean(dim = 'time') # map_value_warm - map_value
    return ds_dtas


def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var, time_period2, d_tas = None):
    if metric_var in ['clouds_timeseries_low_mean', 'clouds_timeseries_high_mean'] and not in_subset(dataset):
        return xr.DataArray(np.nan * np.ones(10), dims=["time"], coords=[range(10)]).mean(dim = 'time'), xr.DataArray(np.nan * np.ones(10), dims=["time"], coords=[range(10)]).mean(dim = 'time')
    else:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
        # -- historical --
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

        if metric_var == 'area_fraction':
            metric_hist = xr.open_dataset(path)[metric_var].std(dim = 'time')
        else:
            metric_hist = xr.open_dataset(path)[metric_var].mean(dim = 'time')

        # -- warm --
        folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
        filename = (                                                                                                        # base result_filename
                f'{metric_name}'                                                                                            #
                f'_{dataset}'                                                                                               #
                f'_{t_freq}'                                                                                                #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
                f'_{time_period2.split(":")[0]}_{time_period2.split(":")[1]}'                                                 #
                )       
        path = f'{folder}/{filename}.nc'

        if metric_var == 'area_fraction':
            metric_warm = xr.open_dataset(path)[metric_var].std(dim = 'time')
        else:
            metric_warm = xr.open_dataset(path)[metric_var].mean(dim = 'time')

        metric = metric_warm - metric_hist

        # -- dtas --
        if d_tas is not None:
            metric = metric / d_tas[dataset]

        return metric, metric_hist

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]                 # convert the variables to a data array
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models


# == plot ==
def plot(plot_path):
    # -- settings common to all, that can be changed --
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8      

    # -- create figure --
    width, height = 6.27, 9.69                  # max size (for 1 inch margins)
    width, height = width * 0.85, height * 0.45    # modulate size and subplot distribution
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
    
    # -- figure description --
    description_x_pos = 0.175
    description_y_pos = 0.275
    fontsize = 8
    fig.text(description_x_pos, description_y_pos,  f'Change in Climatology \n (CMIP)',  transform=fig.transFigure,  fontweight='bold', fontsize = fontsize, ha='center' )

    # -- specify metrics --
    # -- DOC --    
    x1_tfreq,   x1_group,   x1_name,    x1_var,     x1_label,   x1_units =  'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'dA$_m$',           r'[km$^2$ K$^{-1}$]'   
    x2_tfreq,   x2_group,   x2_name,    x2_var,     x2_label,   x2_units =  'daily',    'doc_metrics',      'area_fraction',        'area_fraction',                            r'd$\sigma$(A$_f)$', r'[K$^{-1}$]'    
    x3_tfreq,   x3_group,   x3_name,    x3_var,     x3_label,   x3_units =  'daily',    'doc_metrics',      'number_index',         'number_index',                             r'dN',               r'[K$^{-1}$]'    

    # -- spatial preference --
    x4_tfreq,   x4_group,   x4_name,    x4_var,     x4_label,   x4_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'dC$_z$',           r'[km K$^{-1}$]'  
    # x5_tfreq,   x5_group,   x5_name,    x5_var,     x5_label,   x5_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'dC$_{heq}$',       r'[km K$^{-1}$]'   
    x5_tfreq,   x5_group,   x5_name,    x5_var,     x5_label,   x5_units =  'daily',    'doc_metrics',      'area_fraction',        'area_fraction',             r'$\sigma$(A$_f)$',       r'[%]'   

    # -- mechanisms --
    x6_tfreq,   x6_group,   x6_name,    x6_var,     x6_label,   x6_units =  'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_mean',                      r'dT',               r'[K]'
    x7_tfreq,   x7_group,   x7_name,    x7_var,     x7_label,   x7_units =  'monthly',  'tas',              'tas_gradients',        'tas_gradients_pacific_zonal_clim',         r'dT$_z$',           r'[K K$^{-1}$]'    

    # -- effects --
    x8_tfreq,   x8_group,   x8_name,    x8_var,     x8_label,   x8_units =  'monthly',  'olr',              'olr_timeseries',       'olr_timeseries_mean',                      r'dOLR',             r'[Wm$^{-2}$ K$^{-1}$]'  
    x9_tfreq,   x9_group,   x9_name,    x9_var,     x9_label,   x9_units =  'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',               r'dLCF$_d$',         r'[% K$^{-1}$]'  
    x10_tfreq,  x10_group,  x10_name,   x10_var,    x10_label,  x10_units = 'monthly',  'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                r'dRH',              r'[% K$^{-1}$]'  
    x11_tfreq,  x11_group,  x11_name,   x11_var,    x11_label,  x11_units = 'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_high_mean',              r'dHCF$_a$',         r'[% K$^{-1}$]'  


    # -- add metrics to dataset and create correlation matrix --
    list_clim = []
    for model in cL.get_model_letters():
        ds = xr.Dataset()
        p_id1, p_id2 =      '1970-01:1999-12', '2070-01:2099-12'      
        data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
        d_tas = get_dtas(cL.get_model_letters(), p_id1, p_id2)
        ds[f'{x1_label} {x1_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq,   x1_group,   x1_name,    lon_area, lat_area, res, p_id1, x1_var,     p_id2, d_tas)
        ds[f'{x2_label} {x2_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq,   x2_group,   x2_name,    lon_area, lat_area, res, p_id1, x2_var,     p_id2, d_tas)
        ds[f'{x3_label} {x3_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x3_tfreq,   x3_group,   x3_name,    lon_area, lat_area, res, p_id1, x3_var,     p_id2, d_tas)
        ds[f'{x4_label} {x4_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x4_tfreq,   x4_group,   x4_name,    lon_area, lat_area, res, p_id1, x4_var,     p_id2, d_tas)
        _, ds[f'{x5_label} {x5_units}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x5_tfreq,   x5_group,   x5_name,    lon_area, lat_area, res, p_id1, x5_var,     p_id2, d_tas)
        ds[f'{x6_label} {x6_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x6_tfreq,   x6_group,   x6_name,    lon_area, lat_area, res, p_id1, x6_var,     p_id2)
        ds[f'{x7_label} {x7_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x7_tfreq,   x7_group,   x7_name,    lon_area, lat_area, res, p_id1, x7_var,     p_id2, d_tas)
        ds[f'{x8_label} {x8_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x8_tfreq,   x8_group,   x8_name,    lon_area, lat_area, res, p_id1, x8_var,     p_id2, d_tas)
        ds[f'{x9_label} {x9_units}'], _ =      get_metric(data_tyoe_group, data_tyoe, dataset, x9_tfreq,   x9_group,   x9_name,    lon_area, lat_area, res, p_id1, x9_var,     p_id2, d_tas)
        ds[f'{x10_label} {x10_units}'], _ =    get_metric(data_tyoe_group, data_tyoe, dataset, x10_tfreq,  x10_group,  x10_name,   lon_area, lat_area, res, p_id1, x10_var,    p_id2, d_tas)
        ds[f'{x11_label} {x11_units}'], _ =    get_metric(data_tyoe_group, data_tyoe, dataset, x11_tfreq,  x11_group,  x11_name,   lon_area, lat_area, res, p_id1, x11_var,    p_id2, d_tas)
        # ds[f'{x12_label} {x12_units}'] =    get_metric(data_tyoe_group, data_tyoe, dataset, x12_tfreq,  x12_group,  x12_name,   lon_area, lat_area, res, p_id1, x12_var,    p_id2, d_tas)
        ds = ds.assign_coords(model = model)
        if 'height' in ds:
            ds = ds.drop_vars('height')
        list_clim.append(ds)
        # print(ds)
        # exit()
    ds_clim = xr.concat(list_clim, dim = 'model')
    da_corr_matrix, da_p_value_matrix = cT.get_correlation_matrix(ds_clim) 
    
    # -- plot metrics --
    ds = xr.Dataset()
    ds.attrs.update({
        'scale': 1.05, 'move_row': -0.1, 'move_col': -0.04,
        'cmap': 'RdBu',
        'hide_labels': False,
        'text_size': 7,
        'valuetext_size': 7,
        'hide_cbar': False, 'cbar_pad': 0.025, 'cbar_ypad': 0.1, 'cbar_width': 0.02, 'cbar_height': 0.8, 
        'cbar_numsize': 8, 'cbar_label_pad': 0.0925, 'cbar_label': r'r(metric$_{row}$, metric$_{col}$)', 'cbar_fontsize': 8,
        'axtitle_fontsize': 7, 'axtitle_label': '', 'axtitle_ypad': 0, 'axtitle_xpad': 0,
        })
    pF.plot(fig, ax, da_corr_matrix, da_p_value_matrix, ds)

    # -- save figure --
    path = plot_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran ==
if __name__ == '__main__':
    # ds = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/models/cmip/tas/tas_gradients/ACCESS-CM2/tas_gradients_ACCESS-CM2_monthly_0-360_-30-30_128x64_1970-01_1999-12.nc')
    # print(ds)
    # exit()
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    plot_path = f'{folder_work}/plots/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}_{Path(__file__).stem}.pdf' 
    plot(plot_path)




