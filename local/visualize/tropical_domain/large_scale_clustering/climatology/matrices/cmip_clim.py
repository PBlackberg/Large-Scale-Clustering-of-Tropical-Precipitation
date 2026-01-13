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
pF_M = import_relative_module('util_plot.get_subplot.map_subplot',     'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                    'utils')


# == get metrrics ==
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
    metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        # -- get metric variation -- 
        metric = metric[metric_var]
    # # -- anomalies --
    # try:
    #     metric = cA.detrend_month_anom(metric)                                                                          # correlate anomalies
    # except:                                                                                                             #
    #     metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
    #     metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
    #     metric = cA.detrend_month_anom(metric)                                                                          #
    # metric = metric.assign_coords(time=np.arange(360))                                                                  # make sure they have common time axis (to put in the same dataset)
    # -- mean --
    # metric = metric.mean(dim = 'time')
    return metric

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]                 # convert the variables to a data array
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models

def get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix):
    # -- mean and spread --
    nb_models   = len(ds_corr_matrix.data_vars)
    da_corr     = ds_to_da(ds_corr_matrix)
    corr_mean   = da_corr.mean(dim = 'variable')
    corr_spread = da_corr.std(dim='variable')
    # -- nb of significant --
    da_p        = ds_to_da(ds_p_value_matrix)
    p_threshold = 0.05
    da_p_sig = xr.where(da_p < p_threshold, int(1), 0)
    p_count = da_p_sig.sum(dim = 'variable').data
    # [print(f) for f in [corr_mean, corr_spread, da_p, p_count]]
    # exit()
    return corr_mean, corr_spread, p_count, nb_models


# == plot ==
def plot(plot_path):
    # == specify metrics ==
    # -- settings common to all, that can be changed --
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8      

    # -- DOC --
    x1_tfreq,   x1_group,   x1_name,    x1_var,     x1_label,   x1_units =  'daily',    'doc_metrics',  'mean_area',    'mean_area',        r'A_m',   r'[km$^2$]'        

    # -- MAP -- 
    x2_tfreq,   x2_group,   x2_name,    x2_var,     x2_label,   x2_units =  'daily',    'conv',         'conv_map',     'conv_map_mean' ,   r'C',       r'[%]'      

    # -- tas --
    t_tfreq,   t_group,   t_name,    t_var, t_label,   t_units =            'monthly',  'tas',          'tas_timeseries',    'tas_timeseries_ecs',    r'ECS',   r'[$^o$C]'

    list_x_clim, list_y_clim = [], []
    list_x_warm, list_y_warm = [], []
    list_ecs = []
    for model in cL.get_model_letters():
        # -- historical --
        ds_x, ds_y, ds_ecs = xr.Dataset(),  xr.Dataset(), xr.Dataset()
        p_id =      '1970-01:1999-12'    
        data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
        ds_x[f'{x1_label}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq,   x1_group,   x1_name,    lon_area, lat_area, res, p_id, x1_var).mean(dim = 'time')
        ds_x = ds_x.assign_coords(model = model)
        ds_y[f'{x2_label}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq,   x2_group,   x2_name,    lon_area, lat_area, res, p_id, x2_var)
        ds_y = ds_y.assign_coords(model = model)
        ds_ecs[f'{t_label}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, t_tfreq,   t_group,   t_name,    lon_area, lat_area, res, p_id, t_var).mean(dim = 'time')
        ds_ecs = ds_ecs.assign_coords(model = model)
        if 'height' in ds_ecs:
            ds_ecs = ds_ecs.drop_vars('height')
        list_x_clim.append(ds_x)
        list_y_clim.append(ds_y)
        list_ecs.append(ds_ecs)
        # -- warm --
        ds_x, ds_y = xr.Dataset(),  xr.Dataset()
        p_id =      '2070-01:2099-12'    
        data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', model
        ds_x[f'{x1_units}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq,   x1_group,   x1_name,    lon_area, lat_area, res, p_id, x1_var).mean(dim = 'time')
        ds_x = ds_x.assign_coords(model = model)
        ds_y[f'{x2_label}'] =      get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq,   x2_group,   x2_name,    lon_area, lat_area, res, p_id, x2_var)
        ds_y = ds_y.assign_coords(model = model)            
        list_x_warm.append(ds_x)
        list_y_warm.append(ds_y)
    ds_x_clim = xr.concat(list_x_clim, dim = 'model')
    ds_x_warm = xr.concat(list_x_warm, dim = 'model')
    ds_y_clim = xr.concat(list_y_clim, dim = 'model')
    ds_y_warm = xr.concat(list_y_clim, dim = 'model')
    ds_ecs = xr.concat(list_ecs, dim = 'model')
    ds_x_clim_sorted = ds_x_clim.sortby('A_m', ascending=True)
    models_ds_x_clim_sorted = ds_x_clim_sorted['model'].values


    # == plot ==
    # -- create figure --
    # labels = list(string.ascii_lowercase) 
    # labels = labels + ['A', 'B']
    width, height = 6.27, 9.69                  # max size (for 1 inch margins)
    width, height = width * 1, height * 0.325    # modulate size and subplot distribution
    ncols, nrows  = 4, 7
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
    axes_list = axes.flatten()
    for i, (model, ax) in enumerate(zip(models_ds_x_clim_sorted, axes_list)):
        row, col = divmod(i, ncols)
        # print(row)
        # print(col)
        # print(ds_y_clim[x2_label].sel(model = model))
        # exit()
        
        # -- format for all subplots --
        xticks = [60, 120, 180, 240, 300]
        yticks = [-20, 0, 20]
        cmap = 'Blues'
        hide_xticks =   True
        hide_xlabel =   True
        hide_yticks =   True
        hide_ylabel =   True
        hide_cbar =     True
        ds_map, ds_contour = xr.Dataset(), xr.Dataset()
        ds_map['var'], ds_contour['var'] = ds_y_clim[x2_label].sel(model = model), ds_y_clim[x2_label].sel(model = model)
        var_name = list(ds_map.data_vars.keys())[0]

        # -- format for specific subplots --
        if row == 0:
            move_row = 0.075
        if row == 1:
            move_row = 0.075 - 0.005
        if row == 2:
            move_row = 0.075 - 0.01
        if row == 3:
            move_row = 0.075 - 0.015
        if row == 4:
            move_row = 0.075 - 0.02
        if row == 5:
            move_row = 0.075 - 0.0225
        if row == 6:
            move_row = 0.075 - 0.0275

    
        if col == 0:
            hide_yticks =   False
            hide_ylabel =   False
            move_col = -0.065    
        if col == 1:
            move_col = -0.0325 #+ 0.0025/2
        if col == 2:
            move_col = 0    
        if col == 3:
            move_col = 0.035 - 0.0025
        if row == nrows - 1:
            hide_xticks =   False
            hide_xlabel =   False
            hide_cbar =     False

        # -- plot subplot --
        some_text = r'dA$_m$'
        # print(ds_x_clim[x1_label].sel(model = model).data)
        # print(type(ds_x_clim[x1_label].sel(model = model).data))
        # exit()

        label = f'{cL.get_model_letters()[model]}: {model}, {some_text}: {ds_x_clim[x1_label].sel(model = model).data:.2e}'
        ds_map.attrs.update({ 'scale': 1.35, 'move_row': move_row, 'move_col': move_col,                                                                    # format axes
        'name': f'{var_name}',                                                                                                                              # plot
        'vmin': 0, 'vmax': 20, 'cmap': cmap, 'cbar_height': 0.01, 'cbar_pad': 0.079,                                                                  # colorbar: position
        'cbar_label': f'{x2_label} {x2_units}', 'cbar_fontsize': 5, 'cbar_numsize': 5, 'cbar_label_pad': 0.0775,                                            # colorbar: label                
        'hide_xticks': hide_xticks, 'xticks': xticks, 'xticks_fontsize': 5,                                                                                 # x-axis:   ticks
        'hide_xlabel': hide_xlabel, 'xlabel_label': 'longitude', 'xlabel_pad': 0.065, 'xlabel_fontsize': 5,                                                 # x-axis:   label
        'hide_yticks': hide_yticks, 'yticks': yticks, 'yticks_fontsize':  5,                                                                                # y-axis:   ticks
        'hide_ylabel': hide_ylabel, 'ylabel_label': 'latitude', 'ylabel_pad': 0.055, 'ylabel_fontsize': 5,                                                  # y-axis:   label
        'axtitle_label': f'{label}', 'axtitle_xpad': 0.015, 'axtitle_ypad': 0.005, 'axtitle_fontsize': 4.5,
        'hide_colorbar': hide_cbar,         
        })                                                     # subplot   title
        ds_contour.attrs.update({'threshold': ds_contour[ds_map.attrs['name']].quantile(0.90, dim=('lat', 'lon')).data, 'color': 'k', 'linewidth': 0.5})    # contour
        axes[row, col] = pF_M.plot(fig, nrows, ncols, row, col, ax = axes[row, col], ds = ds_map, ds_contour = ds_contour, lines = [])
        # if i > 3:
        #     break

        # # -- model-mean --
        # move_col = 0.035 - 0.0025
        # i = len(axes_list) - 1
        # row, col = divmod(i, ncols)
        # ds_map, ds_contour = xr.Dataset(), xr.Dataset()
        # ds_map['var'], ds_contour['var'] = ds_y_clim[x2_label].mean(dim = 'model'), ds_y_clim[x2_label].mean(dim = 'model')
        # var_name = list(ds_map.data_vars.keys())[0]
        # # -- plot subplot --
        # ds_map.attrs.update({ 'scale': 1.35, 'move_row': move_row, 'move_col': move_col,                                                                    # format axes
        # 'name': f'{var_name}',                                                                                                                              # plot
        # 'vmin': 0, 'vmax': 20, 'cmap': cmap, 'cbar_height': 0.01, 'cbar_pad': 0.079,                                                                  # colorbar: position
        # 'cbar_label': f'{x2_label} {x2_units}', 'cbar_fontsize': 5, 'cbar_numsize': 5, 'cbar_label_pad': 0.0775,                                            # colorbar: label                
        # 'hide_xticks': hide_xticks, 'xticks': xticks, 'xticks_fontsize': 5,                                                                                 # x-axis:   ticks
        # 'hide_xlabel': hide_xlabel, 'xlabel_label': 'longitude', 'xlabel_pad': 0.065, 'xlabel_fontsize': 5,                                                 # x-axis:   label
        # 'hide_yticks': hide_yticks, 'yticks': yticks, 'yticks_fontsize':  5,                                                                                # y-axis:   ticks
        # 'hide_ylabel': hide_ylabel, 'ylabel_label': 'latitude', 'ylabel_pad': 0.055, 'ylabel_fontsize': 5,                                                  # y-axis:   label
        # 'axtitle_label': f'model-mean', 'axtitle_xpad': 0.015, 'axtitle_ypad': 0.01, 'axtitle_fontsize': 4.5,
        # 'hide_colorbar': hide_cbar,         
        # })                                                     # subplot   title

        # ds_contour.attrs.update({'threshold': ds_contour[ds_map.attrs['name']].quantile(0.90, dim=('lat', 'lon')).data, 'color': 'k', 'linewidth': 0.5})    # contour
        # axes[row, col] = pF_M.plot(fig, nrows, ncols, row, col, ax = axes[row, col], ds = ds_map, ds_contour = ds_contour, lines = [])


    # -- remove extra subplots --
    # for row, col in np.ndindex(nrows, ncols):
        # ''
        # if row > 1 and col > 0:
        #     axes[row, col].remove()
        # if (row == 2 and col == 2):
        #     axes[row, col].remove()
    axes[6, 3].remove()

    # -- figure description --
    description_x_pos = 0.875
    description_y_pos = 0.125
    fontsize = 8
    fig.text(description_x_pos, description_y_pos,  f'Climatology \n (CMIP)',  transform=fig.transFigure,  fontweight='bold', fontsize = fontsize, ha='center' )


    # -- save figure --
    path = plot_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)




if __name__ == '__main__':
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    plot_path = f'{folder_work}/plots/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}_{Path(__file__).stem}.pdf' 
    plot(plot_path)




