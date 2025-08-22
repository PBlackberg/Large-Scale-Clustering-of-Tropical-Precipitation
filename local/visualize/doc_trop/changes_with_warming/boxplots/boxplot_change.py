'''
# ----------
#  Boxplot
# ----------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
mS =    import_relative_module('user_specs',                                        'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                    'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',   'utils')


# == get metric ==
def get_metric(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                       # user settings
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'                          # 
    filename = (                                                                                                                    # base result_filename
            f'{metric_name}'                                                                                                        #
            f'_{dataset}'                                                                                                           #
            f'_{t_freq}'                                                                                                            #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                                   #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                                   #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                         #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                             #
            )       
    path = f'{folder}/{filename}.nc'
    metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        metric = metric[metric_var]
        if metric_var == 'tas_gradients_oni':                                                                                       # rolling-mean metric, doesn't have values for start and end
            metric = metric.ffill(dim='time')                                                                                       # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                                       # Backward fill  (fill first value for 3 month rolling mean)
    return metric

def ds_to_variable(ds):
    ''' Used for example for model-mean calculation '''
    data_arrays = [ds[var] for var in ds.data_vars]
    da = xr.concat(data_arrays, dim = 'variable')
    return da


# == plot ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.35,                                                                                              # x-start
            ax_position.y0 - 0.065,                                                                                             # y-start
            text,                        
            fontsize = 8,  
            # fontweight = 'bold',  
            transform=fig.transFigure,
            )

def move_col(ax, moveby):
    ax_position = ax.get_position()             
    _, bottom, width, height = ax_position.bounds                                                                               # [left, bottom, width, height]
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds                                                                                 # [left, bottom, width, height]
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def format_yticks(ax, ymin, ymax):
    ax.set_ylim([ymin, ymax])
    formatter_y = ticker.ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.yaxis.get_offset_text().set_size(8)
    ax.get_yaxis().get_offset_text().set_ha('right')

def plot_ylabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.175, 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            text, 
            va = 'center', 
            rotation = 'vertical', 
            # fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot(y, y_cmip_example):
    # -- set general fontsize --    
    plt.rcParams['font.size'] = 7
    # -- create figure --    
    width, height = 2.5, 6                                                                                                                                     
    width, height = [f / 2.54 for f in [width, height]]                                                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, scaleby = 0.5)
    scale_ax_y(ax, scaleby = 1)
    move_row(ax, 0)     
    move_col(ax, 0.375)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.set_xlim([0,1])
    ax.set_ylim([-1.25e2, 1.25e2])
    # ax.set_ylim([-1.25e2, 0.1e2])
    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    
    # -- boxplot lines --
    color = 'k'
    linewidth = 1
    alpha = 1 # 0.8
    box_mid = 0.5 #- 0.1
    box_left = 0.25 #- 0.1
    box_right = 0.75 #- 0.1
    box_top_width = 0.1
    # box
    prc_75 = np.percentile(y, 75)
    ax.plot([box_left, box_right],  [prc_75, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_50 = np.percentile(y, 50)
    ax.plot([box_left, box_right],  [prc_50, prc_50],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    prc_25 = np.percentile(y, 25)
    ax.plot([box_left, box_right],  [prc_25, prc_25],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # hortizontal line
    ax.plot([box_left, box_left],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # verical line
    ax.plot([box_right, box_right],  [prc_25, prc_75],   color = color, linestyle = '-', linewidth = linewidth, alpha = alpha)                                   # vertical line
    # max
    y_max = np.max(y)
    ax.plot([box_mid, box_mid],     [prc_75, y_max],    color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                           # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_max, y_max],     color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                           # hortizontal line
    # min
    y_min = np.min(y)
    ax.plot([box_mid, box_mid],     [y_min, prc_25], color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                              # verical line
    ax.plot([box_mid - box_top_width, box_mid + box_top_width],     [y_min, y_min], color = color, linestyle = '-', linewidth = 1, alpha = alpha)                                               # hortizontal line

    # -- icons --
    # if y_obs_p is not None:
    #     if y_obs_p < 0.05:
    #         color = 'g'
    #         x_position = box_mid - 0.2
    #         y_position = y_obs
    #         text = ax.text(x_position, y_position,                                                                                                              # data
    #                         '\u2605',                                                                                                                           # star                                   
    #                         color = 'w', ha='center', va='center',                                                                                              #
    #                         fontsize = 10, fontweight='bold')   
    #         text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = color)])                                                               # background effects

    # if y_highres_p is not None:
    #     if y_highres_p < 0.05:
    #         color = 'purple'
    #         x_position = box_mid + 0.2
    #         y_position = y_highres
    #         text = ax.text(x_position, y_position,                                                                                                              # data
    #                         '\u25C6',                                                                                                                                # star                                   
    #                         color = 'w', ha='center', va='center',                                                                                              #
    #                         fontsize = 9, fontweight='bold')   
    #         text.set_path_effects([PathEffects.withStroke(linewidth = 1, foreground = 'purple')])                                                            # background effects

    # print(y_cmip_example)
    # exit()
    # color = 'k'
    # x_position = box_mid
    # y_position = y_cmip_example
    # text = ax.text(x_position, y_position,                                                                                                              # data
    #                 'P',                                                                                                                                # text                                   
    #                 color = 'w', ha='center', va='center',                                                                                              #
    #                 fontsize = 6, fontweight='bold')                                                                                                  # 
    # text.set_path_effects([PathEffects.withStroke(linewidth=1.25, foreground='k')])                                                                     # background effects

    # -- limits --
    ymax = None
    ymin = None 
    # [Cm]
    # ymax = 0.1
    # ymin = -5.5e1

    # [Cheq]
    ymax = 0.5e1
    ymin = -8.5e1

    format_yticks(ax, ymin = ymin, ymax = ymax)

    return fig, ax


# == main ==
def main():
    # == Specify metrics to use ==
    # -- x-metric --
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'area_fraction',        'area_fraction',                            r'A$_f$',       r''   
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'    
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'   
    x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  

    # -- normalization --
    n_tfreq,   n_group,   n_name,   n_var,  n_label,    n_units =   'monthly',  'tas',              'tas_map',              'tas_map_mean',                             r'T',           r'$^o$C'   

    # == Get metrics ==
    # -- CMIP: metrics --
    x_cmip, model_list = [], []
    for model in cL.get_model_letters():
        data_type_group, data_type, dataset = 'models', 'cmip', model
        lon_area =       '0:360'                                                                                                                                                                             
        lat_area =      '-30:30'                                                                                                                                                                            
        res =           2.8    
        # -- historical --
        time_period =   '1970-01:1999-12'
        x = get_metric(data_type_group, data_type, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
        n = get_metric(data_type_group, data_type, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)
        
        # -- warm --
        time_period =      '2070-01:2099-12'   
        x_warm = get_metric(data_type_group, data_type, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
        n_warm = get_metric(data_type_group, data_type, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)
        
        # -- pre-process metric --
        dn = n_warm.mean(dim = ('lat', 'lon'))  - n.mean(dim = ('lat', 'lon'))
        dx = x_warm.mean(dim = 'time')          - x.mean(dim = 'time')

        dx = dx / dn
        x_cmip.append(dx.data)
        model_list.append(model)    

    y_cmip_example = x_cmip[model_list.index('ACCESS-ESM1-5')]

    # == plot ==
    # -- plot data --
    fig, ax = plot(x_cmip, y_cmip_example)

    # -- plot labels --
    text = rf'd{x_label} [{x_units} K$^{{-1}}$]'
    plot_ax_title(fig, ax, text)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'x_{x_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()



