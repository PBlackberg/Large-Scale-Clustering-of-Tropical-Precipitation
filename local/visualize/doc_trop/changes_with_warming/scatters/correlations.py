'''
# -----------------------
#  scatter: correlation
# -----------------------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from scipy.stats import pearsonr

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

def in_subset(model):
    ''' use a subset where there are some models missing variables '''
    datasets = (                                                                                                                    # Models ordered by change in temperature with warming    
        # 'INM-CM5-0',                                                                                                              # 1   # no cloud
        'IITM-ESM',                                                                                                                 # 2   
        'FGOALS-g3',                                                                                                                # 3    
        # 'INM-CM4-8',                                                                                                              # 4                                
        'MIROC6',                                                                                                                   # 5                                      
        'MPI-ESM1-2-LR',                                                                                                            # 6                         
        # 'KIOST-ESM',                                                                                                              # 7
        'BCC-CSM2-MR',                                                                                                              # 8           
        # 'GFDL-ESM4',                                                                                                              # 9         
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
        # 'EC-Earth3',                                                                                                              # 20 
        'CNRM-CM6-1',                                                                                                               # 21
        # 'CNRM-CM6-1-HR',                                                                                                          # 22 
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
    left, bottom, _1, _2 = ax_position.bounds                                                                                       # [left, bottom, width, height]
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

def move_col(ax, moveby):
    ax_position = ax.get_position()             
    _, bottom, width, height = ax_position.bounds                                                                                   # [left, bottom, width, height]
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds                                                                                     # [left, bottom, width, height]
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def plot_ticks(ax, xticks, yticks):
    ax.xaxis.set_tick_params(labelsize = 8)
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    ax.yaxis.set_tick_params(labelsize = 8) 
    ax.yaxis.set_tick_params(length = 2)
    ax.yaxis.set_tick_params(width = 1)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(right = False)

def plot_xlabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2 - 0.05, 
            ax_position.y0 - 0.15, 
            text, 
            ha = 'center', 
            fontsize = 8, 
            transform = fig.transFigure
            )
    
def plot_ylabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.175, 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            text, 
            va = 'center', 
            rotation = 'vertical', 
            fontsize = 8, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.15,                                                                                              # x-start
            ax_position.y1 + 0.125,                                                                                             # y-start
            text,                        
            fontsize = 8,  
            transform=fig.transFigure,
            )
    
def plot_cbar_label2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x1 + 0.15,                                                                                              # x-start
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2,                                                             # y-start
            text,                                                                              
            rotation = 'vertical', 
            va = 'center', 
            fontsize = 8, 
            transform=fig.transFigure
            )

def cbar_ax_right(fig, ax, h):
    ax_position = ax.get_position()
    cbar_ax = fig.add_axes([ax_position.x1 + 0.0125,                                                                            # left
                            ax_position.y0 + (ax_position.height - ax_position.height * 0.9) / 2,                               # bottom
                            ax_position.width * 0.025,                                                                          # width
                            ax_position.height * 0.9                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation='vertical')
    ticks = cbar.ax.get_yticks()    
    try:
        ticklabels = [f'{int(t)}' for t in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    except:
        pass
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.tick_params(labelsize = 8)
    cbar.ax.yaxis.get_offset_text().set_size(8)
    cbar.ax.yaxis.set_offset_position('left')
    return cbar_ax

def format_xticks(ax, xmin, xmax):
    ax.set_xlim([xmin, xmax])
    formatter_x = ticker.ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter_x)
    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.xaxis.get_offset_text().set_size(8)
    ax.get_xaxis().get_offset_text().set_ha('right')

def format_yticks(ax, ymin, ymax):
    ax.set_ylim([ymin, ymax])
    formatter_y = ticker.ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.tick_params(axis='both', which='major', labelsize = 8)
    ax.yaxis.get_offset_text().set_size(8)
    ax.get_yaxis().get_offset_text().set_ha('right')

def plot(x, y, model_list):
    # -- create figure --    
    width, height = 5, 5                                                                                                        # figure size
    width, height = [f / 2.54 for f in [width, height]]                                                                         # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, 1)
    scale_ax_y(ax, 0.8)
    move_row(ax, 0.1)     
    move_col(ax, 0.075)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # -- limits --
    xmax = None
    xmin = None
    ymax = None
    ymin = None

    # [[x-limits]]
    # [Am]
    # xmax = 3.5e4
    # xmin = 0.25e4    

    # [Tz]
    # xmax = 1e-1
    # xmin = -3e-1

    # [[y-limits]]
    # [Am]
    # ymax = 3.5e4
    # ymin = 0.25e4    

    # [Cz]
    # ymax = 1.5e2
    # ymin = -ymax

    # [Cm]
    # ymax = 0.5e1
    # ymin = -5.5e1

    # [RH]
    # ymax = 1.5
    # ymin = -0.75

    # [LCF]
    # ymax = 5e-1
    # ymin = -6.25e-1

    # [ECS]
    # ymax = 6
    # ymin = 1.5

    ax.scatter(x, y, alpha = 0)                                                                         # needed to place letters
    format_xticks(ax, xmin, xmax)
    format_yticks(ax, ymin, ymax)

    # -- plot letters ontop --
    letters = cL.get_model_letters()                                                                    # letter connection to model
    for i, model in enumerate(model_list):                                                              # plot models first, with the colormap
        # if model == 'ACCESS-ESM1-5':
        #     color = 'w'
        #     text = ax.text(x[i], y[i],                                                                  # data
        #                     letters[model],                                                             # letter                                   
        #                     color = color, ha='center', va='center',                                    #
        #                     fontsize = 8, fontweight = 'bold')                                          #
        #     text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='black')])         # background effects
        # else:
        color = (0.2, 0.2, 0.2)                                                                     # Get color from colormap
        text = ax.text(x[i], y[i],                                                                  # data
                        letters[model],                                                             # letter                                   
                        color = color, ha='center', va='center',                                    #
                        fontsize = 8, fontweight = 'bold')                                          #
        text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])         # background effects

    # -- highlight values changing sign --
    if (x < 0).any() and (x > 0).any():
        ax.axvline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    if (y < 0).any() and (y > 0).any():
        ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)

    # -- create slope --
    r, p = pearsonr(x, y)
    if p < 0.05:
        xpad = 0.5
        ypad = 0.01
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = intercept + slope * x
        ax.plot(x, y_fit, color = 'k', linewidth = 1)
        ax_position = ax.get_position()
        ax.text(ax_position.x0 + xpad,                                                                  # x-start
                ax_position.y1 + ypad,                                                                  # y-start
                rf'R$^2$ = {r**2:0.2}',                        
                fontsize = 8,  
                transform=fig.transFigure,
                color = 'r'
                )
    return fig, ax


# == main ==
def main():
    # -- x-metric --
    # x_tfreq,  x_group,    x_name, x_var,  x_label,        x_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   
    # x_tfreq,  x_group,    x_name, x_var,  x_label,        x_units =   'monthly',  'tas',              'tas_gradients',        'tas_gradients_pacific_zonal_clim',         r'T$_z$',       r'K'  
    # x_tfreq,  x_group,    x_name, x_var,  x_label,        x_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'    
    x_tfreq,  x_group,    x_name, x_var,  x_label,        x_units =   'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'   

    # -- y-metric --
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'   
    y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'   
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',  'clouds',           'clouds_timeseries',    'clouds_timeseries_low_mean',               r'LCF$_d$',     r'%'  
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',  'rel_humid_mid',    'rel_humid_timeseries', 'rel_humid_timeseries_mean',                r'RH',          r'%'  
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_ecs',                       r'ECS',         r'K'

    # -- normalization --
    n_tfreq,    n_group,  n_name,     n_var,  n_label,    n_units =   'monthly',  'tas',              'tas_map',              'tas_map_mean',                             r'T',           r'$^o$C'    

    # -- CMIP metrics --
    x_list, y_list, z_list, model_list = [], [], [], []
    for model in cL.get_model_letters():
        if y_group == 'clouds' and not in_subset(model):
            continue
        model_list.append(model)
        data_type_group, data_tyoe, dataset = 'models', 'cmip', model
        lon_area =  '0:360'                                                                                                                                                                             
        lat_area =  '-30:30'                                                                                                                                                                            
        res =       2.8    
        # -- historical --
        time_period =      '1970-01:1999-12'
        x = get_metric(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
        y = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
        n = get_metric(data_type_group, data_tyoe, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)

        # -- warm --
        time_period =      '2070-01:2099-12'    
        x_warm = get_metric(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
        y_warm = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
        n_warm = get_metric(data_type_group, data_tyoe, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)

        # -- pre-process metric --
        dn = n_warm.mean(dim = ('lat', 'lon'))  - n.mean(dim = ('lat', 'lon'))

        dx = x_warm.mean(dim = 'time')          - x.mean(dim = 'time')
        dx = dx / dn

        if y_var == 'tas_timeseries_ecs':
            dy = y.mean(dim = 'time')
        else:
            dy = y_warm.mean(dim = 'time')          - y.mean(dim = 'time')
            dy = dy / dn
        x_list.append(dx.data)
        y_list.append(dy.data)

    # x_array, y_array = [np.array(f) - np.mean(np.array(f)) for f in [x_list, y_list]]         # anomaly from ensemble-mean
    x_array, y_array = np.array(x_list), np.array(y_list)                                       # absolute values

    # -- plot data --
    fig, ax = plot(x_array, y_array, model_list)

    # -- plot labels --
    tas_norm_text = r'K$^{-1}$'
    text = f'd{x_label} [{x_units} {tas_norm_text}]'
    plot_xlabel(fig, ax, text)
    if y_var == 'tas_timeseries_ecs':
        text = f'{y_label} [{y_units}]'
    else:
        text = f'd{y_label} [{y_units} {tas_norm_text}]'
    plot_ylabel(fig, ax, text)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'x_{x_var}_y_{y_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()






