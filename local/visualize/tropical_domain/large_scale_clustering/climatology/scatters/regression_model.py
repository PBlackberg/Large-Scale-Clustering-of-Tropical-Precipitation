'''
# -------------------
#  Regression maps
# -------------------

'''

# == imports ==
# -- Packages --
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
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
gC = import_relative_module('util_calc.correlations.gridbox_regression',            'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',   'utils')
mlR = import_relative_module('util_calc.multiple_linear_regression.mlr_calc',       'utils')


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
        if metric_var == 'tas_gradients_oni':
            metric = metric.ffill(dim='time')                                                                                       # Forward fill   (fill last value for 3 month rolling mean)
            metric = metric.bfill(dim='time')                                                                                       # Backward fill  (fill first value for 3 month rolling mean)
    return metric

def ds_to_variable(ds):
    ''' Used for example for model-mean calculation '''
    data_arrays = [ds[var] for var in ds.data_vars]
    da = xr.concat(data_arrays, dim = 'variable')
    return da

def get_plot_metric(xy_list):
    x_list = xy_list[:-1]
    y = xy_list[-1]
    # print(f'Linear mdoel coefficients using {len(x_list)} predictor variables')
    vif = mlR.check_variance_inflation_factor(x_list, show = False)
    # print(f'VIF: {vif}')
    show_it = False
    # print(vif)
    # print(coeffs)
    # mlR.get_sequential_variance_decomposition(x_list, y, show = True)
    # r_partial_x2, p_partial_x2 = mlR.get_pearson_partial_correlation(x_list, y, show = show_it)
    # mlR.get_linear_model_components_2(x_list, y, show = True, standardized = True)
    y_hat, coeffs, residual = mlR.get_linear_model_components(x_list, y, show = False, standardized = True)
    # print(coeffs)
    # print(r_partial_x2)
    # exit()    
    return y_hat, coeffs


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
    # x-ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = 7)
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    # ax.xaxis.set_tick_params(labelsize=0)
    # ax.set_xticklabels('')
    # y-ticks
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = 7) 
    ax.yaxis.set_tick_params(length = 2)
    ax.yaxis.set_tick_params(width = 1)
    # ax.yaxis.set_tick_params(labelsize=0)
    # ax.set_yticklabels('')
    # both
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(right = False)


def plot_xlabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - 0.175, 
            text, 
            ha = 'center', 
            fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ylabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.175, 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            text, 
            va = 'center', 
            rotation = 'vertical', 
            fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.15,                                                                                              # x-start
            ax_position.y1 + 0.125,                                                                                              # y-start
            text,                        
            fontsize = 4,  
            transform=fig.transFigure,
            )
    
def plot_cbar_label2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x1 + 0.175,                                                                                                     # x-start
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2,                                                             # y-start
            text,                                                                              
            rotation = 'vertical', 
            va = 'center', 
            fontsize = 7, 
            transform=fig.transFigure
            )

def cbar_ax_right(fig, ax, h):
    ax_position = ax.get_position()
    cbar_ax = fig.add_axes([ax_position.x1 + 0.0125,                                                                              # left
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
    cbar.ax.tick_params(labelsize = 7)
    cbar.ax.yaxis.get_offset_text().set_size(7)
    cbar.ax.yaxis.set_offset_position('left')
    return cbar_ax

def format_xticks(ax, xmin, xmax):
    ax.set_xlim([xmin, xmax])
    formatter_x = ticker.ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter_x)
    ax.tick_params(axis='both', which='major', labelsize = 7)
    ax.xaxis.get_offset_text().set_size(7)
    ax.get_xaxis().get_offset_text().set_ha('right')

def format_yticks(ax, ymin, ymax):
    ax.set_ylim([ymin, ymax])
    formatter_y = ticker.ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.tick_params(axis='both', which='major', labelsize = 7)
    ax.yaxis.get_offset_text().set_size(7)
    ax.get_yaxis().get_offset_text().set_ha('right')

def plot_ax_title2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + 0.06,                                                                                              # x-start
            ax_position.y1 + 0.025,                                                                                              # y-start
            text,                        
            fontsize = 7,  
            transform=fig.transFigure,
            )

def plot(x, y, model_list):
    # print(len(z))
    # exit()

    # -- create figure --    
    # width, height = 6.27, 9.69                                                                                                  # max size (for 1 inch margins)
    width, height = 5, 5                                                                                                        # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, 0.75)
    scale_ax_y(ax, 0.6)
    move_row(ax, 0.35)     
    move_col(ax, 0.075)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # -- plot data --

    # -- limits --
    # Cz, Cheq, Am (Am anom)
    # xmax = 1.5e2
    # xmin = - xmax
    # ymax = 0.1e2
    # ymin = - 0.5e2 
    # vmax = 1e3
    # vmin = - vmax 

    # Cz, Am, Cheq (anom)
    # xmax = 1.5e2
    # xmin = -1.5e2
    # ymax = 2.5e4
    # ymin = -1.25e4
    # vmax = 1e1
    # vmin = - vmax 

    # new
    xmax = None
    xmin = None 
    ymax = None
    ymin = None 
    vmax = 5e1
    vmin = 4.5e1


    ax.scatter(x, y, alpha = 0)                                                                     # needed to place letters
    format_xticks(ax, xmin, xmax)
    format_yticks(ax, ymin, ymax)

    # -- plot letters ontop --
    # norm = mcolors.Normalize(vmin = vmin, vmax = vmax)                                              # colormap
    # cmap = plt.cm.RdBu                                                                              # Choose colormap    
    letters = cL.get_model_letters()                                                                # letter connection to model
    for i, model in enumerate(model_list):                                                          # plot models first, with the colormap
        if model == 'ACCESS-ESM1-5':
            color = 'w'
            text = ax.text(x[i], y[i],                                                                  # data
                            letters[model],                                                             # letter                                   
                            color = color, ha='center', va='center',                                    #
                            fontsize = 6, fontweight = 'bold')                                          #
            text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='black')])         # background effects
        else:
            color = (0.2, 0.2, 0.2)                                                              # Get color from colormap
            text = ax.text(x[i], y[i],                                                                  # data
                            letters[model],                                                             # letter                                   
                            color = color, ha='center', va='center',                                    #
                            fontsize = 6, fontweight = 'bold')                                          #
            text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])         # background effects
    # h = plt.cm.ScalarMappable(cmap=cmap, norm=norm)                                                 # colormap
    # h.set_array([])                                                                                 #
    # cbar_ax = cbar_ax_right(fig, ax, h)


    # -- highlight values changing sign --
    if (x < 0).any() and (x > 0).any():
        ax.axvline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    if (y < 0).any() and (y > 0).any():
        ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)

    return fig, ax


# == main ==
def main():
    # -- x-metric --
    x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km K$^{-1}$' 
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'    

    # -- y-metric --
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km K$^{-1}$'  
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =    'daily',      'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'   
    y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'      
    
    # -- z-metric --
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'       
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =    'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'  
    z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$ K$^{-1}$'   
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  

    # -- normalization --
    n_tfreq,   n_group,   n_name,   n_var,  n_label,    n_units =   'monthly',  'tas',              'tas_map',              'tas_map_mean',                             r'T',           r'$^o$C'    
    # n_tfreq,   n_group,   n_name,   n_var,  n_label,    n_units =   'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_ecs',                       r'T',           r'[$^o$C]'
    # n_tfreq,   n_group,   n_name,   n_var,  n_label,    n_units =   'monthly',  'tas',              'tas_timeseries',       'tas_timeseries_mean',                       r'T',           r'[$^o$C]'

    # -- CMIP metrics --
    model_list = []
    x_list = []
    y_list = []
    z_list = []
    y_hat_list = []
    for model in cL.get_model_letters():
        model_list.append(model)
        data_type_group, data_tyoe, dataset = 'models', 'cmip', model
        lon_area =  '0:360'                                                                                                                                                                             
        lat_area =  '-30:30'                                                                                                                                                                            
        res =       2.8    
        # -- historical --
        time_period =      '1970-01:1999-12'
        x = get_metric(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var).mean(dim = 'time')
        y = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var).mean(dim = 'time')
        z = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area, lat_area, res, time_period, z_var).mean(dim = 'time')
        n = get_metric(data_type_group, data_tyoe, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)

        # -- warm --
        time_period =      '2070-01:2099-12'    
        x_warm = get_metric(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var).mean(dim = 'time')
        y_warm = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var).mean(dim = 'time')
        z_warm = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area, lat_area, res, time_period, z_var).mean(dim = 'time')
        n_warm = get_metric(data_type_group, data_tyoe, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)

        # -- pre-process metric --
        dn = n_warm.mean(dim = ('lat', 'lon'))  - n.mean(dim = ('lat', 'lon'))
        dx = (x_warm - x) / dn
        dy = (y_warm - y) / dn
        dz = (z_warm - z) / dn

        x_list.append(dx.values)
        y_list.append(dy.values)
        z_list.append(dz.values)

    x_array, y_array, z_array = [np.array(f) for f in [x_list, y_list, z_list]]
    # x_array, y_array, z_array = [f - np.mean(f) for f in [x_array, y_array, z_array]]
    x_array, y_array, z_array = [f - np.median(f) for f in [x_array, y_array, z_array]]

    xy_list = [x_array, y_array, z_array]
    y_hat, coeffs = get_plot_metric(xy_list)
    # exit()
    
    # -- plot --
    fig, ax = plot(y_hat, z_array, model_list)
    text = f'CMIP: x: {x_var}, y: {y_var}, \nz: {z_var}'
    # plot_ax_title(fig, ax, text)
    # text = rf'd{z_label} / d{x_label}' 
    # plot_cbar_label(fig, cbar_ax, text)
    # text = f'{z_label} [{z_units}]'
    # plot_cbar_label2(fig, ax, text)

    text = rf'$\hat{{y}}$'
    plot_xlabel(fig, ax, text)

    text = f'{z_label} [{z_units}]'
    plot_ylabel(fig, ax, text)

    # text = rf'$\hat{{y}}$ = {coeffs[0]:.2}C$_z$ + {coeffs[1]:.2}C$_{{heq}}$'
    text = rf'$\hat{{y}}$ = {coeffs[0]:.2} {x_label} + {coeffs[1]:.2} {y_label}'
    plot_ax_title2(fig, ax, text)

    # -- create slope --
    x_pad = 0.0125
    y_pad = - 0.1
    r, p = pearsonr(y_hat, z_array)
    if p < 0.05:
        slope, intercept = np.polyfit(y_hat, z_array, 1)
        y_fit = intercept + slope * y_hat
        ax.plot(y_hat, y_fit, color = 'k', linewidth = 1)
        ax_position = ax.get_position()
        ax.text(ax_position.x0 + x_pad,                                                                                              # x-start
                ax_position.y1 + y_pad,                                                                                              # y-start
                rf'R$^2$ = {r**2:0.2}',                        
                fontsize = 7,  
                transform=fig.transFigure,
                color = 'r'
                )

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'x_{x_var}_y_{y_var}_z_{z_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()


















