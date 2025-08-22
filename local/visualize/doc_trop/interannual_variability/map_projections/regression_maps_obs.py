'''
# -------------------------------
#  Regression maps: observations
# -------------------------------

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


# == plot funcs ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds                                                                                       # [left, bottom, width, height]
    new_width = _1 * scaleby
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
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = 8)
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = 8) 
    ax.yaxis.set_tick_params(length = 2)
    ax.yaxis.set_tick_params(width = 1)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(right = False)

def plot_xlabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - 0.2, 
            text, 
            ha = 'center', 
            transform = fig.transFigure
            )
    
def plot_ylabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.125, 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            text, 
            va = 'center', 
            rotation = 'vertical', 
            transform = fig.transFigure
            )

def cbar_ax_below(fig, ax, h):
    ax_position = ax.get_position()
    w = 0.5
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                       # left
                            ax_position.y0 - 0.25,                                                                                  # bottom
                            ax_position.width * w,                                                                                  # width
                            ax_position.height * 0.125                                                                              # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_size(8)
    return cbar_ax
    
def plot_cbar_label(fig, ax, text, move = 0):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + move,
            ax_position.y0 - 0.25, 
            text, 
            ha = 'left', 
            fontsize = 8, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.1225,                                                                                                # x-start
            ax_position.y1 + 0.1,                                                                                                   # y-start
            text,                        
            fontsize = 8,  
            transform=fig.transFigure,
            )

def plot(map_corr, map_sig, regression_coeff, da_c):
    # -- set general fontsize --    
    plt.rcParams['font.size'] = 8

    # -- create figure --    
    width, height = 8.5, 3.25                                                                                                       # max: 8.5 (single-column) 17.5 (double column) [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                             # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
    ax.remove()
    projection = ccrs.PlateCarree(central_longitude = 180)
    ax = fig.add_subplot(nrows, ncols, 1, projection=projection)

    # -- format ax --
    lat, lon = map_corr.lat, map_corr.lon
    lonm,latm = np.meshgrid(lon, lat)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    xticks = [60, 120, 180, 240, 300]
    yticks = [-20, 0, 20]
    scale_ax(ax, 1.1)
    move_row(ax, 0.225)     
    move_col(ax, 0)
    plot_ticks(ax, xticks, yticks)
    ax.coastlines(resolution = "110m", linewidth = 0.6)

    # -- limits (change for chosen variable) --
    vmax = None
    vmin = None 

    # [conv_map]
    vmax = 1e-4
    vmin = -vmax 

    # [tas map]
    # vmax = 7e-6
    # vmin = -vmax

    # [LCF map]
    # vmax = 1e-4
    # vmin = -vmax 

    # [HCF map]
    # vmax = 1e-4
    # vmin = -vmax 

    # [rel humid map]
    # vmax = 1e-4
    # vmin = -vmax 

    # -- plot data --
    h = ax.pcolormesh(lonm, latm, regression_coeff, 
                          transform=ccrs.PlateCarree(), 
                          cmap = 'RdBu', 
                          vmin = vmin, 
                          vmax = vmax
                          )
    
    # -- put significant cross --
    y_indices, x_indices = np.where(map_sig)   
    for x, y in zip(x_indices, y_indices):
        ax.plot(lon[x], lat[y], 'kx', transform=ccrs.PlateCarree(), markersize = 0.1)

    # -- put contour --
    ax.contour(lonm, latm, da_c, 
                        transform = ccrs.PlateCarree(),
                        levels =        [da_c.quantile(0.90, dim=('lat', 'lon')).data],   # for wap: [0] (change as needed)    
                        colors =        'k', 
                        linewidths =    0.5)

    # -- put colorbar --
    cbar_ax = cbar_ax_below(fig, ax, h)
    return fig, ax, cbar_ax

# == main ==
def main():
    # == specify metric ==
    # -- x-metric --
    x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',        'doc_metrics',    'mean_area',            'mean_area',                            r'A$_m$',       r'km$^2$'    
    
    # -- y-metric --
    y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =   'daily',         'conv',           'conv_map',             'conv_map_mean',                        r'C',           r'%'      
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =   'monthly',       'tas',            'tas_map',              'tas_map_mean',                         r'Ts',          r'K'         
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =   'monthly',       'clouds',         'cloud_map',            'cloud_map',                            r'LCF',         r'%'
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =   'monthly',       'rel_humid_mid',  'rel_humid_map',        'rel_humid_map_mean',                   r'RH',          r'%'   
    
    # -- contour --
    c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =   'daily',         'conv',           'conv_map',             'conv_map_mean',                        r'C',           r'%'      
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =   'monthly',       'tas',            'tas_map',              'tas_map_mean',                         r'T',           r'K'      
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =   'monthly',       'clouds',         'cloud_map',            'cloud_map_low_mean',                   r'LCF',         r'%'  
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =   'monthly',       'rel_humid_mid',  'rel_humid_map',        'rel_humid_map_low_mean',               r'RH',          r'%'
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =   'monthly',       'wap',            'wap_map',              'wap_map_mean',                         r'$\omega$',    r'Pa day$^{-1}$'  

    print(f'plotting {Path(__file__).stem}: x: {x_var}, y: {y_var}, c: {c_var}')

    # == get metric ==
    lon_area =      '0:360'                                                                                                                                                                             
    lat_area =      '-30:30'                                                                                                                                                                            
    res =           2.8    
    time_period =   '1998-01:2022-12'    
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_type, dataset = 'observations', 'ISCCP', 'ISCCP'
    # data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
    
    # -- central metric --
    if y_group == 'clouds':
        map_corr = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_low_vs_{x_var}_corr')
        map_sig = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_low_vs_{x_var}_sig')
        regression_coeff = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_low_vs_{x_var}_regress')
    else:
        map_corr = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_vs_{x_var}_corr')
        map_sig = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_vs_{x_var}_sig')
        regression_coeff = get_metric(data_type_group, data_type, dataset, y_tfreq, y_group, f'{y_name}_correlation', lon_area, lat_area, res, time_period, f'{y_name}_correlation_vs_{x_var}_regress')
    
    # -- contour --
    data_type_group, data_type, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_tyoe, dataset = 'observations', 'ISCCP', 'ISCCP'
    # data_type_group, data_type, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
    da_c = get_metric(data_type_group, data_type, dataset, c_tfreq, c_group, c_name, lon_area, lat_area, res, time_period, c_var)

    # == plot ==
    # -- plot data --    
    fig, ax, cbar_ax = plot(map_corr, map_sig, regression_coeff, da_c)

    # -- plot labels --    
    text = rf'd{y_label} / d{x_label} [{y_units} / {x_units}]' 
    plot_cbar_label(fig, cbar_ax, text, move = -0.25)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'{Path(__file__).stem}_{x_var}_y_{y_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    plt.close(fig)
    print(f'plot saved at: {path}')
    # exit()


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()




