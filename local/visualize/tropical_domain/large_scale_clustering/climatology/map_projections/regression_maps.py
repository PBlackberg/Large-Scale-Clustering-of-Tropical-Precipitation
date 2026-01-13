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
# gC = import_relative_module('util_calc.correlations.gridbox_correlation',           'utils')

# == get model subsets ==
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
        # 'CNRM-CM6-1-HR',                                                                                                          # 22  # no clouds
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
        if metric_var == 'tas_gradients_oni':
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
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                 # left
                            ax_position.y0 - 0.25,                                                                            # bottom
                            ax_position.width * w,                                                                            # width
                            ax_position.height * 0.1                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    # cbar.ax.tick_params(labelsize = 7)
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_size(8)
    return cbar_ax

def plot_cbar_label(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.25, # + (ax_position.width * 0.), 
            ax_position.y0 - 0.25, 
            text, 
            ha = 'left', 
            # fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.1225,                                                                                              # x-start
            ax_position.y1 + 0.1,                                                                                              # y-start
            text,                         
            transform=fig.transFigure,
            )
    
def plot_cbar_label2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x1 + 0.135,                                                                                                     # x-start
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2,                                                             # y-start
            text,                                                                              
            rotation = 'vertical', 
            va = 'center', 
            transform=fig.transFigure
            )

def plot(map_corr, map_sig, regression_coeff = None, da_c = None):
    plt.rcParams['font.size'] = 8
    # -- create figure --    
    width, height = 8.5, 3.25                                                                                                            # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
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

    # -- limits --
    vmax = None
    vmin = None 

    # conv_map
    vmax = 2e-5
    vmin = -vmax

    # tas_map
    # vmax = 3e-5
    # vmin = -vmax 

    # LCF
    # vmax = 1e-4
    # vmin = -vmax 

    # rel_humid
    # vmax = 1e-4
    # vmin = -vmax 

    # rel_humid_cm
    # vmax = 6e-2
    # vmin = -vmax 

    # rel_humid_cz
    # vmax = 1e-2
    # vmin = -vmax 

    # LCF_cz
    # vmax = 1e-2
    # vmin = -vmax 

    # LCF_cm
    # vmax = 7.5e-2
    # vmin = -vmax 

    # other
    # vmax = 0.75e-4
    # vmin = - vmax
    # vmax = 1.25e-5
    # vmin = - vmax
    # vmax = 1e-4
    # vmin = - vmax 

    # vmax = 5e-2
    # vmin = - vmax 

    # -- plot data --
    # h = ax.pcolormesh(lonm, latm, regression_coeff.where(map_sig, np.nan), 
    h = ax.pcolormesh(lonm, latm, regression_coeff, 
                        transform=ccrs.PlateCarree(), 
                        # cmap = 'RdBu', 
                        cmap = 'BrBG',      
                        vmin = vmin, 
                        vmax = vmax
                        )

    # -- put significant cross --
    y_indices, x_indices = np.where(map_sig)   
    for x, y in zip(x_indices, y_indices):
        ax.plot(lon[x], lat[y], 'kx', transform=ccrs.PlateCarree(), markersize = 0.1) # 'kx' for black crosses

    # -- put contour --
    contours = ax.contour(lonm, latm, da_c, 
                        transform = ccrs.PlateCarree(),
                        levels =        [da_c.quantile(0.90, dim=('lat', 'lon')).data],
                        colors =        'k', 
                        linewidths =    0.5)

    # -- put colorbar --
    cbar_ax = cbar_ax_below(fig, ax, h)
    return fig, ax, cbar_ax

# == main ==
def main():
    # -- x-metric --
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units = 'daily',    'doc_metrics',        'mean_area',            'mean_area_thres_precip_prctiles_90',         r'A$_m$',       r'km$^2$'    
    x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units = 'daily',    'doc_metrics',        'mean_area',            'mean_area_thres_precip_prctiles_95',         r'A$_m$',       r'km$^2$'    
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units = 'daily',    'doc_metrics',        'mean_area',            'mean_area_thres_precip_prctiles_97',         r'A$_m$',       r'km$^2$'    


    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units = 'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'      
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'  
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'      
    # x_tfreq,   x_group,   x_name,    x_var, x_label,   x_units =  'monthly',  'wap',              'wap_timeseries',       'wap_timeseries_mean',                      r'A$_a$',       r'%'  
    
    # -- y-metric --
    y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =  'daily',    'conv',             'conv_map',             'conv_map_mean',                              r'C',           r'%'      
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =  'monthly',  'tas',              'tas_map',              'tas_map_mean',                             r'T',           r'K'     
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =  'monthly',  'clouds',           'clouds_map',           'clouds_map_low_mean',                      r'LCF',         r'%'      
    # y_tfreq,   y_group,   y_name,   y_var,  y_label,   y_units =  'monthly',  'rel_humid_mid',    'rel_humid_map',        'rel_humid_map_mean',                       r'RH',          r'%'     

    # -- contour --
    c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =  'daily',    'conv',             'conv_map',             'conv_map_mean',                              r'C',           r'%'      
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =  'monthly',  'tas',              'tas_map',              'tas_map_mean',                             r'C',           r'K'      
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =  'monthly',  'clouds',           'clouds_map',           'clouds_map_low_mean',                      r'LCF',         r'%'   
    # c_tfreq,   c_group,   c_name,   c_var,  c_label,   c_units =  'monthly',  'rel_humid_mid',    'rel_humid_map',        'rel_humid_map_mean',                       r'RH',          r'[%]'     

    print(f'plotting {Path(__file__).stem}: x: {x_var}, y: {y_var}, c: {c_var}')
    # exit()

    # -- CMIP metrics --
    ds_x = xr.Dataset()
    ds_y = xr.Dataset()
    ds_c = xr.Dataset()
    for model in cL.get_model_letters():
        if y_group == 'clouds' and not in_subset(model):
            continue
        data_type_group, data_tyoe, dataset = 'models', 'cmip', model
        lon_area =  '0:360'                                                                                                                                                                             
        lat_area =  '-30:30'                                                                                                                                                                            
        res =       2.8    
        # -- historical --
        time_period =      '1970-01:1999-12'
        x = get_metric(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
        y = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
        c = get_metric(data_type_group, data_tyoe, dataset, c_tfreq, c_group, c_name, lon_area, lat_area, res, time_period, c_var)

        # -- pre-process metric --
        dx = x.mean(dim = 'time')
        dy = y

        ds_x[model] = dx
        ds_y[model] = dy
        ds_c[model] = c

    # -- calculate plot metric --
    da_c = ds_to_variable(ds_c).mean(dim = 'variable')
    da_1d = ds_to_variable(ds_x)
    da_3d = ds_to_variable(ds_y)
    map_corr, map_sig, regression_coeff = gC.calculate_correlation_and_significance(da_1d, da_3d)

    # -- plot --
    fig, ax, cbar_ax = plot(map_corr, map_sig, regression_coeff, da_c = da_c)
    text = f'cross-model: x: {x_var}, y: {y_var}, c: {c_var}'
    # plot_ax_title(fig, ax, text)

    text = rf'd{y_label} / d{x_label} [{y_units}/{x_units}]'
    plot_cbar_label(fig, cbar_ax, text)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'a_plot'
    path = f'{folder}/{plot_name}.svg'
    
    # print(path)
    # exit()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()

