'''
# ---------------------
#  Correlation scatter
# ---------------------

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
    # print(path)
    # '/Users/cbla0002/Desktop/work/metrics/observations/GPCP/doc_metrics/reference_proximity/GPCP/reference_proximity_GPCP_daily_0-360_-30-30_128x64_1998-01_2022-12.nc'
    # '/Users/cbla0002/Desktop/work/metrics/observations/GPCP/doc_metrics/reference_proximity/GPCP/reference_proximity_GPCP_daily_0-360_-30-30_128x64_1998-01_2022-12.nc'
    # exit()
    metric = xr.open_dataset(path)
    # print(dataset)
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

def get_metric2(data_type_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- specify metric --
    data_tyoe_group =   'observations'
    data_type =         'GPCP'
    metric_group =      'doc_metrics'
    metric_name =       metric_name
    metric_var =        metric_var
    dataset =           dataset
    t_freq =            'daily'
    lon_area =          '0:360'
    lat_area =          '-30:30'
    resolution =        2.8
    
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    r_filename = (
            f'{metric_name}'   
            f'_{dataset}'                                                                                                   
            f'_{t_freq}'                                                                                                    
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
            )       
    folder_metric = f'{folder}/{r_filename}' # fiels are stored in years here
    year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
    paths = []
    for year in np.arange(int(year_start), int(year_end) + 1):
        path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
        paths.append(path)

    # -- find metric -- 
    doc_metric = xr.open_mfdataset(paths, combine='by_coords')[metric_var].resample(time='1MS').mean().load()
    return doc_metric


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
    # x-ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = 8)
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    # ax.xaxis.set_tick_params(labelsize=0)
    # ax.set_xticklabels('')
    # y-ticks
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = 8) 
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
    
def cbar_ax_below(fig, ax, h):
    ax_position = ax.get_position()
    w = 0.5
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                 # left
                            ax_position.y0 - 0.25,                                                                            # bottom
                            ax_position.width * w,                                                                            # width
                            ax_position.height * 0.1 / 2                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    # ticks = cbar.ax.get_yticks()    
    # try:
    #     ticklabels = [f'{int(t)}' for t in ticks]
    #     cbar.set_ticks(ticks)
    #     cbar.set_ticklabels(ticklabels)
    # except:
    #     pass
    cbar.ax.tick_params(labelsize = 8)
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_size(8)
    # cbar.ax.xaxis.set_offset_position('left')
    return cbar_ax

def plot_cbar_label(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.1,                                                             # x-start + (ax_position.y1 - ax_position.y0) / 2
            ax_position.y0 - 0.15,                                                                                              # y-start
            text,                                                                              
            rotation = 'horizontal', 
            ha = 'left', 
            fontsize = 8, 
            transform=fig.transFigure
            )

# def cbar_ax_right(fig, ax, h):
#     ax_position = ax.get_position()
#     cbar_ax = fig.add_axes([ax_position.x1 + 0.015,                                                                              # left
#                             ax_position.y0 + (ax_position.height - ax_position.height * 0.9) / 2,                               # bottom
#                             ax_position.width * 0.04,                                                                          # width
#                             ax_position.height * 0.9                                                                            # height
#                             ])      
#     cbar = fig.colorbar(h, cax = cbar_ax, orientation='vertical')
#     ticks = cbar.ax.get_yticks()    
#     try:
#         ticklabels = [f'{int(t)}' for t in ticks]
#         cbar.set_ticks(ticks)
#         cbar.set_ticklabels(ticklabels)
#     except:
#         pass
#     formatter = ticker.ScalarFormatter(useMathText = True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1, 1))
#     cbar.ax.yaxis.set_major_formatter(formatter)
#     cbar.ax.tick_params(labelsize = 7)
#     cbar.ax.yaxis.get_offset_text().set_size(7)
#     cbar.ax.yaxis.set_offset_position('left')
#     return cbar_ax

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

def plot(x, y, z):
    # -- create figure --    
    # width, height = 6.27, 9.69                                                                                                  # max size (for 1 inch margins)
    width, height = 5, 5                                                                                                        # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    # scale_ax(ax, 0.75)
    scale_ax_x(ax, 0.75)
    scale_ax_y(ax, 0.6)
    # colorbar right
    # move_row(ax, 0.1)     
    # move_col(ax, 0.075)

    # colorbar below
    move_row(ax, 0.35)     
    move_col(ax, 0.075)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # -- plot data --

    # -- limits --
    # Af and Am
    # xmax = 1.1e-2
    # xmin = - xmax
    xmax = None
    xmin = None
    ymax = None
    ymin = None
    vmax = None
    vmin = None

    xmax = 2.75e6
    xmin = - xmax

    ymax = 2.25e5
    ymin = - ymax

    # Cz
    # vmax = 1e3
    # vmin = - vmax

    # tas
    vmax = 1.5 * 10
    vmin = - vmax


    h = ax.scatter(x, y, c = z, cmap = 'RdBu_r', s = 10, vmin = vmin, vmax = vmax)
    format_xticks(ax, xmin, xmax)
    format_yticks(ax, ymin, ymax)
    # cbar_ax = cbar_ax_right(fig, ax, h)
    cbar_ax = cbar_ax_below(fig, ax, h)

    # -- highlight values changing sign --
    if (x < 0).any() and (x > 0).any():
        ax.axvline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    if (y < 0).any() and (y > 0).any():
        ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)

    return fig, ax, cbar_ax


# == main ==
def main():
    # -- x-metric --
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_90',   r'A$_f$',       r'km$^2$'   
    x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_95',   r'C',       r'km$^2$'   
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_97',   r'A$_f$',       r'km$^2$'   

    # -- y-metric --
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',       r'A$_m$',       r'km$^2$'    
    y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',       r'A$_m$',       r'km$^2$'    
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',       r'A$_m$',       r'km$^2$'    

    # -- z-metric --
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_90',      r'C$_z$',       r'km'  
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_95',      r'P$_z$',       r'km'  
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line_thres_precip_prctiles_97',      r'C$_z$',       r'km'  

    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_m$',       r'km'   
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'   
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'monthly',    'tas',              'tas_gradients',        'tas_gradients_oni',                        r'ONI',         r'K'   
    z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'monthly',    'slp',              'slp_gradients',        'slp_gradients_SOI',                        r'SOI',         r'K'   

    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',    'doc_metrics',      'gini',                   'gini',                                     r'GINI',       r''    


    # --  metrics --
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8    
    data_type_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_tyoe, dataset = 'observations', 'ISCCP', 'ISCCP'
    # data_type_group, data_tyoe, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
    time_period = '1998-01:2022-12'      
    x = get_metric2(data_type_group, data_tyoe, dataset, x_tfreq, x_group, x_name, lon_area, lat_area, res, time_period, x_var)
    y = get_metric2(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var)
    # data_type_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
    # data_type_group, data_tyoe, dataset = 'observations', 'ISCCP', 'ISCCP'
    # data_type_group, data_tyoe, dataset = 'observations', 'NOAA', 'NOAA'
    data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
    z = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area, lat_area, res, time_period, z_var) * 10
    # print(z.min().data)
    # print(z.max().data)
    # print(z)
    # exit()

    # -- pre-process metric --
    x, y, z = [cA.detrend_data(da) for da in [x, y, z]]    
    x, y, z = [cA.get_monthly_anomalies(da) for da in [x, y, z]]    
    x, y, z = xr.align(x.dropna(dim='time', how='any'), y.dropna(dim='time', how='any'), z.dropna(dim='time', how='any'), join='inner')     
    x, y, z = [f.data for f in [x, y, z]]
    # [print(type(f)) for f in [x, y, z]]
    # exit()

    # -- calculate plot metric --

    # -- plot --
    fig, ax, cbar_ax = plot(x, y, z)
    text = f'OBS: x: {x_var}, y: {y_var}, \nz: {z_var}'
    # plot_ax_title(fig, ax, text)
    # text = rf'd{z_label} / d{x_label}' 
    # plot_cbar_label(fig, cbar_ax, text)
    text = f'{z_label} [{z_units}]'
    plot_cbar_label(fig, cbar_ax, text)          # cbar below
    # plot_cbar_label2(fig, ax, text)       # cbar right

    text = f'{x_label} [{x_units}]'
    plot_xlabel(fig, ax, text)

    text = f'{y_label} [{y_units}]'
    plot_ylabel(fig, ax, text)

    # -- create slope --
    x_pad = 0.0125
    y_pad = - 0.1
    r, p = pearsonr(x, y)
    if p < 0.05:
        ax_position = ax.get_position()
        ax.text(ax_position.x0 + x_pad,                                                                                              # x-start
                ax_position.y1 + y_pad,                                                                                              # y-start
                rf'R$^2$ = {r**2:0.2}',                        
                fontsize = 8,  
                transform=fig.transFigure,
                color = 'r'
                )
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = intercept + slope * x
        ax.plot(x, y_fit, color = 'k', linewidth = 0.5)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'a_plot'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()



