'''
# ---------------------
#   Scatter, binned
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
from scipy.stats import gaussian_kde

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


# == plot ==
def z_percentiles(yb, zb, qs):
    '''
    Input:
    y, z subsetted to the timecoordinates that are wihtin the area fraction bin (ex: the mean area and RH for area fraction 0-0.1)
    qs - quantiles in y to further subset (ex: the RH within 0-0.2 percentile mean area, wihtin the 0-0.1 bin in area fraction)
    The indicesa for yb percentiles are found, and zb is then indexed using those (then taking the meean in the bin).
    '''
    if len(yb) == 0:
        return np.full(len(qs)+1, np.nan)
    q_edges = np.quantile(yb, [0, *qs, 1])
    z_vals = []
    for qlo, qhi in zip(q_edges[:-1], q_edges[1:]):
        sel = (yb >= qlo) & (yb <= qhi)
        if np.any(sel):
            z_vals.append(np.nanmedian(zb[sel]))
            # z_vals.append(np.nanmean(zb[sel]))
        else:
            z_vals.append(np.nan)        
    return np.array(z_vals)

def plot(x, y, z):
    # -- create figure --    
    width, height = 7, 6.5
    width, height = [f / 2.54 for f in [width, height]] # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, 0.55)
    scale_ax_y(ax, 0.5)
    move_row(ax, 0.25)     
    move_col(ax, 0.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # -- limits --
    xmax = None
    xmin = None
    ymax = None
    ymin = None
    vmax = None
    vmin = None

    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(z)
    ymin = np.min(z)
    vmax = None
    vmin = None
 
    # -- bins --
    p = np.linspace(0, 1, 11)               # quantiles in for example Af (10 bins, 0.1, 0.2, 0.3, ..)
    # print(p)
    # exit()
    edges = np.quantile(x, p)
    x_lowers = edges[:-1]
    x_uppers = edges[1:] 

    # -- y percentile points and its associated z --
    # qs = [0.20, 0.40, 0.60, 0.80]                                                               # quantiles in for example mean area
    qs = [0.33, 0.66]                                                               # quantiles in for example mean area

    z_ypercentiles = np.full((len(qs) + 1, len(x_lowers)), np.nan)
    for i, (lo, hi) in enumerate(zip(x_lowers, x_uppers)):
        m = (x >= lo) & (x < hi)                                                                # include rightmost bin separately if needed
        z_ypercentiles[:, i] = z_percentiles(y[m], z[m], qs)
        # exit()

    # -- plot data --
    # colors = ['r', 'orange', 'g', 'cyan', 'b']
    # labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

    colors = ['b', 'g', 'r']
    labels = ['0-0.33', '0.33-0.66', '0.66-1']

    lw = 0.5
    for z_values, color in zip(z_ypercentiles, colors):
        for i, (x_lower, x_upper) in enumerate(zip(x_lowers, x_uppers)):
            # plt.plot([x_lower, x_lower], [np.min(z), z_values[i]], color = color)                                             # left vertical line
            plt.plot([x_lower, x_upper], [z_values[i], z_values[i]], color = color, linewidth=lw)                               # horizontal top line
            if i > 0 and z_values[i] <= z_values[i-1]:                                                                          #
                plt.plot([x_lower, x_lower], [z_values[i], z_values[i - 1]], color = color, linewidth=lw)                       # right vertical line down
            elif i > 0 and z_values[i] > z_values[i-1]:                                                                         #
                plt.plot([x_lower, x_lower], [z_values[i], z_values[i - 1]], color = color, linewidth=lw)                       # right vertical line up
            else:
                pass

    for label, color in zip(labels[::-1], colors[::-1]):
        plt.plot([], [], color = color, label = label)
    # plt.legend(frameon=False, fontsize=4, handlelength=1, handletextpad=0.3, labelspacing=0.2, loc='lower right')

    # plt.legend(
    #     frameon=False, fontsize=4, handlelength=1, handletextpad=0.3,
    #     labelspacing=0.2, bbox_to_anchor=(1.06, 0.95)
    # )

    plt.legend(
        frameon=False, fontsize=4, handlelength=1, handletextpad=0.3,
        labelspacing=0.2, bbox_to_anchor=(0.4, 0.95)
    )

    text = r'A$_m$ percentiles:'
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + 0.025,                                                                                              # x-start
            ax_position.y1 - 0.02,                                                                                             # y-start
            text,                        
            fontsize = 4,  
            transform=fig.transFigure,
            )

    # -- plot colorbar --
    # cbar_ax = cbar_ax_below(fig, ax, h)
    cbar_ax = ''
    format_xticks(ax, xmin, xmax)
    format_yticks(ax, ymin, ymax)

    # -- add density distribution --
    n = 100
    pos = ax.get_position()
    # top
    ax_histx = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.11], sharex=ax)
    kde = gaussian_kde(x)
    x_vals = np.linspace(min(x), max(x), n)
    ax_histx.axvline(x.mean(), color='k', linestyle='--', linewidth=0.5)
    ax_histx.plot(x_vals, kde(x_vals), linewidth = 0.5, color = 'k')
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['bottom'].set_linewidth(0.5)
    ax_histx.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_histx.set_yticks([])
    ax_histx.set_ylabel("")
    # right
    ax_histy = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.11, pos.height], sharey=ax)
    kde_y = gaussian_kde(z)
    y_vals = np.linspace(min(z), max(z), n)
    ax_histy.axhline(z.mean(), color='k', linestyle='--', linewidth=0.5)
    ax_histy.plot(kde_y(y_vals), y_vals, linewidth=0.5, color='k')
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.spines['left'].set_linewidth(0.5)
    ax_histy.tick_params(axis='y', which='both', labelleft=False, left=False)
    ax_histy.set_xticks([])
    ax_histy.set_xlabel("")
    return fig, ax, cbar_ax


# == main ==
def main():
    # -- x-metric --
    x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_90',   r'A$_f$',       r''   
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_95',   r'A$_f$',       r''   
    # x_tfreq,    x_group,    x_name, x_var,  x_label,    x_units =   'daily',      'doc_metrics',      'area_fraction',      'area_fraction_thres_precip_prctiles_97',   r'A$_f$',       r''   

    # -- y-metric --
    y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_90',       r'A$_m$',       r'km$^2$'    
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_95',       r'A$_m$',       r'km$^2$'    
    # y_tfreq,   y_group,   y_name,    y_var, y_label,   y_units =  'daily',      'doc_metrics',      'mean_area',            'mean_area_thres_precip_prctiles_97',       r'A$_m$',       r'km$^2$'    

    # -- z-metric --
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'$\Delta$C$_m$',       r'km'   
    # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'   
    z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'monthly',    'tas',              'tas_gradients',        'tas_gradients_oni',                        r'ONI',         r'K'   

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
    data_type_group, data_tyoe, dataset = 'observations', 'NOAA', 'NOAA'
    # data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
    z = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area, lat_area, res, time_period, z_var)

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
    # text = f'{z_label} [{z_units}]'
    # plot_cbar_label(fig, cbar_ax, text)          # cbar below
    # plot_cbar_label2(fig, ax, text)       # cbar right

    text = f'{x_label} [{x_units}]'
    plot_xlabel(fig, ax, text)

    # text = f'{y_label} [{y_units}]'
    text = f'{z_label} [{z_units}]'
    plot_ylabel(fig, ax, text)

    # # -- create slope --
    # x_pad = 0.0125
    # y_pad = - 0.1
    # r, p = pearsonr(x, y)
    # if p < 0.05:
    #     ax_position = ax.get_position()
    #     ax.text(ax_position.x0 + x_pad,                                                                                              # x-start
    #             ax_position.y1 + y_pad,                                                                                              # y-start
    #             rf'R$^2$ = {r**2:0.2}',                        
    #             fontsize = 8,  
    #             transform=fig.transFigure,
    #             color = 'r'
    #             )
    #     slope, intercept = np.polyfit(x, y, 1)
    #     y_fit = intercept + slope * x
    #     ax.plot(x, y_fit, color = 'k', linewidth = 0.5)

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[3].name}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'x_{x_var}_y_{y_var}_z_{z_var}'
    path = f'{folder}/{plot_name}.png'
    # print(path)
    # exit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()







