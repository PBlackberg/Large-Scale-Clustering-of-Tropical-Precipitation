'''
# ---------------------
#  Regression scatter
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
from matplotlib.colors import LogNorm

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

def pre_process_metric(xy_list):
    xy_list = [cA.detrend_data(da) for da in xy_list]    
    xy_list = [cA.get_monthly_anomalies(da) for da in xy_list]    
    xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
    xy_list = list(xr.align(*xy_list, join='inner'))
    xy_list = [da.data for da in xy_list]
    # [print(type(f)) for f in xy_list]
    # exit()
    return xy_list

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

def get_area_matrix(lat, lon):
    ''' # area of domain: cos(lat) * (dlon * dlat) R^2 (area of gridbox decrease towards the pole as gridlines converge) '''
    lonm, latm = np.meshgrid(lon, lat)
    dlat = lat.diff(dim='lat').data[0]
    dlon = lon.diff(dim='lon').data[0]
    R = 6371     # km
    area =  np.cos(np.deg2rad(latm))*np.float64(dlon * dlat * R**2*(np.pi/180)**2) 
    da_area = xr.DataArray(data = area, dims = ["lat", "lon"], coords = {"lat": lat, "lon": lon}, name = "area")
    return da_area


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
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, # + 0.05, # + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - 0.15, 
            text, 
            ha = 'center', 
            # fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ylabel(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.185, 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            text, 
            va = 'center', 
            rotation = 'vertical', 
            # fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.15,                                                                                              # x-start
            ax_position.y1 + 0.125,                                                                                              # y-start
            text,                        
            # fontsize = 4,  
            transform=fig.transFigure,
            )
    
def plot_ax_title2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.15,                                                                                              # x-start
            ax_position.y1 + 0.1,                                                                                              # y-start
            text,                        
            # fontsize = 5,  
            ha = 'left', 
            transform=fig.transFigure,
            )

# def cbar_ax_right(fig, ax, h):
#     ax_position = ax.get_position()
#     cbar_ax = fig.add_axes([ax_position.x1 + 0.0125,                                                                              # left
#                             ax_position.y0 + (ax_position.height - ax_position.height * 0.9) / 2,                               # bottom
#                             ax_position.width * 0.025,                                                                          # width
#                             ax_position.height * 0.9                                                                            # height
#                             ])      
#     cbar = fig.colorbar(h, cax = cbar_ax, orientation='vertical')
#     ticks = cbar.ax.get_yticks()    
#     # try:
#     #     ticklabels = [f'{int(t)}' for t in ticks]
#     #     cbar.set_ticks(ticks)
#     #     cbar.set_ticklabels(ticklabels)
#     # except:
#     #     pass
#     formatter = ticker.ScalarFormatter(useMathText = True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1, 1))
#     cbar.ax.yaxis.set_major_formatter(formatter)
#     cbar.ax.tick_params(labelsize = 6)
#     cbar.ax.yaxis.get_offset_text().set_size(7)
#     cbar.ax.yaxis.set_offset_position('left')
#     cbar.ax.tick_params(pad=1)
#     return cbar_ax

def cbar_ax_below(fig, ax, h):
    ax_position = ax.get_position()
    w = 0.8
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                 # left
                            ax_position.y0 - 0.215,                                                                            # bottom
                            ax_position.width * w,                                                                            # width
                            ax_position.height * 0.1/2                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    # cbar.ax.tick_params(labelsize = 7)
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.get_offset_text().set_size(7)
    cbar.ax.yaxis.set_offset_position('left')
    return cbar_ax
    
# def plot_cbar_label1(fig, ax, text):
#     ax_position = ax.get_position()
#     ax.text(ax_position.x1 - 0.115, 
#             ax_position.y1 + 0.075, 
#             text, 
#             ha = 'center', 
#             # fontsize = 7, 
#             transform = fig.transFigure
#             )

def plot_cbar_label(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.width * 0.5), 
            ax_position.y0 - 0.15, 
            text, 
            ha = 'center', 
            # fontsize = 7, 
            transform = fig.transFigure
            )
    
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

def plot_cbar_label2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x1 + 0.14,                                                                                                     # x-start
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2,                                                             # y-start
            text,                                                                              
            rotation = 'vertical', 
            va = 'center', 
            # fontsize = 7, 
            transform=fig.transFigure
            )
    
def plot(x, y):
    plt.rcParams['font.size'] = 7
    # -- create figure --    
    # width, height = 6.27, 9.69                                                                                                  # max size (for 1 inch margins)
    width, height = 5, 5                                                                                                        # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax_x(ax, 0.75)
    scale_ax_y(ax, 0.6)
    move_row(ax, 0.275)     
    move_col(ax, 0.075)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # -- plot data --
    # xmax = 1.1e-2
    # xmin = - xmax
    # ymax = 2.25e5
    # ymin = - ymax
    # vmax = 1e3
    # vmin = - vmax

    xmax = 3
    xmin = - xmax
    ymax = 2.25e5
    ymin = - ymax
    # h = ax.scatter(x, y, s = 10)
    h = ax.hist2d(x, y, bins = 20, cmap='Greys', norm=LogNorm())  # vmin = 1e-2, vmax = 20)

    format_xticks(ax, xmin, xmax)
    format_yticks(ax, ymin, ymax)
    # cbar_ax = cbar_ax_right(fig, ax, h[3])
    cbar_ax = cbar_ax_below(fig, ax, h[3])

    # -- highlight values changing sign --
    if (x < 0).any() and (x > 0).any():
        ax.axvline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    if (y < 0).any() and (y > 0).any():
        ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
        
    return fig, ax, cbar_ax


# == main ==
def main():
    # -- x1-metric --
    x1_tfreq,    x1_group,  x1_name,    x1_var, x1_label,   x1_units =   'daily',   'doc_metrics',      'area_fraction',        'area_fraction',                            r'A$_f$',       r''   

    # -- x2-metric --    
    x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_meridional_line',      r'C$_z$',       r'km'   

    # -- x3-metric --    
    # x3_tfreq,   x3_group,   x3_name,    x3_var, x3_label,   x3_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  
    x3_tfreq,   x3_group,   x3_name,    x3_var, x3_label,   x3_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_{m}$',     r'km'  

    # -- x4-metric --    
    x4_tfreq,   x4_group,   x4_name,    x4_var, x4_label,   x4_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_zonal_line',           r'C$_{m}$',     r'km'  
    # x4_tfreq,   x4_group,   x4_name,    x4_var, x4_label,   x4_units =  'daily',    'doc_metrics',      'reference_proximity',  'reference_proximity_eq_hydro',             r'C$_{heq}$',   r'km'  

    # -- y-metric --
    y_tfreq,   y_group,     y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',      'mean_area',            'mean_area',                                r'A$_m$',       r'km$^2$'

    # -- normalization --
    n_tfreq,   n_group,     n_name,     n_var,  n_label,    n_units =   'daily',    'conv',              'conv_map',            'conv_map_mean',                            r'C',           r'%'    


    # --  metrics --
    lon_area =  '0:360'                                                                                                                                                                             
    lat_area =  '-30:30'                                                                                                                                                                            
    res =       2.8    
    data_type_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
    time_period = '1998-01:2022-12'      
    n = get_metric(data_type_group, data_tyoe, dataset, n_tfreq, n_group, n_name, lon_area, lat_area, res, time_period, n_var)
    domain_area = get_area_matrix(n.lat, n.lon).sum(dim = {'lat', 'lon'}).data
    length_scale = np.sqrt(domain_area)
    x1 = get_metric(data_type_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, time_period, x1_var) # * domain_area
    x2 = get_metric(data_type_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, time_period, x2_var) # / length_scale
    x3 = get_metric(data_type_group, data_tyoe, dataset, x3_tfreq, x3_group, x3_name, lon_area, lat_area, res, time_period, x3_var) # / length_scale
    x4 = get_metric(data_type_group, data_tyoe, dataset, x4_tfreq, x4_group, x4_name, lon_area, lat_area, res, time_period, x4_var) # / length_scale
    y = get_metric(data_type_group, data_tyoe, dataset, y_tfreq, y_group, y_name, lon_area, lat_area, res, time_period, y_var) # / domain_area

    # -- pre-process metric --
    # xy_list = [x1, x2, x3, x4, y]
    xy_list = [x1, x2, x3, y]
    xy_list = pre_process_metric(xy_list)

    # -- calculate plot metric --
    y_hat, coeffs = get_plot_metric(xy_list)
    # print(y_hat)
    # print(coeffs)
    # exit()

    # -- plot data --
    y_plot = xy_list[-1] - np.mean(xy_list[-1])
    fig, ax, cbar_ax = plot(y_hat, y_plot)
    text = f'OBS, LM: x: {x1_label}\n{x2_label}, {x3_label}\n{y_label}'

    # -- plot labels --
    # plot_ax_title(fig, ax, text)
    text = rf'Data density [Nb]' 
    plot_cbar_label(fig, cbar_ax, text)

    text = rf'$\hat{{y}}$ []' #[{y_units}]'
    plot_xlabel(fig, ax, text)
    
    # text = rf'$\hat{{y}}_{{spatial}}$ []' #[{y_units}]'
    text = f'{y_label} [{y_units}]'
    plot_ylabel(fig, ax, text)

    # text = rf'$\hat{{y}}$ = $\alpha$A$_f$ + $\beta$C$_z$ + $\gamma$C$_{{heq}}$'
    # text = rf'$\hat{{y}}$ = $\alpha$A$_f$ + $\beta$C$_z$ + $\gamma$C$_{{heq}}$ + $\delta$C$_{{m}}$'
    # text = (rf'$\hat{{y}}$ = {coeffs[0]:.2}A$_f$'  +
    #         rf'{coeffs[1]:.2}C$_z$' + '\n' +
    #         rf'{coeffs[2]:.2}C$_{{heq}}$') 
    text = rf'$\hat{{y}}$ = {coeffs[0]:.2} {x1_label} + {coeffs[1]:.2} {x2_label} + {coeffs[2]:.2} {x3_label}'
    plot_ax_title2(fig, ax, text)

    # -- create slope --
    x_pad = 0.0125
    y_pad = - 0.1
    r, p = pearsonr(y_hat, y_plot)
    if p < 0.05:
        slope, intercept = np.polyfit(y_hat, y_plot, 1)
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
    # fig.text(0.75,                                                                                              # x-start
    #         0.5,                                                                                              # y-start
    #         rf'f)',                        
    #         fontsize = 7,  
    #         fontweight = 'bold',  
    #         transform=fig.transFigure,
    #         )

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'LM_x_{x1_var}_{x2_var}_{x3_var}_y_{y_var}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == when this script is ran / global variables ==
if __name__ == '__main__':    
    main()







