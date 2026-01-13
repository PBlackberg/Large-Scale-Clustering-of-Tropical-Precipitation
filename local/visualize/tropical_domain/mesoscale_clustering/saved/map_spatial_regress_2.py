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
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

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
    # exit()
    # '/Users/cbla0002/Desktop/work/metrics/observations/ERA5/satfrac/satfrac_regress2/ERA5/satfrac_regress2_ERA5_daily_0-360_-30-30_360x180_2001-01_2021-12.nc'
    metric = xr.open_dataset(path)
    # print(metric)
    # exit()
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        metric = metric[metric_var]
    return metric


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
    # x-ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) 
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = 7)
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    # y-ticks
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = 7) 
    ax.yaxis.set_tick_params(length = 2)
    ax.yaxis.set_tick_params(width = 1)
    # both
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
    w = 0.6
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,                                   # left
                            ax_position.y0 - 0.25,                                                                              # bottom
                            ax_position.width * w,                                                                              # width
                            ax_position.height * 0.1                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    # formatter = ticker.ScalarFormatter(useMathText = True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_size(7)
    return cbar_ax

def plot_cbar_label(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.15, # + (ax_position.width * 0.), 
            ax_position.y0 - 0.25, 
            text, 
            ha = 'left', 
            fontsize = 7, 
            transform = fig.transFigure
            )
    
def plot_ax_title(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.1225,                                                                                            # x-start
            ax_position.y1 + 0.1,                                                                                               # y-start
            text,                        
            fontsize = 6,  
            transform=fig.transFigure,
            )

def plot_ax_title2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + 0.1225,                                                                                            # x-start
            ax_position.y1 + 0.1,                                                                                               # y-start
            text,                        
            fontsize = 7,  
            transform=fig.transFigure,
            )
    
def plot_cbar_label2(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x1 + 0.135,                                                                                             # x-start
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2,                                                             # y-start
            text,                                                                              
            rotation = 'vertical', 
            va = 'center', 
            # fontsize = 7, 
            transform=fig.transFigure
            )

def add_rectangles(ax, color = 'white', label = 'meso', idx_rect = 0):
    lon_areas = (                                                                                                                   # set lon extent
        '0:49',                                                                                                                   # Africa
        '50:99',                                                                                                                  # Indian Ocean
        '100:149',                                                                                                                # Maritime Continent
        '150:204',                                                                                                                # West / central Pacific    (55 degrees, 5 degrees wider)
        '205:259',                                                                                                                # East Pacific              (55 degrees, 5 degrees wider)
        '260:309',                                                                                                                # Amazon
        '310:359',                                                                                                                # Atlantic
        '0:359',        
        )               
    labels = (
        'Africa',                                                                              
        'Indian Ocean',                                                                                          
        'MTC',                                                                               
        'West Pacific',                                                                                                   
        'East Pacific',                                                                                                          
        'Amazon',                                                                                                              
        'Atlantic', 
        'Deep Tropics',                                                                                                              
        ) 

    if idx_rect == 'all':
        for idx, (lon_area, label) in enumerate(zip(lon_areas[:-1], labels[:-1])):
            lon_start = int(lon_area.split(':')[0])
            lon_end = int(lon_area.split(':')[1])
            lat_start = -13
            lat_end = 13
            rect = Rectangle((lon_start, lat_start),                                        # (left, bottom)
                            lon_end - lon_start,                                            # width
                            abs(lat_start - lat_end),                                       # height
                            linewidth = 0.5, edgecolor=color, facecolor='none', 
                            transform=ccrs.PlateCarree(),
                            path_effects=[withStroke(linewidth = 1, foreground='black')])
            ax.add_patch(rect)
            ax.text((lon_start + lon_end) / 2, 
                    lat_end + 0.25, 
                    label, 
                    fontsize = 3, transform = ccrs.PlateCarree(), 
                    color = 'w', #''lightgrey', 
                    weight = 'bold', ha = 'center', va = 'bottom',
                    path_effects=[withStroke(linewidth = 2, foreground = 'black')]) 
    else:
        for idx, (lon_area, label) in enumerate(zip(lon_areas, labels)):
            if idx == idx_rect:
                lon_start = int(lon_area.split(':')[0])
                lon_end = int(lon_area.split(':')[1])
                lat_start = -13
                lat_end = 13
                rect = Rectangle((lon_start, lat_start),                                        # (left, bottom)
                                lon_end - lon_start,                                            # width
                                abs(lat_start - lat_end),                                       # height
                                linewidth = 0.5, edgecolor=color, facecolor='none', 
                                transform=ccrs.PlateCarree(),
                                path_effects=[withStroke(linewidth = 1, foreground='black')])
                ax.add_patch(rect)
                ax.text((lon_start + lon_end) / 2, 
                        lat_end + 0.25, 
                        label, 
                        fontsize = 3, transform = ccrs.PlateCarree(), 
                        color = 'w', #''lightgrey', 
                        weight = 'bold', ha = 'center', va = 'bottom',
                        path_effects=[withStroke(linewidth = 2, foreground = 'black')]) 

def get_area_matrix(lat, lon):
    lonm, latm = np.meshgrid(lon, lat)
    dlat = lat.diff(dim='lat').data[0]
    dlon = lon.diff(dim='lon').data[0]
    R = 6371     # km
    area =  np.cos(np.deg2rad(latm))*np.float64(dlon * dlat * R**2*(np.pi/180)**2) # area of domain: cos(lat) * (dlon * dlat) R^2 (area of gridbox decrease towards the pole as gridlines converge)
    da_area = xr.DataArray(data = area, dims = ["lat", "lon"], coords = {"lat": lat, "lon": lon}, name = "area")
    return da_area

def get_area_normalization_factor(lon_area):
    # -- tropical grid --
    path = '/Users/cbla0002/Desktop/work/data/IMERG_data/i_org_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2023-12_var_2001_1_1.nc'
    da = xr.open_dataset(path).isel(time = 0)['var']
    lat_area = '-13:13'
    # lon_area = '100:149'
    # print(da)
    # print(lon_area)
    # print(int(lon_area.split('-')[0]))
    # print(int(lon_area.split('-')[1]))
    # exit()
    da = da.sel(lon = slice(int(lon_area.split('-')[0]), int(lon_area.split('-')[1])),              # Maritime Continenet domain
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))               #
                ).load()                                                                            #
    da_area = get_area_matrix(da.lat, da.lon).load()                                                # matrix to weight by gridbox area (to keep consistent with original calculation)

    # print(da_area)
    # exit()
    return da_area


def plot(map_sig, regression_coeff, da_c, beta, lon_area, idx_rect = 0): #, da_c):
    plt.rcParams['font.size'] = 7
    # -- create figure --    
    width, height = 8, 3                                                                                                        # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
    ax.remove()
    projection = ccrs.PlateCarree(central_longitude = 180)
    ax = fig.add_subplot(nrows, ncols, 1, projection=projection)

    # -- format ax --
    lat, lon = regression_coeff.lat, regression_coeff.lon
    lonm,latm = np.meshgrid(lon, lat)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    xticks = [60, 120, 180, 240, 300]
    yticks = [-20, 0, 20]
    scale_ax(ax, 1)
    move_row(ax, 0.225)     
    move_col(ax, 0)
    # plot_ticks(ax, xticks, yticks)
    ax.coastlines(resolution = "110m", linewidth = 0.6)

    # -- plot data --
    vmax = None
    vmin = None
    # vmax = 0.75
    # vmin = -0.75

    # satfrac_mlr
    # vmax = 0.5
    # vmin = -vmax

    # print(np.unique(map_sig))
    # exit()
    # vmax = 200
    # vmin = -200
    # if idx_rect == 7:
    #     vmax = 2000
    #     vmin = -2000

    # -- satfrac, af|Am --
    if beta == 0:
        # da_area = get_area_normalization_factor(lon_area).sum()
        # regression_coeff = (regression_coeff / da_area) * 1e6 # Convert units to [% / 100km^2]
        # vmax = 0.25
        # vmin = -0.25    
        if lon_area == '0-360':
            regression_coeff = regression_coeff / 7 # to keep the units [%] relative to the same area
            ''

        vmax = 30
        vmin = -30            
        # vmax = 5    # satfrac
        # vmin = -5   # satfrac
        
        
        # text = r'$\alpha \ [K \ / \ \%]$'
        # ax_position = ax.get_position()
        # ax.text(ax_position.x0 + 0.3,                                                                                              # x-start
        #         ax_position.y0 - 0.525,                                                                                              # y-start
        #         text,                        
        #         fontsize = 7,  
        #         transform = fig.transFigure,
        #         )
    
        # text = r'$(\delta CRH / \delta A_{tot})|_{A_m = c}$'
        # text = r'CRH = $\alpha A_{tot} + \beta A_{mean}$'
        # text = r'Ta$_{500hPa}$ = $\alpha A_{tot} + \beta A_{mean}$'
        # ax_position = ax.get_position()
        # ax.text(ax_position.x0,                                                                                              # x-start
        #         ax_position.y1 + 0.025,                                                                                              # y-start
        #         text,                        
        #         fontsize = 7,  
        #         transform = fig.transFigure,
        #         )

    # -- satfrac, Am|af --
    if beta == 1:
        if lon_area == '0-360':
            regression_coeff = regression_coeff / 7   
        vmax = 5000
        vmin = -5000
        # else:
        #     vmax = 2000
        #     vmin = -2000

        # vmax = 200      # satfrac
        # vmin = -200     # satfrac


        # text = r'$\beta \ [K \ / \ \%]$'
        # ax_position = ax.get_position()
        # ax.text(ax_position.x0 + 0.3,                                                                                              # x-start
        #         ax_position.y0 - 0.525,                                                                                              # y-start
        #         text,                        
        #         fontsize = 7,  
        #         transform = fig.transFigure,
        #         )
    
        # # text = r'$(\delta CRH / \delta A_{tot})|_{A_m = c}$'
        # text = r'CRH = $\alpha A_{tot} + \beta A_{mean}$'
        # ax_position = ax.get_position()
        # ax.text(ax_position.x0,                                                                                              # x-start
        #         ax_position.y1 + 0.025,                                                                                              # y-start
        #         text,                        
        #         fontsize = 7,  
        #         transform = fig.transFigure,
        #         )

    h = ax.pcolormesh(lonm, latm, regression_coeff.where(map_sig == 1), 
                          transform=ccrs.PlateCarree(), 
                          cmap = 'RdBu_r', 
                        #   cmap = 'RdBu_r', 
                          vmin = vmin, 
                          vmax = vmax
                          )
    
    # -- put significant cross --
    # y_indices, x_indices = np.where(map_sig)   
    # for x, y in zip(x_indices, y_indices):
    #     ax.plot(lon[x], lat[y], 'kx', transform=ccrs.PlateCarree(), markersize = 0.01) # 'kx' for black crosses

    # -- put contour --
    lat, lon = da_c.lat, da_c.lon
    lonm,latm = np.meshgrid(lon, lat)
    contours = ax.contour(lonm, latm, da_c, 
                        transform = ccrs.PlateCarree(),
                        levels =        [0],
                        colors =        'k', 
                        linewidths =    0.35)

    # -- plot colormap --
    # cbar_ax = cbar_ax_below(fig, ax, h)
    cbar_ax = ''
    add_rectangles(ax, idx_rect = idx_rect)

    return fig, ax, cbar_ax


# == main ==
def main():
    # -- x-metric --
    x_tfreq,  x_group,    x_name,     x_var,  x_label,    x_units =   'daily',    'doc_metrics',          'area_fraction',                        'area_fraction_thres_pr_percentiles_95',    r'A$_f$',   r''    

    # -- y-metric --
    # y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',          'i_org',                                'i_org',                                    r'Iorg',    r''    
    y_tfreq,  y_group,    y_name,     y_var,  y_label,    y_units =   'daily',    'doc_metrics',          'mean_area',                            'mean_areathres_pr_percentiles_95',         r'A$_m$',   r'km$^2$'    

    # -- z-metric --
    # z_tfreq,  z_group,    z_name,     z_var,  z_label,    z_units =   'daily',    'satfrac',               'satfrac_regress2',                         'satfrac_regress2',                r'CRH',      r'' 
    z_tfreq,  z_group,    z_name,     z_var,  z_label,    z_units =   'daily',    'ta',               'ta_regress2',                         'ta_regress2',                                   r'Ta$_{500hPa}$',      r'K'   


    # -- contour --
    # c_tfreq,  c_group,    c_name,     c_var,  c_label,    c_units =   'monthly',  'tas',                  'tas_map',                              'tas_map_mean',                             r'T',       r'$^o$C'          
    da_c = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/observations/ERA5/wap/wap_map/ERA5/wap_map_ERA5_monthly_0-360_-30-30_128x64_1998-01_2022-12.nc')['wap_map_mean']
    
    # == OBS metrics ==
    lon_area_x =  '0:360'                                                                                                                                                                             
    lat_area_y =  '-30:30'                                                                                                                                                                            
    res =       1.0
    time_period = '2001-01:2021-12'    
    lon_areas_x = (                                                                                                                                            
        '0:49',                     # Africa
        '50:99',                    # Indian Ocean
        '100:149',                  # Maritime Continent                                   
        '150:204',                  # West / central Pacific    (55 degrees, 5 degrees wider)
        '205:259',                  # East Pacific              (55 degrees, 5 degrees wider)
        '260:309',                  # Amazon
        '310:359',                  # Atlantic
        '0:360',                    # Tropics
        )    
    
    standardized = False
    regression_coeff_addition, regression_coeff_addition2 = [], []
    for ii, lon_area in enumerate(lon_areas_x):
        lon_area = f'{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'
        # -- MLR --
        # data_type_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
        # data_type_group, data_tyoe, dataset = 'observations', 'ISCCP', 'ISCCP'
        data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
        if standardized:
            metric_name_coeffs =   f'{z_name}_{x_name}_{y_name}_regress_standard_{lon_area}'
        else:
            metric_name_coeffs =   f'{z_name}_{x_name}_{y_name}_regress_{lon_area}'
        regression_coeffs = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area_x, lat_area_y, res, time_period, metric_name_coeffs)
        # print(regression_coeffs)
        # exit()

        if standardized:
            metric_name_pvalues =   f'{z_name}_{x_name}_{y_name}_pvalues_standard_{lon_area}'
        else:
            metric_name_pvalues =   f'{z_name}_{x_name}_{y_name}_pvalues_{lon_area}'
        map_pvalues = get_metric(data_type_group, data_tyoe, dataset, z_tfreq, z_group, z_name, lon_area_x, lat_area_y, res, time_period, metric_name_pvalues)
        # print(map_pvalues)
        # exit()

        # -- contour  --
        # data_type_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
        # data_type_group, data_tyoe, dataset = 'observations', 'ISCCP', 'ISCCP'
        # data_type_group, data_tyoe, dataset = 'observations', 'ERA5', 'ERA5'
        # data_type_group, data_tyoe, dataset = 'observations', 'NOAA', 'NOAA'
        # time_period = '1998-01:2022-12'    
        # da_c = get_metric(data_type_group, data_tyoe, dataset, c_tfreq, c_group, c_name, lon_area, lat_area, res, time_period, c_var)

        # print(regression_coeffs)
        # print(map_pvalues)
        # print(da_c)
        # exit()

        # -- plot --
        for i, (b, p) in enumerate(zip(regression_coeffs.beta, map_pvalues.beta)):
            regression_coeff = regression_coeffs.sel(beta = b)
            map_sig = map_pvalues.sel(beta = b)
            fig, ax, cbar_ax = plot(map_sig, regression_coeff, da_c, b, lon_area, idx_rect = ii) #, da_c) d

            # -- save figure --
            folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
            folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}/x_{x_name}_y_{y_name}_z_{z_name}'
            filename = f'coeff_{i}_domain_{lon_area}'
            path = f'{folder}/{filename}.png'
            # print(path)
            # exit()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi = 500)
            print(f'plot saved at: {path}')
            plt.close(fig)
            # break
            # exit()
            if b == 0:
                regression_coeff_addition.append(regression_coeff)
            else:
                regression_coeff_addition2.append(regression_coeff)

    # exit()
    # print(sum(regression_coeff_addition))
    # exit()
    # -- plot addition (af) --
    regression_coeff = sum(regression_coeff_addition)
    # print(regression_coeff)
    fig, ax, cbar_ax = plot(map_sig, regression_coeff, da_c, beta = 0, lon_area = lon_area, idx_rect = 'all') #, da_c) d

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}/x_{x_name}_y_{y_name}_z_{z_name}'
    filename = f'coeff_sum_af'
    path = f'{folder}/{filename}.png'
    # print(path)
    # exit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)
    
    
    # -- plot addition (af) --
    regression_coeff = sum(regression_coeff_addition2)
    # print(regression_coeff)
    fig, ax, cbar_ax = plot(map_sig, regression_coeff, da_c, beta = 1, lon_area = lon_area, idx_rect = 'all') #, da_c) d

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}/x_{x_name}_y_{y_name}_z_{z_name}'
    filename = f'coeff_sum_am'
    path = f'{folder}/{filename}.png'
    # print(path)
    # exit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)
    exit()

# == when this script is ran / global variables ==
if __name__ == '__main__':    
    # path = '/Users/cbla0002/Desktop/work/metrics/observations/ERA5/ta/ta_mlr/ERA5/ta_mlr_ERA5_daily_0-360_-30-30_360x180_2001-01_2021-12.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()
    main()




