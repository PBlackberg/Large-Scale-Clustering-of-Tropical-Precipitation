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
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from matplotlib import ticker

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
def get_metric(data_type_group, data_type, dataset, resolution, time_period, tfreq, group, name, var, lon_area, lat_area, extra):
    if extra:
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                       # user settings
        folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{group}_GPCP/{name}/{dataset}'                                        #     
        filename = (                                                                                                                    # base result_filename
                f'{name}'                                                                                                               #
                f'_{dataset}'                                                                                                           #
                f'_{tfreq}'                                                                                                             #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                                   #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                                   #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                                         #
                f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                             #
                f'_{extra}'
                )    
    else:
        filename = (                                                                                                                    # base result_filename
                f'{name}'                                                                                                               #
                f'_{dataset}'                                                                                                           #
                f'_{tfreq}'                                                                                                             #
                f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                                   #
                f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                                   #
                f'_{int(360/resolution)}x{int(180/resolution)}'                                                                         #
                f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                             #
                )       
    path = f'{folder}/{filename}.nc'
    ds = xr.open_dataset(path)
    return ds
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
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2 + 0.2,                                   # left
                            ax_position.y0 - 0.25,                                                                              # bottom
                            ax_position.width * w,                                                                              # width
                            ax_position.height * 0.1                                                                            # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation = 'horizontal')
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_size(7)
    return cbar_ax

def plot_cbar_label(fig, ax, text):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - 0.125, # + (ax_position.width * 0.), 
            ax_position.y0 - 0.275, 
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
        # '0:49',                                                                                                                   # Africa
        # '50:99',                                                                                                                  # Indian Ocean
        # '100:149',                                                                                                                # Maritime Continent
        # '150:204',                                                                                                                # West / central Pacific    (55 degrees, 5 degrees wider)
        # '205:259',                                                                                                                # East Pacific              (55 degrees, 5 degrees wider)
        # '260:309',                                                                                                                # Amazon
        # '310:359',                                                                                                                # Atlantic
        '0:359',        
        )               
    labels = (
        # 'Africa',                                                                              
        # 'Indian Ocean',                                                                                          
        # 'MTC',                                                                               
        # 'West Pacific',                                                                                                   
        # 'East Pacific',                                                                                                          
        # 'Amazon',                                                                                                              
        # 'Atlantic', 
        'Deep Tropics',                                                                                                              
        ) 

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
        
def plot(regression_coeff, map_pvalue, da_c, idx_rect = 0): #, da_c):
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
    plot_ticks(ax, xticks, yticks)
    ax.coastlines(resolution = "110m", linewidth = 0.6)

    # == plot data ==
    # -- limits --
    vmax = None
    vmin = None

    # mesosacle
    # vmax = 2e-3
    # vmin = -2e-3

    # largescale
    vmax = 1e-4
    vmin = -1e-4



    map_pvalue = map_pvalue * 0 + 1.
    h = ax.pcolormesh(lonm, latm, regression_coeff.where(map_pvalue == 1), 
                          transform=ccrs.PlateCarree(), 
                        #   cmap = 'BrBG',                           
                        #   cmap = 'RdBu', 
                        #   cmap = 'RdBu_r', 
                          cmap = 'coolwarm',
                          vmin = vmin, 
                          vmax = vmax
                          )

    # -- put contour --
    lat, lon = da_c.lat, da_c.lon
    lonm,latm = np.meshgrid(lon, lat)
    contours = ax.contour(lonm, latm, da_c, 
                        transform = ccrs.PlateCarree(),
                        levels =        [0],
                        colors =        'k', 
                        linewidths =    0.35)

    # -- plot colormap --
    cbar_ax = cbar_ax_below(fig, ax, h)
    # cbar_ax = ''
    # add_rectangles(ax, idx_rect = idx_rect)
    return fig, ax, cbar_ax


# == main ==
def main():
    # == Specify metrics ==
    # // IMERG //
    # == Tropical ==
    # -- radiative fluxes --

    # -- cloud-radiative effect --
    z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p_one', 'mlr_p_one_area_fraction_mean_area', r'obs_all_toa_net-obs_clr_toa_net', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p_one', 'mlr_p_one_area_fraction_mean_area', r'obs_all_toa_sw-obs_clr_toa_sw', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p_one', 'mlr_p_one_area_fraction_mean_area', r'obs_all_toa_lw-obs_clr_toa_lw', r'W m$^{-2}$' 

    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p_one', 'mlr_p_one_area_fraction_mean_area', r'obs_all_toa_lw', r'W m$^{-2}$' 


    # -- relative humidity --
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'ERA5', 'ERA5', 1.0, '2001-01:2021-12', 'daily', 'crh', 'mlr_p_one', 'mlr_p_one_area_fraction_mean_area', r'', r''   


    # == subdomain ==
    # -- radiative fluxes --
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'toa_sw_insol', r'W m$^{-2}$'    
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_net', r'W m$^{-2}$'    
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_clr_toa_net', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_sw', r'W m$^{-2}$'    
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_clr_toa_sw', r'W m$^{-2}$'    
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_lw', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_clr_toa_lw', r'W m$^{-2}$' 

    # -- cloud-radiative effect --
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_net-obs_clr_toa_net', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_sw-obs_clr_toa_sw', r'W m$^{-2}$' 
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'CERES', 'CERES', 1.0, '2001-01:2021-12', '3hrly', 'rad', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'obs_all_toa_lw-obs_clr_toa_lw', r'W m$^{-2}$' 

    # -- Relative humidity --
    # z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, z_label, z_units = 'observations', 'ERA5', 'ERA5', 1.0, '2001-01:2021-12', 'daily', 'crh', 'mlr_p', 'mlr_p_area_fraction_mean_area', r'', r''   




    # -- contour --
    da_c = xr.open_dataset('/Users/cbla0002/Desktop/work/metrics/observations/ERA5/wap/wap_map/ERA5/wap_map_ERA5_monthly_0-360_-30-30_128x64_1998-01_2022-12.nc')['wap_map_mean']

    # -- load metric --
    lon_area_z =  '0:360'                                                                                                                                                                             
    lat_area_z =  '-30:30'                                                                                                                                                                            
    ds = get_metric(z_data_type_group, z_data_type, z_dataset, z_resolution, z_time_period, z_tfreq, z_group, z_name, z_var, lon_area_z, lat_area_z, z_label)
    # print(ds)
    # exit()
    # '/Users/cbla0002/Desktop/work/metrics/observations/ERA5/crh/mlr_a/ERA5/mlr_a_ERA5_3hrly_0-360_-30-30_360x180_2001-01_2021-12.nc'
    # '/Users/cbla0002/Desktop/work/metrics/observations/ERA5/crh/mlr_a/ERA5/mlr_a_ERA5_daily_0-360_-30-30_360x180_2001-01_2021-12.nc'
    
    # regression_coeffs = ds[f'{z_var}_regress'] #* 100
    # map_pvalues = ds[f'{z_var}_pvalues']
    # print(ds)
    # print(regression_coeffs)
    # print(map_pvalues)
    # exit()
    
    # regression_coeffs = ds[f'{z_var}_regress_-13-13']
    # map_pvalues = ds[f'{z_var}_pvalues_-13-13']

    regression_coeffs = - ds[f'{z_var}_regress_-30-30']
    map_pvalues = ds[f'{z_var}_pvalues_-30-30']

    # -- variations --
    lon_areas_x = (                                                                                                                                            
        # '0:49',                     # Africa
        # '50:99',                    # Indian Ocean
        # '100:149',                  # Maritime Continent                                   
        # '150:204',                  # West / central Pacific    (55 degrees, 5 degrees wider)
        # '205:259',                  # East Pacific              (55 degrees, 5 degrees wider)
        # '260:309',                  # Amazon
        # '310:359',                  # Atlantic
        '0:360',                    # Tropics
        )    
    
    for ii, (b, lon_area) in enumerate(zip(regression_coeffs.beta, lon_areas_x)):
        lon_area = f'{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'
        # -- plot --
        # regression_coeff = regression_coeffs.sum("beta")
        # map_pvalue = map_pvalues.mean("beta")*0 + True


        regression_coeff = regression_coeffs.sel(beta = b)
        map_pvalue = map_pvalues.sel(beta = b)
        fig, ax, cbar_ax = plot(regression_coeff, map_pvalue, da_c, idx_rect = ii) #, da_c)

        # -- text --
        text = rf'$\delta$CRE / $\delta$am|af [{z_units} / km$^2$]'
        plot_cbar_label(fig, ax, text)

        # -- save figure --
        folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
        folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{Path(__file__).stem}/{z_var}'
        filename = f'{z_var}_domain_{lon_area}'
        path = f'{folder}/{filename}.png'
        # print(path)
        # exit()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi = 500)
        print(f'plot saved at: {path}')
        plt.close(fig)
        # exit()
    # exit()

# == when this script is ran / global variables ==
if __name__ == '__main__':    
    # path = '/Users/cbla0002/Desktop/work/metrics/observations/ERA5/crh/mlr_b/ERA5/mlr_b_ERA5_daily_0-360_-30-30_360x180_2001-01_2021-12.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()
    main()




    # '/Users/cbla0002/Desktop/work/metrics/observations/CERES/rad/mlr_p/CERES/mlr_p_CERES_3hrly_0-360_-30-30_360x180_2001-01_2021-12.nc'
    # '/Users/cbla0002/Desktop/work/metrics/observations/CERES/rad/mlr_p/CERES/mlr_p_CERES_3hrly_0-360_-30-30_360x180_2001-01_2021-12_toa_sw_insol.nc'



