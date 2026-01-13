'''
# ---------------------
#  Correlation matrix
# ---------------------

'''

# == imports ==
# -- Packages --
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from scipy.stats import pearsonr
import pandas as pd

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
mS = import_relative_module('user_specs',                                           'utils')
cW = import_relative_module('util_calc.area_weighting.globe_area_weight',           'utils')
mlR = import_relative_module('util_calc.multiple_linear_regression.mlr_calc',                   'utils')

# == metric funcs ==
# for year in np.arange(int(year_start), int(year_end) + 1):
#     if year > 2021:
#         continue
def open_metric(data_type_group, data_type, dataset, resolution, time_period,
                t_freq, metric_group, metric_name, metric_var, lat_area, lon_area,
                single_value = True 
                ):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    r_filename = (
            f'{metric_name}'   
            f'_{dataset}'                                                                                                   
            f'_{t_freq}'                                                                                                    
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
            )       
    folder_metric = f'{folder}/{r_filename}'    # fiels are stored in years here
    year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
    # -- metrics saved in months or years, respectively --
    paths = []
    if metric_name in ['L_org']: # stored in months
        for year in np.arange(int(year_start), int(year_end) + 1):
            if year > 2021:
                continue
            for month in np.arange(1, 13):
                path = f'{folder_metric}/{r_filename}_{year}_{month}-{year}_{month}.nc'
                paths.append(path)
    else:                       # stored in years
        for year in np.arange(int(year_start), int(year_end) + 1):
            if year > 2021:
                continue
            path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
            paths.append(path)

    if metric_name == 'L_org':
        try:
            ds = xr.open_mfdataset(paths, combine='by_coords').load()         
        except:
            time = pd.date_range("2001-01-01", "2023-12-31", freq="3h")
            an_array = (np.zeros((1, len(time))) * np.nan).squeeze()
            metric = xr.DataArray(an_array, coords={"time": time}, dims=["time"])
            # print(metric)
            return metric
    else:    
        ds = xr.open_mfdataset(paths, combine='by_coords').load()
    # -- calculate single value metric (per timestep) --
    if single_value:
        if metric_var == 'i_org':
            metric = np.trapezoid(ds['i_org_obs_thres_pr_percentiles_95'], ds['i_org_random_thres_pr_percentiles_95'])
            metric = xr.DataArray(metric, coords={"time": ds.time}, dims=["time"])
        elif metric_var == 'N_100':
            L_obs = ds['L_org_obs_thres_pr_percentiles_95']
            da = xr.open_dataset('/Users/cbla0002/Desktop/work/data/IMERG_data/i_org_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2023-12_var_2001_1_1.nc')['var'].isel(time = 0)
            # print(da)
            # exit()
            da_area_lat = da['lat'].sel(lat = slice(-30, 30)).load()    # for area weighting later
            da_area_lon = da['lon'].sel(lon = slice(0, 360)).load()
            da_area = cW.get_area_matrix(da_area_lat, da_area_lon)   
            da_area = da_area.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                                    lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                                )
            
            # remove area fraction weighting
            paths = []
            for year in np.arange(int(year_start), int(year_end) + 1):
                if year > 2021:
                    continue
                a_str =f'{lon_area.split(":")[0]}-{lon_area.split(":")[1]}_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'
                path = f'/Users/cbla0002/Desktop/work/metrics/observations/IMERG/doc_metrics/area_fraction/IMERG/area_fraction_IMERG_3hrly_{a_str}_3600x1800_2001-01_2023-12/area_fraction_IMERG_3hrly_{a_str}_3600x1800_2001-01_2023-12_{year}_1-{year}_12.nc'
                paths.append(path)
            x = xr.open_mfdataset(paths, combine='by_coords').load()['area_fraction_thres_pr_percentiles_95']        
            # print(x)
            # exit()

            y = np.pi * (L_obs**2 / da_area.sum()) * (x * (len(da_area['lat']) * len(da_area['lon']))) # * 100 # (frraction of points as percentage) #  # dividing by the number of "cores" (when considering all convective points). This represents the mean fraction of points within a radius

            # print(y)
            # exit()
            r_bin_lim = 100 # characteristic bin
            y = y.sel(r_bin_edges = r_bin_lim, method='nearest')
            metric = xr.DataArray(y, coords={"time": ds.time}, dims=["time"])
            # print(metric)
            # exit()
        elif metric_var == 'L_org':
            L_obs = ds['L_org_obs_thres_pr_percentiles_95']
            L_random = ds['L_org_random_thres_pr_percentiles_95']
            r_bin_edges = ds['r_bin_edges']
            metric = np.trapezoid(L_obs - L_random, x = r_bin_edges)
            metric = xr.DataArray(metric, coords={"time": ds.time}, dims=["time"])

        elif metric_name == 'mean_area':
            da = xr.open_dataset('/Users/cbla0002/Desktop/work/data/IMERG_data/i_org_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2023-12_var_2001_1_1.nc')['var'].isel(time = 0)
            # print(da)
            # exit()
            da_area_lat = da['lat'].sel(lat = slice(-30, 30)).load()    # for area weighting later
            da_area_lon = da['lon'].sel(lon = slice(0, 360)).load()
            da_area = cW.get_area_matrix(da_area_lat, da_area_lon)   
            da_area = da_area.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                                    lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                                )
            metric = ds[metric_var] / da_area.sum()
        else:
            metric = ds[metric_var]
    return metric

def open_metric2(data_type_group, data_type, dataset, resolution, time_period,
                t_freq, metric_group, metric_name, metric_var, lat_area, lon_area,
                single_value = True 
                ):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)                        # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_type_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    r_filename = (
            f'{metric_name}'   
            f'_{dataset}'                                                                                                   
            f'_{t_freq}'                                                                                                    
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                           
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                           
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                                 
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                     
            )       
    folder_metric = f'{folder}/{r_filename}'    # fiels are stored in years here
    year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
    # -- metrics saved in months or years, respectively --
    paths = []
    if metric_var in ['L_org']: # stored in months
        for year in np.arange(int(year_start), int(year_end) + 1):
            if year > 2021:
                continue
            for month in np.arange(1, 13):
                path = f'{folder_metric}/{r_filename}_{year}_{month}-{year}_{month}.nc'
                paths.append(path)
    else:                       # stored in years
        for year in np.arange(int(year_start), int(year_end) + 1):
            path = f'{folder_metric}/{r_filename}_{year}_1-{year}_12.nc'
            paths.append(path)
    ds = xr.open_mfdataset(paths, combine='by_coords').load()
    # -- calculate single value metric (per timestep) --
    if single_value:
        if metric_var == 'i_org':
            metric = np.trapezoid(ds['i_org_obs_thres_percentile_95'], ds['i_org_random_thres_percentile_95'])
            metric = xr.DataArray(metric, coords={"time": ds.time}, dims=["time"])
        elif metric_var == 'L_org':
            L_obs = ds['L_org_obs_thres_percentile_95']
            L_random = ds['L_org_random_thres_percentile_95']
            r_bin_edges = ds['r_bin_edges']
            metric = np.trapezoid(L_obs - L_random, x = r_bin_edges)
            metric = xr.DataArray(metric, coords={"time": ds.time}, dims=["time"])
        elif metric_name == 'mean_area':
            da = xr.open_dataset('/Users/cbla0002/Desktop/work/data/GRIDSAT_data/2001/GRIDSAT-B1.2001.01.01.00.v02r01.nc')['irwin_cdr'].isel(time = 0) #['var'].isel(time = 0)
            # print(da)
            # exit()
            da_area_lat = da['lat'].sel(lat = slice(-30, 30)).load()    # for area weighting later
            da_area_lon = da['lon'].sel(lon = slice(0, 360)).load()
            da_area = cW.get_area_matrix(da_area_lat, da_area_lon)   
            da_area = da_area.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                                    lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                                )
            metric = ds[metric_var] / da_area.sum()
            # print(metric)
            # exit()
        else:
            metric = ds[metric_var]
    return metric

def ds_to_da(ds):
    data_arrays = [ds[var] for var in ds.data_vars]
    da_models = xr.concat(data_arrays, dim = 'variable')
    return da_models

def remove_season_from_daily(da):
    da = da.resample(time='1D').mean()                                                                      # remove diurnal variability
    da_smooth = da.rolling(time=7, center=True, min_periods=1).mean()                                       # smooth timescale of weather patterns
    anomalies = da_smooth.groupby("time.dayofyear") - da_smooth.groupby("time.dayofyear").mean("time")      # remove seasonal cycle
    return anomalies

def get_correlation_matrix(ds, time_period, lat_area, lon_area, response):
    ''' Correlates row_i with all the other rows in the dataset (each row is a separate metric) '''
    # -- identify metrics --
    metric_names = list(ds.data_vars.keys())
    num_metrics = len(metric_names)
    data_array = ds.to_array() #.values

    # -- create empty numpy array to fill --
    correlation_matrix = np.zeros((num_metrics, num_metrics))
    p_value_matrix = np.zeros((num_metrics, num_metrics))

    # -- fill metrix with correlations --
    for i in range(num_metrics):
        for j in range(num_metrics):
            if np.all(np.isnan(data_array[i, :])) or np.all(np.isnan(data_array[j, :])):
                correlation_matrix[i, j] = np.nan
                p_value_matrix[i, j] = np.nan
            else:

                # # remove part of the metric that is explained by the area fraction
                # year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
                # paths = []
                # for year in np.arange(int(year_start), int(year_end) + 1):
                #     if year > 2021:
                #         continue
                #     a_str =f'{lon_area.split(":")[0]}-{lon_area.split(":")[1]}_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'
                #     path = f'/Users/cbla0002/Desktop/work/metrics/observations/IMERG/doc_metrics/area_fraction/IMERG/area_fraction_IMERG_3hrly_{a_str}_3600x1800_2001-01_2023-12/area_fraction_IMERG_3hrly_{a_str}_3600x1800_2001-01_2023-12_{year}_1-{year}_12.nc'
                #     paths.append(path)
                # af = xr.open_mfdataset(paths, combine='by_coords').load()['area_fraction_thres_pr_percentiles_95']        

                # year_start, year_end = time_period.split(":")[0].split('-')[0], time_period.split(":")[1].split('-')[0]
                # paths = []
                # for year in np.arange(int(year_start), int(year_end) + 1):
                #     if year > 2021:
                #         continue
                #     a_str =f'{lon_area.split(":")[0]}-{lon_area.split(":")[1]}_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'
                #     path = f'/Users/cbla0002/Desktop/work/metrics/observations/GRIDSAT/doc_metrics/area_fraction/GRIDSAT/area_fraction_GRIDSAT_3hrly_{a_str}_5142x2571_2001-01_2023-12/area_fraction_GRIDSAT_3hrly_{a_str}_5142x2571_2001-01_2023-12_{year}_1-{year}_12.nc'
                #     paths.append(path)
                # # print(xr.open_mfdataset(paths, combine='by_coords'))
                # # exit()
                # af2 = xr.open_mfdataset(paths, combine='by_coords').load()['area_fraction_thres_percentile_95']        

                # -- pre-process --
                xy_list = [data_array[i, :], data_array[j, :], response]
                xy_list = [remove_season_from_daily(da) for da in xy_list]
                xy_list = [da.dropna(dim='time', how='any') for da in xy_list]
                xy_list = list(xr.align(*xy_list, join='inner'))
                xy_list = [da.data for da in xy_list]

                if len(xy_list[0]) < 2 or np.isnan(xy_list[0]).all() or np.isnan(xy_list[1]).all():
                    y = [np.nan, np.nan]
                else:
                    try:
                        y = mlR.get_linear_model_components(xy_list[:-1], xy_list[-1], show = False, standardized = True)[1]    # y_hat, coeffs, residual
                    except:
                        y = [np.nan, np.nan]
                correlation_matrix[i, j] = y[0]
                p_value_matrix[i, j] = 0.0005


    # -- give dimensions and coords --
    coords = {'metric1': metric_names, 'metric2': metric_names}  
    da_correlation_matrix  = xr.DataArray(correlation_matrix,  dims = ('metric1', 'metric2'), coords = coords)
    da_p_value_matrix      = xr.DataArray(p_value_matrix,      dims = ('metric1', 'metric2'), coords = coords)        
  
    return da_correlation_matrix, da_p_value_matrix

def get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix):
    # -- mean and spread --
    da_corr     = ds_to_da(ds_corr_matrix)
    corr_mean   = da_corr.mean(dim = 'variable')
    corr_spread = da_corr.std(dim='variable')
    # q25 = da_corr.quantile(0.25, dim='variable')
    # q75 = da_corr.quantile(0.75, dim='variable')
    # -- nb of models --
    nb_models   = xr.where(np.isnan(da_corr), 0, int(1)).sum(dim = 'variable').data
    # -- nb of significant --
    da_p        = ds_to_da(ds_p_value_matrix)
    p_threshold = 0.05
    da_p_sig = xr.where(da_p < p_threshold, int(1), 0)
    p_count = da_p_sig.sum(dim = 'variable').data
    return corr_mean, corr_spread, p_count, nb_models


# == plot funcs ==
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

def cbar_ax_right(fig, ax, h):
    ax_position = ax.get_position()
    cbar_ax = fig.add_axes([ax_position.x1 + 0.0125,                                                                                # left
                            ax_position.y0 + (ax_position.height - ax_position.height * 0.9) / 2,                                   # bottom
                            ax_position.width * 0.025,                                                                              # width
                            ax_position.height * 0.9                                                                                # height
                            ])      
    cbar = fig.colorbar(h, cax = cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize = 7)
    return cbar_ax


# == plot ==
def plot(correlation_matrix, p_value_matrix):
    labels = correlation_matrix['metric1'].data
    p_threshold = 0.05
    plt.rcParams['font.size'] = 7

    # -- create figure --    
    width, height = 10.5, 8                                                                                                                             # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                                                 # convert to inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))

    # -- format ax --
    scale_ax(ax, 1)
    move_row(ax, -0.1)     
    move_col(ax, 0)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # -- plot colormap --
    # mask = np.tril(np.ones_like(da_corr_mean, dtype=bool), k = -1)                                                                                      # Specify upper right triangle
    # masked_corr = np.where(mask, np.nan, da_corr_mean)                                                                                                  # apply mask
    # h = ax.imshow(masked_corr, cmap = 'RdBu', vmin = - 1, vmax = 1)                  
    h = ax.imshow(correlation_matrix, cmap = 'RdBu', vmin = - 1, vmax = 1)    

    # -- add correlation values and labels --
    text_size = 6
    for i in range(len(labels)):                                                                                                                        # metric i
        # for j in range(i, len(labels)):                                                                                                                 # metric j (Only lower triangle + diagonal (j ≤ i))
        for j in range(len(labels)):
            corr_value = correlation_matrix[i, j].data
            p_value = p_value_matrix[i, j]
            if p_value > p_threshold:                                                                                               # put statistically insignificant in ()
                text = f"({corr_value:.2f})"                        
            else:
                text = f"{corr_value:.2f}"                             
            ax.text(j, i, text, ha='center', va='center', color='black' if abs(corr_value) < 0.5 else 'white', fontsize = text_size - 2)                # plot text in color to show contrast with colormap
            # if i >= j:
            ax.text(j - 0.2, - 0.65, labels[j], ha='left', va='bottom', fontsize = text_size, rotation = 45)                                        # top label
            ax.text(j, - 0.5, '-', ha='center', va='bottom', fontweight = 'bold', fontsize = text_size + 0.5, rotation = 90)                        # dash from plot
                # pass
        ax.text(- 0.8, i, labels[i], ha='right', va='center', fontsize = text_size)                                                                   # side label
        ax.text(- 0.5, i, '-', ha='right', va='center', fontweight = 'bold', fontsize = text_size + 0.5)                                              # dash from plot
        # ax.text(i - 0.45, i, '-', ha='right', va='center', fontweight = 'bold', fontsize=fontsize + 0.5)                                              # dash from plot

    # cbar_ax = cbar_ax_below(fig, ax, h)
    cbar_ax = cbar_ax_right(fig, ax, h)

    # -- title --
    # ax_position = ax.get_position()
    # ax.text(ax_position.x0 + 0.1,                                                                                                                     # x-start
    #         ax_position.y1 + 0.145,                                                                                                                   # y-start
    #         title,                        
    #         fontsize = 10,  
    #         transform=fig.transFigure,
    #         )

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'correlation_matrix'
    path = f'{folder}/{plot_name}.png'
    # print(path)
    # exit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == main ==
def main():
    # -- settings common to all, that can be changed --
    lat_area = '-13:13'
    # lat_area =  '-30:30'      
    lon_areas = (                             
        '0:360',                      # Deep tropics                                                                                                     
        # '0:49',                         # Africa
        # '50:99',                        # Indian Ocean
        # '100:149',                      # Maritime Continent                                   
        # '150:204',                      # West / central Pacific    (55 degrees, 5 degrees wider)
        # '205:259',                      # East Pacific              (55 degrees, 5 degrees wider)
        # '260:309',                      # Amazon
        # '310:359',                      # Atlantic
        )    
    # lon_area = lon_areas[0]
    labels = (
        'Deep tropics',            
        # 'Africa',                                                                  
        # 'Indian Ocean',                                                                                          
        # 'MTC',                                                                               
        # 'West Pacific',                                                                                                   
        # 'East Pacific',                                                                                                          
        # 'Amazon',                                                                                                              
        # 'Atlantic',                                                                                                               
        ) 

    # -- load metrics --
    ds_corr_matrix, ds_p_value_matrix = xr.Dataset(), xr.Dataset()
    time_period = '2001-01:2023-12'

    for idx, (lon_area, label) in enumerate(zip(lon_areas, labels)):
        ds = xr.Dataset()
        # == PREDICTORS ==
        # IMERG
        data_type_group, data_type, dataset =   'observations', 'IMERG', 'IMERG'  
        resolution = 0.1
        x1_tfreq,   x1_group,     x1_name,     x1_var,  x1_label,    x1_units =   '3hrly',      'doc_metrics',      'L_org',                        'N_100',                                    r'N100km[I]',       r''   
        x2_tfreq,   x2_group,     x2_name,     x2_var,  x2_label,    x2_units =   '3hrly',      'doc_metrics',      'area_fraction',                'area_fraction_thres_pr_percentiles_95',    r'Af[I]',           r''   
        x3_tfreq,   x3_group,     x3_name,     x3_var,  x3_label,    x3_units =   '3hrly',      'doc_metrics',      'mean_area',                    'mean_area_thres_pr_percentiles_95',        r'Am[I]',           r''   
        # x4_tfreq,   x4_group,     x4_name,     x4_var,  x4_label,    x4_units =   '3hrly',      'doc_metrics',      'Ncores',                       'Ncores_obs_thres_pr_percentiles_95',       r'Ncores[I]',       r''   
        x5_tfreq,   x5_group,     x5_name,     x5_var,  x5_label,    x5_units =   '3hrly',      'doc_metrics',      'i_org',                        'i_org',                                    r'Iorg[I]',         r''   
        x6_tfreq,   x6_group,     x6_name,     x6_var,  x6_label,    x6_units =   '3hrly',      'doc_metrics',      'perimeter',                    'perimeter_thres_pr_percentiles_95',        r'PM[I]',           r''   
        x7_tfreq,   x7_group,     x7_name,     x7_var,  x7_label,    x7_units =   '3hrly',      'doc_metrics',      'L_org',                        'L_org',                                    r'Lorg[I]',         r''   

        ds[f'{x1_label} {x1_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x1_tfreq, x1_group, x1_name, x1_var, lat_area, lon_area)
        ds[f'{x2_label} {x2_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x2_tfreq, x2_group, x2_name, x2_var, lat_area, lon_area)
        ds[f'{x3_label} {x3_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x3_tfreq, x3_group, x3_name, x3_var, lat_area, lon_area)
        # ds[f'{x4_label} {x4_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x4_tfreq, x4_group, x4_name, x4_var, lat_area, lon_area)
        ds[f'{x5_label} {x5_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x5_tfreq, x5_group, x5_name, x5_var, lat_area, lon_area)
        ds[f'{x6_label} {x6_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x6_tfreq, x6_group, x6_name, x6_var, lat_area, lon_area)
        ds[f'{x7_label} {x7_units}'] =      open_metric(data_type_group, data_type, dataset, resolution, time_period, x7_tfreq, x7_group, x7_name, x7_var, lat_area, lon_area)

        # GRIDSAT
        data_type_group, data_type, dataset =   'observations', 'GRIDSAT', 'GRIDSAT'  
        resolution = 0.07
        x8_tfreq,   x8_group,   x8_name,    x8_var,     x8_label,   x8_units =  '3hrly',        'doc_metrics',      'area_fraction',                'area_fraction_thres_percentile_95',        r'Af[G]',           r''   
        x9_tfreq,   x9_group,   x9_name,    x9_var,     x9_label,   x9_units =  '3hrly',        'doc_metrics',      'mean_area',                    'mean_area_thres_percentile_95',            r'Am[G]',           r'' 
        x10_tfreq,  x10_group,  x10_name,   x10_var,    x10_label,  x10_units = '3hrly',        'doc_metrics',      'Ncores',                       'Ncores_thres_percentile_95',               r'Ncores[G]',       r''    
        x11_tfreq,  x11_group,  x11_name,   x11_var,    x11_label,  x11_units = '3hrly',        'doc_metrics',      'i_org',                        'i_org',                                    r'Iorg[G]',         r''    
        x12_tfreq,  x12_group,  x12_name,   x12_var,    x12_label,  x12_units = '3hrly',        'doc_metrics',      'perimeter',                    'perimeter_thres_percentile_95',            r'PM[G]',           r''   

        ds[f'{x8_label} {x8_units}'] =      open_metric2(data_type_group, data_type, dataset, resolution, time_period, x8_tfreq, x8_group, x8_name, x8_var, lat_area, lon_area)
        ds[f'{x9_label} {x9_units}'] =      open_metric2(data_type_group, data_type, dataset, resolution, time_period, x9_tfreq, x9_group, x9_name, x9_var, lat_area, lon_area)
        ds[f'{x10_label} {x10_units}'] =    open_metric2(data_type_group, data_type, dataset, resolution, time_period, x10_tfreq, x10_group, x10_name, x10_var, lat_area, lon_area)
        ds[f'{x11_label} {x11_units}'] =    open_metric2(data_type_group, data_type, dataset, resolution, time_period, x11_tfreq, x11_group, x11_name, x11_var, lat_area, lon_area)
        ds[f'{x12_label} {x12_units}'] =    open_metric2(data_type_group, data_type, dataset, resolution, time_period, x12_tfreq, x12_group, x12_name, x12_var, lat_area, lon_area)


        # == RESPONSE VARIABLE ==
        data_type_group, data_type, dataset = 'observations', 'ERA5', 'ERA5'
        resolution = 1.    
        lat_area = '-30:30'
        # z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'satfrac',      'satfrac_timeseries',       'satfrac_timeseries',                                               r'RH',       r'%'    
        z_tfreq,   z_group,   z_name,    z_var, z_label,   z_units =  'daily',      'ta',      'ta_timeseries',       'ta_timeseries',                                               r'Ta500hpa',       r'K'    
        response = open_metric(data_type_group, data_type, dataset, resolution, time_period, z_tfreq, z_group, z_name, z_var, lat_area, lon_area)
        lat_area = '-13:13'

        # == CORRELATE ==
        da_corr_matrix, da_p_value_matrix = get_correlation_matrix(ds, time_period, lat_area, lon_area, response) 
        # print(da_corr_matrix, da_p_value_matrix)
        # exit()

        # -- add to xr.dataset --
        # ds_corr_matrix[lon_area] =     da_corr_matrix
        # ds_p_value_matrix[lon_area] =  da_p_value_matrix
        # print(ds_corr_matrix)
        # print(ds_p_value_matrix)
        # exit()



        # -- model mean, spread, and significance --
        # [print(f) for f in [da_corr_mean, da_corr_spread, p_count, nb_domains]]
        # exit()
        # da_corr_mean, da_corr_spread, p_count, nb_domains = get_model_ensemble_stats(ds_corr_matrix, ds_p_value_matrix)

        # -- plot --
        plot(da_corr_matrix, da_p_value_matrix)



# == when this script is ran ==
if __name__ == '__main__':
    # path = '/Users/cbla0002/Desktop/work/metrics/observations/GRIDSAT/doc_metrics/perimeter/GRIDSAT/perimeter_GRIDSAT_3hrly_0-49_-13-13_5142x2571_2001-01_2023-12/perimeter_GRIDSAT_3hrly_0-49_-13-13_5142x2571_2001-01_2023-12_2001_1-2001_12.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()
    main()






