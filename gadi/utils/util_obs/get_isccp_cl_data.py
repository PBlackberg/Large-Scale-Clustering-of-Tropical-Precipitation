'''
# ----------------------------
#  IISCCP cloud states data
# ----------------------------
Link to weather states data:
https://isccp.giss.nasa.gov/analysis/climanal5.html

'''

# == imports ==
# -- packages --
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -- imported scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
import utils.util_obs.conserv_interp       as cI
def import_relative_module(module_name, file_path):
    ''' import module from relative path '''
    if file_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        cwd = os.getcwd()
        relative_path = os.path.relpath(file_path, cwd) # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
pf_M = import_relative_module('helper_funcs.plot_func_map',                         __file__)


# == pre-process ==
def pre_process(ds, dataset, regrid_resolution, t_freq, var):
    # -- fix coordinates and pick out data --
    ds = ds.sortby('lat')
    da = ds[var]
    ds_frame = xr.open_dataset('/g/data/k10/cb4968/data/observations/tas/sst.mnmean.nc').sel(time = slice('1998-01', '2017-06'))        # something wrong wiht the coordinate on the original dataset, "generic coordinates". Replace the coordinates wiht this dataset (same resolution)
    ds_frame = ds_frame.sortby('lat')
    da_new = xr.DataArray(data=da.data, dims=['time', 'lat', 'lon'], coords={'time': ds_frame['time'], 'lat': ds_frame['lat'], 'lon': ds_frame['lon']})

    # -- regrid --
    da = cI.conservatively_interpolate(da_in =              da_new.load(), 
                                        res =               regrid_resolution, 
                                        switch_area =       None,                      # regrids the whole globe for the moment 
                                        simulation_id =     dataset
                                        )
    return da


# == plotting ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def move_col(ax, moveby):
    ax_position = ax.get_position()
    _, bottom, width, height = ax_position.bounds
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

        # # figure size
        # width, height = 6.27, 9.69                      # max size (for 1 inch margins)
        # width, height = width * 0.65, 0.45 * height     # modulate size and subplot distribution
        # ncols, nrows  = 1, 1
        # fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
        # plt.rcParams.update({'font.size': 12})
        # ax.imshow(da)

        # # ax size
        # scale_ax(ax, scaleby = 0.9)
        # move_col(ax, moveby = 0.1)
        # move_row(ax, moveby = 0.05)

        # # ylabel
        # ax.set_ylabel('Pressure [mb]', labelpad = 10)
        # y_tick_labels = np.array([1000, 800, 680, 560, 440, 310, 180, 50])
        # y_tick_labels = y_tick_labels[:-1] + (y_tick_labels[1:] - y_tick_labels[:-1]) / 2
        # y_tick_labels = y_tick_labels[::-1]
        # y_ticks = np.arange(0, len(y_tick_labels))
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_tick_labels)

        # # xlabel
        # ax.set_xlabel('cloud optical depth []', labelpad = 10)
        # x_tick_labels = np.array([0, 1.3, 3.6, 9.4, 23, 60, 379])
        # x_tick_labels = x_tick_labels[:-1] + (x_tick_labels[1:] - x_tick_labels[:-1]) / 2
        # x_ticks = np.arange(0, len(x_tick_labels))
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_tick_labels, fontsize = 10)

        # # title
        # title = (
        #     f'TCC: {np.sum(da):.1f} %\n'
        #     f'HCF: {da_high:.1f} %\n'
        #     f'LCF: {da_low:.1f} %\n'
        #     )
        # ax.text(
        #     0.02,
        #     0.85, 
        #     title,
        #     fontsize = 10,
        #     transform=fig.transFigure,
        #     fontweight = 'bold'
        #     )
        # ax.text(
        #     0.02,
        #     0.0225, 
        #     f'colorbar \nunits [%]',
        #     fontsize = 10,
        #     transform=fig.transFigure,
        #     fontweight = 'bold'
        #     )
        # plt.title(f'{ws_string}', fontweight = 'bold')

        # # row, col = np.unravel_index(np.argmax(da), da.shape)
        # # val = da[row, col]
        # # ax.text(col, row, f'{val:.1f}', ha='center', va='center', color='black',
        # #         path_effects=[pe.withStroke(linewidth=1, foreground='white')], fontsize = 10)
        # # print(da)
        # # print(np.shape(da))
        # # print(da.sum())
        # # exit()
        # for ii in range(da.shape[0]):
        #     for j in range(da.shape[1]):
        #         val = da[ii, j]
        #         ax.text(j, ii, f'{val:.1f}', ha='center', va='center', color='k',
        #                 path_effects=[pe.withStroke(linewidth=1, foreground='white')], fontsize = 9.5)

        # # save figure
        # folder = '/home/565/cb4968/gadi/utils/util_obs/plots'
        # filename = f'ws_{i+1}.png'
        # path = f'{folder}/{filename}'
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # os.remove(path) if os.path.exists(path) else None
        # fig.savefig(path)
        # print(f'plot saved at: {path}')
        # plt.close(fig)
        # exit()
    # exit()
    # print(ds_low_conversion)
    # print(ds_high_conversion)
    # exit()
            # # -- plot --
            # plot = True
            # if plot and t < 8:
            #     plot_here = xr.DataArray(data = np.rot90(xr.where(scene_day.data > 0, 1, np.nan).data), dims=['lat', 'lon'], coords={'lat': np.arange(90, -90, -1), 'lon': scene_month.longitude.data})
            #     folder = f'/home/565/cb4968/gadi/utils/util_obs/plots/snapshots'
            #     filename = f'snapshot_{t}.png'
            #     path = f'{folder}/{filename}'
            #     title = f'day: {counter}, slice: {t}'
            #     pf_M.plot(plot_here.sel(lat = slice(30, -30)), path, title = title)
            #     # exit()

# == process year ==
def process_year(ds, year, cloud_type = 'cl_high'):
    # -- find cloud fraction conversion from weather state --
    ws = ds['ws'].data
    ds_low_conversion = xr.Dataset()
    ds_high_conversion = xr.Dataset()
    for i in range(10):
        ws_string = f'ws_{i + 1}'
        da = ws[i,:].reshape(6,7).T *100                                                                                                # this is following the instructions here: https://isccp.giss.nasa.gov/outgoing/HGGWS/README.txt
        da_high = np.sum(da[0:3, :])                                                                                                    # cloud fraction of high-clouds (above 400hpa)
        da_low = np.sum(da[-3:, :])                                                                                                     # cloud fraction of low-clouds (below 600hpa)
        ds_low_conversion[f'{ws_string}'] = da_low
        ds_high_conversion[f'{ws_string}'] = da_high
    # -- convert cloud states on 3hr slice to daily values and accumulate cloud fraction for each gridbox in the month --
    da_list = []
    for i, month in enumerate(ds.data_vars):
        if month == 'ws':                                                                                                               # first variable is the states, so skip that one
            continue    
        da_month = ds[month]                                                                                                            # has dims (3hr_slice, lon, lat)
        nb_days = len(da_month[da_month.dims[0]]) / 8        
        slice_dict = {da_month.dims[0]: 0}
        scene_month = xr.zeros_like(da_month.isel(**slice_dict)).astype('float64')                                                      # initialize monthly cloud fraction, such that it can store float later when it is multiplied with a float
        scene_day = xr.zeros_like(da_month.isel(**slice_dict))                                                                          # initialize daily scene, built from 3hr polar orbiting scans
        counter = 0
        for t in da_month[da_month.dims[0]].data:
            counter += 1
            slice_dict = {da_month.dims[0]: t}
            scene_slice = da_month.isel(**slice_dict)                                                                                   # get 3hr scan
            scene_day = scene_day.where(scene_day > 0, scene_slice)                                                                     # only add where there is not already a value, or a missing value has been added
            if counter == 8:                                                                                                            # a day has been accumulated (removing overlap of scans) (using first scan value if next scan is overlapping) 
                for ws in np.arange(1, 11):                                                                                             # interpret associated cloud fraction from the weather state
                    if cloud_type == 'cl_high':
                        scene_month += xr.where(scene_day == ws, 1, 0) * ds_high_conversion[f'ws_{ws}'].data                            # convert the cloud state to cloud fraction
                    else:
                        scene_month += xr.where(scene_day == ws, 1, 0) * ds_low_conversion[f'ws_{ws}'].data
                scene_day = scene_day * 0                                                                                               # reset the daily scene
                counter = 0                                                              
        # -- monthly mean cloud fraction --
        scene_month = scene_month / nb_days
        scene_month = xr.DataArray(data = np.rot90(scene_month.data), dims=['lat', 'lon'], coords={'lat': np.arange(90, -90, -1), 'lon': scene_month.longitude.data})
        time_dt = datetime.datetime.strptime(f'{year}-{(i):02d}-01', '%Y-%m-%d').date()
        scene_month = scene_month.expand_dims(dim = 'time')
        scene_month = scene_month.assign_coords(time = pd.to_datetime([time_dt]))
        da_list.append(scene_month)
    da_year = xr.concat(da_list, dim = 'time')
    return da_year


# == get data ==
def get_data(process_request, process_data_further):
    var, dataset, t_freq, resolution, time_period = process_request

    # -- get time_period --
    time_period =   '1998-01:2022-12'
    year1 = time_period.split(":")[0].split('-')[0]
    year2 = time_period.split(":")[1].split('-')[0]
    years = np.arange(int(year1), int(year2))

    da_list = []
    for year in years:
        if year > 2017:
            continue     # no data past this
        ds = xr.open_dataset(f'/g/data/k10/cb4968/temp_saved/data/obs/weather_states/{str(year)}.nc')
        da_year = process_year(ds, year, var)
        da_list.append(da_year)
    da = xr.concat(da_list, dim = 'time')
    ds = xr.Dataset({var: da})

    # -- pre-process --
    da = pre_process(ds, dataset, regrid_resolution = resolution, t_freq = t_freq, var = var)
    
    # -- custom process --
    da = process_data_further(da)

    return da # xr.Dataset(data_vars = {f'{var}': da}, attrs = ds.attrs)        

# == when this script is ran ==
if __name__ == '__main__':
    # var =           'cl_low'
    var =           'cl_high'
    dataset =       'ISCCP'
    # t_freq =        'daily'
    t_freq =        'monthly'
    resolution =    2.8
    time_period =   '1998-01:2022-12'
    # ws_list = [5, 7, 8, 9, 10] 
    # ws_list = [1, 2, 3, 4, 5, 6]
    # ws_list = [1, 2, 3]
    # var = f"ws_{'_'.join(map(str, ws_list))}"
    process_request = [var, dataset, t_freq, resolution, time_period]
    def process_data_further(da):
        ''
        return da
    da = get_data(process_request, process_data_further)
    print(da)
    exit()





        