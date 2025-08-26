'''
# ----------------------------
#  IISCCP cloud states data
# ----------------------------
Link to weather states data:
https://isccp.giss.nasa.gov/analysis/climanal5.html
I've downloaded it and put it in a folder. Change this line:
path_to_saved_data = f'/g/data/k10/cb4968/temp_saved/data/obs/weather_states/{str(year)}.nc'
to where you save it

'''

# == imports ==
# -- packages --
import xarray as xr
import numpy as np
import pandas as pd
import datetime

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_obs.conserv_interp       as cI


# == pre-process ==
def pre_process(ds, dataset, regrid_resolution, t_freq, var):
    # -- fix coordinates and pick out data --
    ds = ds.sortby('lat')
    da = ds[var]
    ds_frame = xr.open_dataset('/g/data/k10/cb4968/data/observations/tas/sst.mnmean.nc').sel(time = slice('1998-01', '2017-06'))        # something wrong with the coordinate for regridding on the original dataset, "generic coordinates". Replace the coordinates wiht this dataset (same resolution)
    ds_frame = ds_frame.sortby('lat')
    da_new = xr.DataArray(data=da.data, dims=['time', 'lat', 'lon'], coords={'time': ds_frame['time'], 'lat': ds_frame['lat'], 'lon': ds_frame['lon']})
    # -- regrid --
    da = cI.conservatively_interpolate(da_in =              da_new.load(), 
                                        res =               regrid_resolution, 
                                        switch_area =       None,                                                                       # regrids the whole globe
                                        simulation_id =     dataset
                                        )
    return da


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
        path_to_saved_data = f'/g/data/k10/cb4968/data/observations/weather_states/{str(year)}.nc'
        ds = xr.open_dataset(path_to_saved_data)
        da_year = process_year(ds, year, var)
        da_list.append(da_year)
    da = xr.concat(da_list, dim = 'time')
    ds = xr.Dataset({var: da})

    # -- pre-process --
    da = pre_process(ds, dataset, regrid_resolution = resolution, t_freq = t_freq, var = var)
    
    # -- custom process --
    da = process_data_further(da)

    return da

# == when this script is ran ==
if __name__ == '__main__':         
    var =           'cl_high'   # 'cl_low'
    dataset =       'ISCCP'
    t_freq =        'monthly'
    resolution =    2.8
    time_period =   '1998-01:2022-12'
    process_request = [var, dataset, t_freq, resolution, time_period]
    def process_data_further(da):
        ''
        return da
    da = get_data(process_request, process_data_further)
    print(da)
    exit()

