'''
# ----------------
#  get_IFS_data
# ----------------

'''

# == imports ==
# -- Packages --
import os
import sys
import intake
import numpy as np
import xarray as xr
from datetime import datetime

# -- Imported scripts --
sys.path.insert(0, os.getcwd())
import utils.util_ifs.conservative_interp         as crG


def get_ds(year = '2025', da_dim = '2d'):
    ''' Gets all variables '''
    print('no ds_dask')
    print('getting ds_dask from catalogue ..')
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    if int(year) < 2020:
        cat = cat.IFS["IFS_9-FESOM_5-production-hist"]                                                      # IFS: current
    else:
        cat = cat.IFS["IFS_9-FESOM_5-production"]                                                           # IFS: warm
    if not da_dim == '2d':                                                                                  # 3D variables stored decades at a time
        entry = cat["3D_monthly_0.25deg"] 
    else:
        entry = cat["2D_hourly_0.25deg"]                                                                    # ERA5 resolution (0.25 deg, hourly)
    ds = entry.to_dask()                                                                                    # Convert the entry to an xarray dataset
    return ds

def process_data(ds, process_request):    
    print('processing data')
    var_name, switch_area, switch_process, switch_res, switch_time = process_request   
    da = ds[var_name]

    # -- pick section (day, month, or year of data) --
    if switch_time.get('day', False):
        time_str = f"{switch_time['year']}-{switch_time['month']:02d}-{switch_time['day']:02d}"      
        da = da.sel(time=slice(f"{time_str}T00:00:00", f"{time_str}T23:59:59")).load()
    elif switch_time.get('month', False):
        time_str = f"{switch_time['year']}-{switch_time['month']:02d}" 
        da = da.sel(time=slice(f"{time_str}", f"{time_str}")).load()
    elif switch_time.get('year', False):
        time_str = f"{switch_time['year']}" 
        da = da.sel(time=slice(f"{time_str}", f"{time_str}")).load()
    else:
        print('pick a time period to load')
        print('exiting')
        exit()

    # -- resample (day, month) --
    if switch_process.get('resample_daily', False):
        da = da.resample(time='1D').mean()
    elif switch_process.get('resample_monthly', False):
        da = da.resample(time='1MS').mean()
    else:
        print('pick temporal sample space')
        print('exiting')
        exit()

    # -- regrid --
    if switch_process.get('latlon_grid', False):
        da = da.set_index(value=("lat", "lon")).unstack("value")                                            # give lat lon coordinates
    if switch_process.get('regridded', False):
        if switch_res['lon_res'] == 0.25:                                                                   # this version of the model is already on latlon at this resolution
            da = da.assign_coords(lon=((da.lon + 360) % 360))
            da = da.sortby('lon')
            pass
        else:
            da = crG.conservatively_interpolate(da, switch_res, switch_area, simulation_id = 'IFS_9_FESOM_5')
    
    # -- pick area --
    if switch_area.get('lat_area', False) or switch_area.get('lon_area', False):
       da = da.sel( lat = slice(switch_area['lat_area'][0], switch_area['lat_area'][1]), 
                    lon = slice(switch_area['lon_area'][0], switch_area['lon_area'][1]))
    return da


def get_data(ds_dask, year, process_request, dask_adapted, temp_data_path, process_data_further, da_dim = '2d'):
    '''
    If dask adapted one month of data is saved at a time, from which each worker can load data
    (can also be good for testing, as full dask_array doesn't need to be regenerated)
    '''
    # -- if saving progressively -- 
    if dask_adapted:
        if os.path.exists(temp_data_path):
            return ds_dask, None                                                                            # no need to process if already saved
        else:
            if ds_dask is None:                                                                             # generating this dask array only happens once per job
                ds_dask = get_ds(year, da_dim)
            da = process_data(ds_dask, process_request)
            da = process_data_further(da)
            xr.Dataset({'var': da}).to_netcdf(temp_data_path, mode="w") 
            return ds_dask, None                                                                            # data is read from file later so no need to return da
    # -- if not saving progressively -- 
    else:
        if ds_dask is None:                                                                                 # generating this dask array only happens once per job
            ds_dask = get_ds(year, da_dim)
        da = process_data(ds_dask, process_request)
        da = process_data_further(da)
        return ds_dask, da                                                                                  # returning da after processing

def get_valid_days(year, month):
    valid_days = []
    for day in range(1, 32):
        try:
            datetime(int(year), int(month), int(day))                                                       # Check if the date is valid
            valid_days.append(day)
        except ValueError:
            pass
    return valid_days

def get_timesections(n_jobs, time_period):
    year1, month1 = map(int, time_period.split(':')[0].split('-'))
    year2, month2 = map(int, time_period.split(':')[1].split('-'))
    timesteps = [(year, month) for year in range(int(year1), int(year2) + 1) 
                    for month in range(1, 13) 
                    if not (year == year1 and month < month1) and not (year == year2 and month > month2)]   # year, month pair
    time_sections = np.array_split(timesteps, n_jobs)
    return time_sections


def main():
    # -- specify data request --
    switch_var = {
        'tp':   False,                                                                                      # total precipitation
        'cc':   True,                                                               
        }
    switch_process = {
        'resample_daily':       False,   'resample_monthly': True,
        'latlon_grid':          True,
        'regridded':            True,
        }
    switch_res = {   # basic latlon grid resolution is 0.1 degrees
        'lon_res':  0.25,
        'lat_res':  0.25,
        # 'lon_res':  1,
        # 'lat_res':  1,
        # 'lon_res':  2.8,
        # 'lat_res':  2.8,
        }
    switch_area = {
        'lat_area':      [-30, 30],
        'lon_area':      [0, 360]
        }
    switch_time = {
        'year':     2025,
        'month':    3,
        # 'day':      22,    
        }

    print(f'''-- Running {__file__} --
Input:
var_name:       {next((key for key, value in switch_var.items() if value), None)}
dataset:        IFS_9_FESOM
''')
    [print(f"{key}:\t{value}") for key, value in switch_process.items() if value]
    [print(f"{key}:\t{value}") for key, value in switch_res.items()]
    [print(f"{key}:     \t{value}") for key, value in switch_area.items()]
    [print(f"{key}:     \t{value}") for key, value in switch_time.items()]

    # -- get variable --
    var_name = next((key for key, value in switch_var.items() if value), None)
    ds = get_ds(da_dim = '3d')
    process_request = [var_name, switch_area, switch_process, switch_res, switch_time]
    da = process_data(ds, process_request)
    print(da)


if __name__ == '__main__':
    main()







