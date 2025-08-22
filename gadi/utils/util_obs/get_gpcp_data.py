''' 
# ---------------
#   GPCP_data
# ---------------
GPCP - Global Precipitation Climatology Project
https://climatedataguide.ucar.edu/climate-data/gpcp-monthly-global-precipitation-climatology-project

'''

# == imports ==
# -- packages --
import xarray as xr
import numpy as np

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_obs.conserv_interp       as cI


# == pre-process ==
def pre_process(da, dataset, regrid_resolution):
    da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    # -- deal with nan values --
    valid_range = [0, 10000]                                                
    da = da.where((da >= valid_range[0]) & (da <= valid_range[1]), np.nan)  # There are some e+33 values in the dataset
    da = da.dropna('time', how='all')                                       # One day all values are NaN
    da = da.where(da.sum(dim=['lat', 'lon']) != 0, drop=True)               # for a few days, all values are zero
    
    # -- remove duplicate days --                                           # this was surprisingly quick
    da_list = []
    for day in np.unique(da['time'].values):
        # print(day)
        da_day = da.sel(time = day)
        try:
            length = len(da_day.time)
            if length > 1:
                da_day = da_day.isel(time = 0)
                da_list.append(da_day)
        except:
            da_list.append(da_day)
    da = xr.concat(da_list, dim = 'time')

    # -- regrid --
    da = cI.conservatively_interpolate(da_in =              da.load(), 
                                        res =               regrid_resolution, 
                                        switch_area =       None,                      # regrids the whole globe for the moment 
                                        simulation_id =     dataset
                                        )
    return da


# == get data ==
def get_data(process_request, process_data_further):
    var, dataset, t_freq, resolution, time_period = process_request

    # -- get time_period --
    year1 = time_period.split(":")[0].split('-')[0]
    year2 = time_period.split(":")[1].split('-')[0]

    # -- get files --
    path_gen = '/g/data/ia39/aus-ref-clim-data-nci/gpcp/data/day/v1-3'
    years = range(int(year1), int(year2)+1)
    paths = [f'{path_gen}/gpcp_v1-3_day_{year}.nc' for year in years]
    # print(paths[0])

    # -- concatenate --
    ds = xr.open_mfdataset(paths, combine='by_coords', parallel = True)
    da = ds['precip'].load()

    # -- pre-process --
    da = pre_process(da, dataset, regrid_resolution = resolution)

    # -- custom process --
    da = process_data_further(da)
    
    return da # xr.Dataset(data_vars = {f'{var}': da}, attrs = ds.attrs)         


# == when this script is ran ==
if __name__ == '__main__':
    var =           'pr'
    dataset =       'gpcp'
    t_freq =        'daily'
    resolution =    2.8
    time_period =   '1998-01:2022-12'
    process_request = [var, dataset, t_freq, resolution, time_period]
    def process_data_further(da):
        ''
        return da

    da = get_data(process_request, process_data_further)    
    print(da)
    exit()





    # ds = xr.open_dataset(paths[0])
    # print(ds)
    # exit()