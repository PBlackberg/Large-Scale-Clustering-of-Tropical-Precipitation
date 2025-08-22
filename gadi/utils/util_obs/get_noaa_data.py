''' 
# ---------------
#   NOAA_data
# ---------------
NOAA - National Oceanic and Atmopsheric Administration
https://psl.noaa.gov/data/gridded/
https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/
https://downloads.psl.noaa.gov/Datasets/gistemp/combined/250km/

'''

# == imports ==
# -- packages --
import xarray as xr

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_obs.conserv_interp       as cI


# == get raw data ==
def get_raw_data():
    ''' Download from website '''
    path_gen = '/g/data/k10/cb4968/data/observations/tas/sst.mnmean.nc' # '/g/data/k10/cb4968/sample_data/tas/obs/sst.mnmean.nc'
    return path_gen


# == pre-process ==
def pre_process(ds, dataset, regrid_resolution):
    # -- fix coordinates and pick out data --
    ds = ds.sortby('lat')
    da = ds['tas']

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
    path_gen = get_raw_data()

    # -- concatenate --
    ds = xr.open_dataset(path_gen)
    ds = ds.sel(time = slice(year1, year2))
    # print(ds)
    # exit()

    # -- pre-process --
    da = pre_process(ds, dataset, regrid_resolution = resolution)
    
    # -- custom process --
    da = process_data_further(da)

    return da # xr.Dataset(data_vars = {f'{var}': da}, attrs = ds.attrs)        


# == when this script is ran ==
if __name__ == '__main__':
    # ds = xr.open_dataset('/g/data/k10/cb4968/data/observations/tas/sst.mnmean.nc')
    # print(ds)
    # exit()
    var =           'tas'
    dataset =       'NOAA'
    t_freq =        'monthly'
    resolution =    2.8
    time_period =   '1998-01:2022-12'
    process_request = [var, dataset, t_freq, resolution, time_period]
    def process_data_further(da):
        ''
        return da
    ds = get_data(process_request, process_data_further)

    print(ds)
    exit()
