''' 
# ------------------------
#   Cl_vert_interp
# ------------------------
This script interpolates cloud fraction data from the original grid (on hybrid-sigma pressure coordinates) to pressure levels on a coarser grid 
It is saved for easy access later.
So, change this line:
path_to_saved_data = f'/g/data/k10/cb4968/data1/cmip/cl/cmip6/{dataset}_cl_{timescale}_{experiment}_{int(360/resolution)}x{int(180/resolution)}.nc'
to where you want to save it
'''

# -- Packages --
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

# -- util- and local scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
import utils.util_cmip.get_cmip_data        as gD


# == interpolation ==
def interp_z_to_p(switch, da, z, z_new): 
    ''' From hybrid-sigma pressure levels to height coordinates [m], later converted to pressure [hPa] 
    da    - xarray data array (dim = (time, lev, lat, lon))
    z     - xarray data array (dim = (lev, lat, lon))
    z_new - numpy array       (dim = (lev))
    '''
    print(f'Column cloud fraction value before interp:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = 5).values} \n at heights [m]: \n {z.isel(lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}') if switch['show_one_column'] else None
    interpolated_data = np.empty((len(da['time']), len(z_new), len(da['lat']), len(da['lon'])))
    for time_i in range(len(da['time'])):
        print(f'month:{time_i} started')
        for lat_i in range(len(da['lat'])):
            for lon_i in range(len(da['lon'])):
                z_1d = z.sel(lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values if not 'time' in z.dims else z.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                var_1d = da.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                f = interp1d(z_1d, var_1d, kind='linear', bounds_error=False, fill_value=0)     
                var_1d_interp = f(z_new)                                                                          
                interpolated_data[time_i, :, lat_i, lon_i] = var_1d_interp
    da_z_fixed = xr.DataArray(interpolated_data, dims=('time', 'lev', 'lat', 'lon'), coords={'time': da['time'], 'lev': z_new, 'lat': da['lat'], 'lon': da['lon']})
    da_z_fixed['lev'] = 101325*(1-((2.25577e-5)*da_z_fixed['lev']))**(5.25588)
    da_z_fixed = da_z_fixed.rename({'lev':'plev'})
    print(f'Column cloud fraction value after interp to pressure coordinates:\n {da_z_fixed.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp heights [hPa]: \n {da_z_fixed.plev.values}') if switch['show_one_column'] else None
    return da_z_fixed

def interp_p_to_p_new(switch, da, p, p_new): 
    ''' From pressure levels to common pressure levels (when pressure has same dimensions as variable) 
    da    - xarray data array (dim = (time, plev, lat, lon))
    p     - xarray data array (dim = (lev, time, lat, lon))
    p_new - numpy array       (dim = (plev))
    '''
    print(f'Column cloud fraction value before interp to new pressure coordinates:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at pressure [hPa]: \n {p.isel(lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}') if switch['show_one_column'] else None
    interpolated_data = np.empty((len(da['time']), len(p_new), len(da['lat']), len(da['lon'])))
    for time_i in range(len(da['time'])):
        print(f'month:{time_i} started')
        for lat_i in range(len(da['lat'])):
            for lon_i in range(len(da['lon'])):
                p_1d = p.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                var_1d = da.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                f = interp1d(p_1d, var_1d, kind='linear', bounds_error=False, fill_value=0)     
                var_1d_interp = f(p_new)                                                                          
                interpolated_data[time_i, :, lat_i, lon_i] = var_1d_interp                                       
    da_p_new = xr.DataArray(interpolated_data, dims=('time', 'plev', 'lat', 'lon'), coords={'time': da['time'], 'plev': p_new, 'lat': da['lat'], 'lon': da['lon']})
    print(f'Column cloud fraction value after interp to new pressure coordinates:\n {da_p_new.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da_p_new.plev.values}') if switch['show_one_column'] else None
    return da_p_new

def interp_p_to_p_new_xr(switch, da, p_new):                                                                                                                                            # does the same thing as with scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (when pressure is 1D) 
    da    - xarray data array (dim = (time, plev, lat, lon))
    p_new - numpy array       (dim = (plev))
    '''
    print(f'Column cloud fraction value before interp to new pressure coordinates:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da.plev.values}') if switch['show_one_column'] else None
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})                                                                                   # warnings.resetwarnings() # the decpreciation warnings come from this function
    print(f'Column cloud fraction value after interp to new pressure coordinates:\n {da_p_new.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da_p_new.plev.values}') if switch['show_one_column'] else None
    return da_p_new


# == Get raw data ==
def get_p_hybrid(ds, dataset):
    if dataset == 'IITM-ESM':               
        da = ds['plev']
    elif dataset == 'IPSL-CM6A-LR':         
        da = ds['presnivs']
    elif dataset in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']: 
        da = ds['ap'] + ds['b']*ds['ps']     
    elif dataset in ['FGOALS-g2', 'FGOALS-g3']:  
        da = ds['ptop'] + ds['lev']*(ds['ps']-ds['ptop'])
    elif dataset in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:
        da = ds['lev']+ds['b']*ds['orog']
    else:
        da = ds['a']*ds['p0'] + ds['b']*ds['ps']
    return da


# == main funcs ==
def save_interp_data(switch, dataset, source, experiment, ds, timescale, resolution):
    if switch['save']:
        path_to_saved_data = f'/g/data/k10/cb4968/data/cmip/cl/cmip6/{dataset}_cl_{timescale}_{experiment}_{int(360/resolution)}x{int(180/resolution)}.nc'
        os.makedirs(os.path.dirname(path_to_saved_data), exist_ok=True)
        ds.to_netcdf(path_to_saved_data, mode="w")
        print(f'saved interpolated {dataset}-{experiment} clouds at:\n{path_to_saved_data}')

def run_interp(switch, dataset, source, experiment, da, p_hybrid, z_new, p_new, timescale, resolution):
    if dataset == 'IITM-ESM':               
        pass                                                                                                                                                                            # already on pressure levels (19 levels)
    elif dataset == 'IPSL-CM6A-LR':                                                                                                                                                     #
        da = da.rename({'presnivs':'plev'})                                                                                                                                             #
        da = interp_p_to_p_new_xr(switch, da, p_new)                                                                                                                                    # already on pressure levels (79 levels)
    elif dataset in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']:                   #       
        da = interp_p_to_p_new(switch, da, p_hybrid, p_new)                                                                                                                             # same shape as variable
    elif dataset in ['FGOALS-g2', 'FGOALS-g3']:                                                                                                                                         #
        da = da.rename({'lev':'plev'})         
        da = interp_p_to_p_new_xr(switch, da, p_new)                                                                                                                                    # same shape as variable (no longer true, so use 1d interp)
    elif dataset in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:                                                                                         #
        da = interp_z_to_p(switch, da, z = p_hybrid, z_new = z_new)                                                                                                                     # Some are 3D with no time dependence
        da = interp_p_to_p_new_xr(switch, da, p_new)                                                                                                                                    #
    else:                                                                                                                                                                               #
        da = interp_p_to_p_new(switch, da, p_hybrid, p_new)                                                                                                                             # same shape as variable
    save_interp_data(switch, dataset, source, experiment, xr.Dataset(data_vars = {'cl': da}), timescale, resolution)                                                                    #

def process_ds_cl(da):
    ''
    return da

def run_experiment(switch, dataset, source, timescale, resolution):
    experiments = [
        'historical',
        'ssp585'
        ]
    for experiment in experiments:
        print(f'\t\t {experiment}')        
        time_period = '1970-01:1999-12' if experiment == 'historical' else '2070-01:2099-12' 
        process_request = ['ds_cl', dataset, timescale, resolution, time_period]
        ds = gD.get_data(process_request, process_data_further = process_ds_cl)
        # print(ds)
        p_hybrid = get_p_hybrid(ds, dataset)
        z_new =    np.linspace(0, 15000, 30)                                                                                                                                            # [m]
        p_new =    np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])                              # [hPa]
        da = ds['cl']
        # print(da)
        run_interp(switch, dataset, source, experiment, da, p_hybrid, z_new, p_new, timescale, resolution)

def run_dataset(switch, timescale, resolution):
    print(f'Vertically regridding cmip cloudfraction data')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    datasets = (                                                                                                                                                                        # Models ordered by change in temperature with warming    
        # 'INM-CM5-0',                                                                                                                                                                  # 1 # no cloud
        'IITM-ESM',                                                                                                                                                                     # 2   
        # 'FGOALS-g3',                                                                                                                                                                    # 3    # this one has changed since generation
        # 'INM-CM4-8',                                                                                                                                                                  # 4                                
        'MIROC6',                                                                                                                                                                       # 5                                      
        'MPI-ESM1-2-LR',                                                                                                                                                                # 6                         
        # 'KIOST-ESM',                                                                                                                                                                  # 7
        'BCC-CSM2-MR',                                                                                                                                                                  # 8           
        # 'GFDL-ESM4',                                                                                                                                                                  # 9         
        'MIROC-ES2L',                                                                                                                                                                   # 10 
        'NorESM2-LM',                                                                                                                                                                   # 11      
        # 'NorESM2-MM',                                                                                                                                                                 # 12                      
        'MRI-ESM2-0',                                                                                                                                                                   # 13                            
        'GFDL-CM4',                                                                                                                                                                     # 14      
        'CMCC-CM2-SR5',                                                                                                                                                                 # 15                
        'CMCC-ESM2',                                                                                                                                                                    # 16                                    
        'NESM3',                                                                                                                                                                        # 17     
        'ACCESS-ESM1-5',                                                                                                                                                                # 18 
        'CNRM-ESM2-1',                                                                                                                                                                  # 19 
        # 'EC-Earth3',                                                                                                                                                                  # 20 
        'CNRM-CM6-1',                                                                                                                                                                   # 21
        # 'CNRM-CM6-1-HR',                                                                                                                                                              # 22
        'KACE-1-0-G',                                                                                                                                                                   # 23            
        'IPSL-CM6A-LR',                                                                                                                                                                 # 24
        'ACCESS-CM2',                                                                                                                                                                   # 25 
        'TaiESM1',                                                                                                                                                                      # 26                      
        'CESM2-WACCM',                                                                                                                                                                  # 27   
        'CanESM5',                                                                                                                                                                      # 28  
        'UKESM1-0-LL',                                                                                                                                                                  # 29
        ) 
    for dataset in datasets:
        source = 'cmip'
        print(f'\t{dataset} ({source})')
        run_experiment(switch, dataset, source, timescale, resolution)


# == run ==
if __name__ == '__main__':
    # path = '/g/data/k10/cb4968/data1/cmip/cl/cmip6/IITM-ESM_cl_monthly_historical_128x64.nc'
    # ds = xr.open_dataset(path)
    # print(ds)
    # exit()

    timescale = 'monthly'
    resolution = 2.8
    switch = {
            'show_one_column': True,                                                                                                                                                    # example of what interpolation does to cloud fraction in one column
            'save':            False                                                                                                                                                    # save interpolated data
            }
    run_dataset(switch, timescale, resolution)


