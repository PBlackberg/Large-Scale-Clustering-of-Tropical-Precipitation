'''
# -----------------
#   detrend_anom
# -----------------

'''

# == imports ==
import xarray as xr
from scipy import signal
import numpy as np


# == calc ==
def detrend_data(da):
    time_axis = da.get_axis_num('time')
    try:
        da_detrended = xr.apply_ufunc(signal.detrend, da, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])
    except ValueError as e:                                                                                                                                                                     # if fails due to inf or NaN
        if len(da.dims) > 1 and 'lat' in da.dims and 'lon' in da.dims:                                                                                                                          # most cases have lat lon grid, if not, custom treatment needed
            da_detrended = []                                                                                                                                                                   # detrend each dim
            for lat_idx in range(da.sizes['lat']):                                                                                                                                              #
                da_detrended_lat = []                                                                                                                                                           #
                for lon_idx in range(da.sizes['lon']):                                                                                                                                          #
                    da_timeseries = da.isel(lat=lat_idx, lon=lon_idx)                                                                                                                           # one timeseries at a time
                    try:                                                                                                                                                                        #
                        da_timeseries_detrended = xr.apply_ufunc(signal.detrend, da_timeseries, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])                         # Most cases will work normally
                    except ValueError as e:                                                                                                                                                     # Some will have the NaN, inf issue
                        if da_timeseries.isnull().all():                                                                                                                                        #
                            da_timeseries_detrended = da_timeseries                                                                                                                             # if all are NaN just return it as is
                        else:                                                                                                                                                                   #
                            try:                                                                                                                                                                #
                                da_timeseries_interp = da_timeseries.interpolate_na(dim='time', method='linear')                                                                                # otherwise try to interpolate over the NaN
                                da_timeseries_detrended = xr.apply_ufunc(signal.detrend, da_timeseries_interp, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])          # 
                            except ValueError as e:                                                                                                                                             # if that doesn't work, replace the NaN with zero
                                try:                                                                                                                                                            #
                                    da_timeseries_detrended = xr.apply_ufunc(signal.detrend, da_timeseries.fillna(0), kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])   #   
                                except ValueError as e:                                                                                                                                         #
                                    da_timeseries_detrended = da_timeseries * np.nan                                                                                                            # last resort, just make all NaN 
                    da_timeseries_detrended = da_timeseries
                    da_timeseries_detrended = da_timeseries_detrended.expand_dims({'lon': [da_timeseries.lon.item()], 'lat': [da_timeseries.lat.item()]})
                    da_detrended_lat.append(da_timeseries_detrended)
                da_detrended_lat = xr.concat(da_detrended_lat, dim = 'lon')    
                da_detrended.append(da_detrended_lat)
            da_detrended = xr.concat(da_detrended, dim = 'lat')    
            da_detrended = da_detrended.transpose('time', 'lat', 'lon')                                                                                                                         # make sure the dims are in the right order
        else:                                                                                                                                                                                   # if not lat, lon, time and doesn't work. Treat separately
            print('something went wron whenn detrending')
            print('check the data array.')
            print('here it is:')
            print(da)
            print('exiting')
            exit()
    return da_detrended

def get_monthly_anomalies(da_detrended):
    da_detrended = da_detrended.resample(time='1MS').mean()
    climatology = da_detrended.groupby('time.month').mean('time')
    da_deseasoned = da_detrended.groupby('time.month') - climatology
    return da_deseasoned


# == main ==
def detrend_month_anom(da):
    da_detrended = detrend_data(da)
    da_detrend_anom = get_monthly_anomalies(da_detrended)
    return da_detrend_anom


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')






# def detrend_data(da):
#     time_axis = da.get_axis_num('time')
#     try:
#         da_detrended = xr.apply_ufunc(signal.detrend, da, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])
#     except ValueError as e:                                                 # cannot detrend with inf or NaN
#         if da.isnull().all():                                               # if all values are NaN, just return the original
#             return da                                                       #
#         da = da.interpolate_na(dim='time', method='linear')                 # interpolate the NaN to detrend 
#         da_detrended = xr.apply_ufunc(signal.detrend, da, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])
#     return da_detrended

