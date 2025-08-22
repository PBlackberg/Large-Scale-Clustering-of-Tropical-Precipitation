'''
# -----------------
#   detrend_anom
# -----------------

'''

# == imports ==
import xarray as xr
import numpy as np
from scipy import signal

# == main ==
def detrend_data(da):
    time_axis = da.get_axis_num('time')
    return xr.apply_ufunc(signal.detrend, da, kwargs={'axis': time_axis}, dask="parallelized", output_dtypes=[float])

def get_monthly_anomalies(da_detrended):
    da_detrended = da_detrended.resample(time='1MS').mean()
    climatology = da_detrended.groupby('time.month').mean('time')
    da_deseasoned = da_detrended.groupby('time.month') - climatology
    return da_deseasoned

def detrend_month_anom(da):
    da_detrended = detrend_data(da)
    da_detrend_anom = get_monthly_anomalies(da_detrended)
    return da_detrend_anom


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')




