'''
# -----------------
#   util_calc
# -----------------

'''

# == imports ==
import xarray as xr
import numpy as np
from scipy.signal import detrend


# == calc ==
# -- oni --
def oni_calc(da):
    ''' Taken from example '''
    da = da.resample(time='1MS').mean()
    sst_clim = da.groupby('time.month').mean(dim='time')
    sst_anom = da.groupby('time.month') - sst_clim
    time_axis = da.get_axis_num('time')
    sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom, kwargs={'axis': time_axis}).where(~sst_anom.isnull())    
    sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(-5, 5), lon=slice(190, 240))
    sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('lon', 'lat'))
    oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()
    return oni

# -- zonal gradient --
def zonal_gradient_calc(da):
    ''' Can do give da as rolling mean here too, but excluded to apply to metric in clim and with time coordinate '''
    lat_min, lat_max, lon_min, lon_max = -5, 5,  80, 150                                    # west pacific ocean
    da_w = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), 
                    np.nan)
    lat_min, lat_max, lon_min, lon_max = -5, 5, 180, 280                                    # east pacific ocean 
    da_e = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), np.nan)

    gradient = (da_w.mean(dim = ('lat', 'lon')) - da_e.mean(dim = ('lat', 'lon')))          # gradient
    return gradient

# -- meridional gradient --
def meridional_gradient_calc(da):
    ''' Can give da as rolling mean here too, but excluded to apply to metric in clim and with time coordinate '''
    # -- edge regions --
    lat_min, lat_max, lon_min, lon_max = 12.5, 35, 150, 250                                 # northern meridional section (pacific ocean)
    da_n = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), np.nan).copy()   
    lat_min, lat_max, lon_min, lon_max  = -35, -12.5, 150, 250                              # southern meridional section (pacific ocean)
    da_s = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), np.nan).copy()
    da_ns = da.where(da_n.notnull() | da_s.notnull())                                       # edge regions together
    # -- central region --
    lat_min, lat_max, lon_min, lon_max = -5, 5, 150, 250                                    # central meridional section (pacific ocean) 
    da_c = da.where((da.lat >= lat_min) & (da.lat <= lat_max) & 
                    (da.lon >= lon_min) & (da.lon <= lon_max), np.nan)
    # -- calculate gradient --
    gradient = da_c.mean(dim = ('lat', 'lon')) - da_ns.mean(dim = ('lat', 'lon'))
    return gradient

