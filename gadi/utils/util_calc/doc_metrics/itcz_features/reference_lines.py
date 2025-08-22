'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == reference lines ==
def meridional_line(domain, lon_centre):
    diff =                  abs(domain.lon - lon_centre)
    closest_reference_lon = domain.lon.isel(lon = diff.argmin())                                                    # data might not have exactly lon_centre, so pick closest
    array_line =            xr.DataArray(np.full((domain.lat.size,), closest_reference_lon),                        #
                                           dims = "lat", coords = {"lat": domain.lat.data})                         #
    ds_line =               xr.Dataset({'var': array_line})                                                         # for plots                                                                           
    da_line =               xr.ones_like(domain).where((domain.lon == closest_reference_lon), 0)                    # for metric calc  
    return ds_line, da_line

def zonal_line(domain, lat_centre):
    diff =                  abs(domain.lat - lat_centre)
    closest_reference_lat = domain.lat.isel(lat = diff.argmin())
    array_line =            xr.DataArray(np.full((domain.lon.size,), closest_reference_lat), 
                                           dims="lon", coords={"lon": domain.lon.data})
    ds_line =               xr.Dataset({'var': array_line})
    da_line =               xr.ones_like(domain).where((domain.lat == closest_reference_lat), 0)
    return ds_line, da_line

def hyrdo_eq_line(domain, hus):
    max_lat_per_lon =   hus.idxmax(dim='lat')                                                                       # Hydrological equator as latitude of max specific humidity as a function of longitude
    ds_line =           xr.Dataset({'var': max_lat_per_lon})
    if not 'time' in hus.dims:
        da_line = xr.ones_like(domain).where(domain.lat == max_lat_per_lon, 0)
    else:
        da_line_list = []
        for month_idx, time_month in enumerate(hus['time']):                                                                
            da_line_t = xr.ones_like(domain).where(domain.lat == max_lat_per_lon.isel(time = month_idx), 0)
            da_line_list.append(da_line_t.assign_coords(time = time_month))
        da_line = xr.concat(da_line_list, dim='time')
    return ds_line, da_line

def hydro_eq_median_line(domain, hus):
    max_lat_per_lon =       hus.idxmax(dim='lat')                                                                   # Hydrological equator as latitude of max specific humidity as a function of longitude
    median_lat =            max_lat_per_lon.median(dim='lon')                                                       # median of those latitudes (constant with longitude)
    if not 'time' in hus.dims:
        da_line = xr.ones_like(domain).where(domain.lat == max_lat_per_lon, 0)
    else:
        ds_line_list, da_line_list = [], []
        for month_idx, time_month in enumerate(hus['time']):        
            diff =                  abs(domain.lat - median_lat.isel(time = month_idx))
            closest_reference_lat = domain.lat.isel(lat = diff.argmin())
            array_line =            xr.DataArray(np.full((domain.lon.size,), closest_reference_lat), 
                                                dims="lon", coords={"lon": domain.lon.data})
            ds_line_list.append(xr.Dataset({'var': array_line}).assign_coords(time = time_month))
            da_line_list.append(xr.ones_like(domain).where((domain.lat == closest_reference_lat), 0).assign_coords(time = time_month))
        ds_line = xr.concat(ds_line_list, dim='time')
        da_line = xr.concat(da_line_list, dim='time')
    return ds_line, da_line, median_lat

def centre_point(domain, lon_centre, lat_centre):
    array_line =    xr.DataArray(np.full((1,), lat_centre), dims="lon", coords={"lon": [lon_centre]})
    ds_line =       xr.Dataset({'var': array_line})
    da_line =       xr.ones_like(domain).where((domain.lat == domain.sel(lat=lat_centre, method="nearest").lat) & 
                                         (domain.lon == domain.sel(lon=lon_centre, method="nearest").lon), 0)
    return ds_line, da_line


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')







