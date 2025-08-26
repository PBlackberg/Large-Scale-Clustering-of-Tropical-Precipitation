'''
# -----------------
#   util_calc
# -----------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == calc ==
def get_area_matrix(lat, lon):
    lonm, latm = np.meshgrid(lon, lat)
    dlat = lat.diff(dim='lat').data[0]
    dlon = lon.diff(dim='lon').data[0]
    R = 6371     # km
    area =  np.cos(np.deg2rad(latm))*np.float64(dlon * dlat * R**2*(np.pi/180)**2) # area of domain: cos(lat) * (dlon * dlat) R^2 (area of gridbox decrease towards the pole as gridlines converge)
    da_area = xr.DataArray(data = area, dims = ["lat", "lon"], coords = {"lat": lat, "lon": lon}, name = "area")
    return da_area

def area_weight(da):
    da_area = get_area_matrix(da.lat, da.lon)
    da_area_weight = (da * da_area).sum() / da_area.sum()
    return da_area_weight


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')










