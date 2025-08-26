'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == general distance funcs ==
def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula)
    input: 
    lon range: [-180, 180]
    lat range: [-90, 90]
    (Takes vectorized input) 

    Formula:
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 
    where 
    phi -latitutde
    lambda - longitude
    '''
    R = 6371                                                                                    # radius of earth in km
    lat1 = np.deg2rad(lat1)                                                                     # function requires degrees in radians 
    lon1 = np.deg2rad(lon1-180)                                                                 # and lon in range [-180, 180]
    lat2 = np.deg2rad(lat2)                                                                     #
    lon2 = np.deg2rad(lon2-180)                                                                 #
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2     # Haversine formula
    h = np.clip(h, 0, 1)                                                                        # float point precision sometimes give error
    result =  2 * R * np.arcsin(np.sqrt(h))                                                     # formula rearranged for spherical distance
    return result


# == shortest distance from any gridbox to every other point in the domain ==
def create_distance_matrix(lat, lon):         
    ''' This matrix will have the size (len(lat) * len(lon), len(lat), len(lon))
    For coarse cmip6 model: (2816, 22, 128) '''                               
    lonm, latm = np.meshgrid(lon, lat)
    latm3d = np.expand_dims(latm, axis=2)                                                       # used for broadcasting
    lonm3d = np.expand_dims(lonm, axis=2)                                                       # used for broadcasting

    I, J = zip(*np.argwhere(np.ones_like(latm)))                                                # All spatial coordinates (i, j)
    I, J = list(I), list(J)                                                                     # index structured as (lat1, lon1), (lat1, lon2) ..
    n = len(I)                                                                                  # n = number of gridboxes
    
    lati3d = np.tile(lat[I], reps =[len(lat), len(lon), 1])                                     # Each gridbox lat replicated as a 2D field, resulting size: (n, lat, lon) 
    loni3d = np.tile(lon[J], reps =[len(lat), len(lon), 1])                                     # Each gridbox lon replicated as a 2D field
    latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, n])                                           # Each latitude in domain replicated n times 
    lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, n])                                           # Each longitude in domain replicated n times

    distance_matrix = haversine_dist(lati3d, loni3d, latm3d, lonm3d)                            # distance from gridbox i to every other point in the domain
    return xr.DataArray(distance_matrix, 
                        dims = ['lat', 'lon', 'gridbox'], 
                        coords = {'lat': lat, 'lon': lon, 'gridbox': np.arange(0, n)})


# == shortest distance from reference to all other points ==
def find_distance(da_ref, distance_matrix):
    ''' distance from reference scene (like a line) to all other points in the domain 
    da_ref needs to be [1, 0], cannot be nan
    '''
    I, J = zip(*np.argwhere(da_ref.data))                                                       # find (i, j) coordinates 
    I, J = list(I), list(J)                                                                     #
    I, J = np.array(I), np.array(J)                                                             #
    n_list = I * len(distance_matrix['lon']) + J                                                # convert to gridbox number from distance_matrix
    distance_from_ref_map = distance_matrix.sel(gridbox = n_list).min(dim = 'gridbox')          # pick out the combination of minimum gridbox distances, and take the min
    return distance_from_ref_map

