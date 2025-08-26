'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == calc ==
def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])
    return da


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')










