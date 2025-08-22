''' 
# ----------------
#   regrid_vert
# ----------------

'''

# == imports ==
# -- packages --
import numpy as np
import warnings

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_cmip.conserv_interp       as cI


# == regrid ==
def regrid_vert(da):                                                                                # does the same thing as scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (cloud fraction is dealt with separately)'''
    p_new = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])      
    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    return da_p_new





