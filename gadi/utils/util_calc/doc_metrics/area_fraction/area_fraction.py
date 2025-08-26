'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == calc ==
# -- area_fraction --
def area_fraction(conv_regions, da_area):
    area_fraction = (conv_regions * da_area).sum(dim = ('lat', 'lon')) / da_area.sum()
    return area_fraction



# == when this script is ran ==
if __name__ == '__main__':
    print('executes')































