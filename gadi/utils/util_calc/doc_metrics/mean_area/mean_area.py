'''
# ------------
#  util_calc
# ------------

'''

# == imports ==
import xarray as xr
import numpy as np


# == calc ==
# -- mean_area --
def get_mean_area(labels_xr, labels, da_area):
    obj_areas = []
    for i, obj_label in enumerate(labels): 
        obj_scene = labels_xr.isin(obj_label)
        obj_scene = xr.where(obj_scene > 0, 1, 0)
        obj_areas.append((da_area * obj_scene).sum()) 
    return np.mean(np.array(obj_areas))


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')




