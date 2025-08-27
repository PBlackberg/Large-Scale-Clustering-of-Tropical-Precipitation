'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
# -- Packages --
import os
import sys
import importlib
import numpy as np
import xarray as xr

# -- util- and local scripts --
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, file_path):
    ''' import module from relative path '''
    if file_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        cwd = os.getcwd()
        relative_path = os.path.relpath(file_path, cwd)  # This ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                   'utils')
sL = import_relative_module('util_files.save_and_load',     'utils')
jS = import_relative_module('submit_as_job',                __file__)
pF = import_relative_module('helper_funcs.plot_func_map',   __file__)


# == plot settings ==
def plot_settings(ds, title = '', threshold = 0, vmin = None, vmax = None, cmap = 'Blues'):
    ''' format plot '''
    width, height = 6.27, 9.69                      # max (for 1 inch margins)
    ds.attrs.update({
        # -- Figure -- 
        'width':            1.5 * width,    
        'height':           0.3 * height,   
        'nrows':            1,
        'ncols':            1,
        # -- Figure -- 
        'axtitle_label':    f'{title}',
        'axtitle_xpad':     0.01,
        'axtitle_ypad':     0.045,
        'axtitle_fontsize': 12,
        # -- plot -- 
        'name':             'var',
        'cmap':             cmap,
        'vmin':             vmin,
        'vmax':             vmax,
        'threshold':        threshold,
        # -- axes -- 
        'scale':            1.15,
        'move_row':         0.1,
        'move_col':         -0.05,
        # 'yticks':           [-20, 0, 20],
        'yticks_fontsize':  10,
        'ylabel_pad':       0.065,
        'ylabel_label':     'latitude',
        'ylabel_fontsize':  10,
        # 'xticks':           [30, 90, 150, 210, 270, 330],
        'xticks_fontsize':  10,
        'xlabel_pad':       0.15,
        'xlabel_label':     'longitude',
        'xlabel_fontsize':  10,
        'yticks':           np.round(np.linspace(ds.lat.min(), ds.lat.max(), 5)).astype(int),
        'xticks':           np.round(np.linspace(ds.lon.min(), ds.lon.max(), 6)).astype(int),
        # -- colorbar --
        'cbar_height':      0.035,   
        'cbar_pad':         0.2,
        'cbar_label_pad':   0.125,
        'cbar_label':       r'[mm day$^{-1}$]',
        'cbar_numsize':     10,    
        'cbar_fontsize':    10,
        })
    return ds


# == get_metric ==
def get_metric(da, year, month, day, i, plot = True):
    plot_folder = sL.create_plot_folder(__file__)
    # -- plot --
    if plot and year == '2025' and month == '1' and day == 1:
        ds_plot = xr.Dataset({'var': da})
        plot_settings(ds_plot, vmin = 0, vmax = 14)
        plot_path = f'{plot_folder}/daily_field.png'   # save in current folder
        pF.plot_snapshot(ds_plot, plot_path)

    # -- fill xr.dataset with metric --
    ds = xr.Dataset()
    ds['pr'] = da     
    ds = ds.expand_dims(dim = 'time')
    ds = ds.assign_coords(time=[da.time.data])
    return ds





