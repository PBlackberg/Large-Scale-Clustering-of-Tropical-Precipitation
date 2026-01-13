'''
# -----------------
#   plot_figure
# -----------------

'''

# == imports ==
# -- Packages --
import string
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

# -- Imported scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
def import_relative_module(module_name, plot_path):
    ''' import module from relative path '''
    if plot_path == 'utils':
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'utils')):
            print('put utils folder in cwd')
            print(f'current cwd: {cwd}')
            print('exiting')
            exit()
        module_path = f"utils.{module_name}"        
    else:
        relative_path = plot_path.replace(os.getcwd(), "").lstrip("/")
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                                                       'utils')
pF = import_relative_module('util_plot.get_subplot.corr_table_subplot',                         'utils')
cT = import_relative_module('util_calc.correlations.self_correlation_matrix',                   'utils')
cA = import_relative_module('util_calc.anomalies.monthly_anomalies.detrend_anom',               'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')
pF_M = import_relative_module('util_plot.get_subplot.map_subplot',                              'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                                'utils')


# == get metrrics ==
def get_metric(data_tyoe_group, data_type, dataset, t_freq, metric_group, metric_name, lon_area, lat_area, resolution, time_period, metric_var):
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    # -- find path --
    folder = f'{folder_work}/metrics/{data_tyoe_group}/{data_type}/{metric_group}/{metric_name}/{dataset}'
    filename = (                                                                                                        # base result_filename
            f'{metric_name}'                                                                                            #
            f'_{dataset}'                                                                                               #
            f'_{t_freq}'                                                                                                #
            f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                       #
            f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                       #
            f'_{int(360/resolution)}x{int(180/resolution)}'                                                             #
            f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                 #
            )       
    path = f'{folder}/{filename}.nc'
    # -- find metric -- 
    metric = xr.open_dataset(path)
    if not metric_var:
        print('choose a metric variation')
        print(metric)
        print('exiting')
        exit()
    else:
        # -- get metric variation -- 
        try:
            metric = metric[metric_var]
        except:
            metric = metric['mean_area']
            
    # # -- anomalies --
    # try:
    #     metric = cA.detrend_month_anom(metric)                                                                          # correlate anomalies
    # except:                                                                                                             #
    #     metric = metric.ffill(dim='time')                                                                               # Forward fill   (fill last value for 3 month rolling mean)
    #     metric = metric.bfill(dim='time')                                                                               # Backward fill  (fill first value for 3 month rolling mean)
    #     metric = cA.detrend_month_anom(metric)                                                                          #
    # metric = metric.assign_coords(time=np.arange(360))                                                                  # make sure they have common time axis (to put in the same dataset)
    # -- mean --
    # metric = metric.mean(dim = 'time')
    return metric

def ds_to_variable(ds):
    ''' Used for example for model-mean calculation '''
    data_arrays = [ds[var] for var in ds.data_vars]
    da = xr.concat(data_arrays, dim = 'variable')
    return da

# == plot ==
def plot():
    # == specify metrics ==
    # -- settings common to all, that can be changed --
    lon_area =  '0:360'                                                                                                                                                                             # area
    lat_area =  '-30:30'                                                                                                                                                                            #
    res =       2.8      
    # -- x1 --
    x1_tfreq,   x1_group,   x1_name,    x1_var, x1_label,   x1_units =              'daily',    'conv',         'conv_map',     'conv_map_nino_vs_not_nino',    r'C',   r'[%]'

    # -- x2 --
    x2_tfreq,   x2_group,   x2_name,    x2_var, x2_label,   x2_units =              'daily',    'conv',         'conv_map',     'conv_map_mean',                r'C',   r'[%]'      

    # -- contour --
    xc_tfreq,   xc_group,   xc_name,    xc_var, xc_label,   xc_units =              'daily',    'conv',         'conv_map',     'conv_map_mean',                r'C',   r'[%]'      

    # -- doc --
    doc_tfreq,   doc_group,   doc_name,    doc_var,     doc_label,   doc_units =    'daily',    'doc_metrics',  'mean_area',    'mean_area_thres_precip_prctiles_95',                    r'A_m', r'[km$^2$]'        

    # -- tas --
    tas_tfreq,   tas_group,   tas_name,    tas_var, tas_label,   tas_units =        'monthly',  'tas',          'tas_map',      'tas_map_mean',                 r'T',   r'[$^o$C]'  


    # == cmip ==
    data_tyoe_group, data_tyoe, dataset = 'models', 'cmip', 'ACCESS-ESM1-5'
    ds_nino_diff_cmip = xr.Dataset()
    Am_diff_cmip = []
    for i, dataset in enumerate(cL.get_model_letters()):
        # print(dataset)
        # -- historical --
        p_id =      '1970-01:1999-12'       
        x1 =        get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var)
        x2 =        get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
        xc_cmip =   get_metric(data_tyoe_group, data_tyoe, dataset, xc_tfreq, xc_group, xc_name, lon_area, lat_area, res, p_id, xc_var)
        doc =       get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
        tas =       get_metric(data_tyoe_group, data_tyoe, dataset, tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))
        # -- warm --
        p_id =      '2070-01:2099-12'       
        x2_warm =    get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
        doc_warm =   get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
        tas_warm =   get_metric(data_tyoe_group, data_tyoe, dataset, tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))

        d_nino_cmip = x1
        d_Am_cmip = x1.attrs['mean_area_nino_diff']
        d_tas_cmip = tas_warm - tas
        d_conv_warming_cmip = (x2_warm - x2) #/ d_tas_cmip
        d_doc_cmip = (doc_warm - doc) #/ d_tas_cmip
        ds_nino_diff_cmip[dataset] = d_nino_cmip
        Am_diff_cmip.append(d_Am_cmip)

    Am_diff_cmip = np.array(Am_diff_cmip)
    Am_diff_cmip = np.mean(Am_diff_cmip)
    da_nino_diff_cmip = ds_to_variable(ds_nino_diff_cmip)
    d_nino_cmip = da_nino_diff_cmip.mean(dim = 'variable')
    d_nino_cmip.attrs['mean_area_nino_diff'] = Am_diff_cmip

    # == obs ==
    data_tyoe_group, data_tyoe, dataset = 'observations', 'GPCP', 'GPCP'
    # -- historical --
    p_id = '1998-01:2022-12'         
    x1 =    get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var)
    x2 =    get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
    xc_obs =    get_metric(data_tyoe_group, data_tyoe, dataset, xc_tfreq, xc_group, xc_name, lon_area, lat_area, res, p_id, xc_var)
    doc =   get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
    tas =   get_metric(data_tyoe_group, 'NOAA', 'NOAA', tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))
    # -- warm --
    p_id = '2010-01:2022-12'        
    x2_warm =    get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
    doc_warm =   get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
    tas_warm =   get_metric(data_tyoe_group, 'NOAA', 'NOAA', tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))

    d_nino_obs = x1
    d_tas_obs = tas_warm - tas
    d_conv_warming_obs = (x2_warm - x2) #/ d_tas_obs
    d_doc_obs = (doc_warm - doc) #/ d_tas_obs

    # == IFS ==
    data_tyoe_group, data_tyoe, dataset = 'models', 'IFS', 'IFS_9_FESOM_5'
    # -- historical --
    p_id = '2025-01:2049-12'          
    x1 =    get_metric(data_tyoe_group, data_tyoe, dataset, x1_tfreq, x1_group, x1_name, lon_area, lat_area, res, p_id, x1_var)
    x2 =    get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
    xc_ifs =    get_metric(data_tyoe_group, data_tyoe, dataset, xc_tfreq, xc_group, xc_name, lon_area, lat_area, res, p_id, xc_var)
    doc =   get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
    tas =   get_metric(data_tyoe_group, data_tyoe, dataset, tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))
    # -- warm --
    p_id = '2038-01:2049-12'         
    x2_warm =    get_metric(data_tyoe_group, data_tyoe, dataset, x2_tfreq, x2_group, x2_name, lon_area, lat_area, res, p_id, x2_var)
    doc_warm =   get_metric(data_tyoe_group, data_tyoe, dataset, doc_tfreq, doc_group, doc_name, lon_area, lat_area, res, p_id, doc_var).mean(dim = 'time')
    tas_warm =   get_metric(data_tyoe_group, data_tyoe, dataset, tas_tfreq, tas_group, tas_name, lon_area, lat_area, res, p_id, tas_var).mean(dim = ('lat', 'lon'))

    d_nino_ifs = x1
    d_tas_ifs = tas_warm - tas
    d_conv_warming_ifs = (x2_warm - x2) #/ d_tas_ifs
    d_doc_ifs = (doc_warm - doc) # / d_tas_ifs

    # == plot ==
    # -- create figure --
    # labels = list(string.ascii_lowercase) 
    # labels = labels + ['A', 'B']
    width, height = 6.27, 9.69                  # max size (for 1 inch margins)
    # width, height = width * 0.85, height * 0.25    # modulate size and subplot distribution

    width, height = 17.5, 8                                                                                                            # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]   

    ncols, nrows  = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height))
    axes_list = axes.flatten()
    for i, (model, ax) in enumerate(zip(['ACCESS-ESM1-5', 'GPCP', 'IFS_9_FESOM_5'] * 2, axes_list)):
        row, col = divmod(i, ncols)

        # -- format for all subplots --
        xticks = [60, 120, 180, 240, 300]
        yticks = [-20, 0, 20]
        # cmap = 'RdBu'
        cmap = 'BrBG'
        hide_xticks =   False
        hide_xlabel =   True
        hide_yticks =   True
        hide_ylabel =   True
        hide_cbar =     False
        metric_text = r'dA$_m$'
        units_text = r'km$^2$'
        ds_map, ds_contour = xr.Dataset(), xr.Dataset()
        if row == 0:
            if model == 'ACCESS-ESM1-5':
                ds_map['var'], ds_contour['var'] = d_nino_cmip, xc_cmip
                # print(d_nino_cmip.attrs["mean_area_nino_diff"])
                # exit()
                label = f'model-mean, {metric_text}: {d_nino_cmip.attrs["mean_area_nino_diff"]:.2e}'
            elif model == 'GPCP':
                ds_map['var'], ds_contour['var'] = d_nino_obs, xc_obs
                label = f'\u2605 {model}, {metric_text}: {d_nino_obs.attrs["mean_area_nino_diff"]:.2e}'
            elif model == 'IFS_9_FESOM_5':
                ds_map['var'], ds_contour['var'] = d_nino_ifs, xc_ifs
                label = f'\u25C6 {model}, {metric_text}: {d_nino_ifs.attrs["mean_area_nino_diff"]:.2e}'
        else:
            if model == 'ACCESS-ESM1-5':
                ds_map['var'], ds_contour['var'] = d_conv_warming_cmip, xc_cmip
                label = f'{cL.get_model_letters()[model]}: {model}, {metric_text}: {d_doc_cmip.data:.2e} {units_text}'
            elif model == 'GPCP':
                ds_map['var'], ds_contour['var'] = d_conv_warming_obs, xc_obs
                label = f'\u2605: {model}, {metric_text}: {d_doc_obs.data:.2e} {units_text}'
            elif model == 'IFS_9_FESOM_5':
                ds_map['var'], ds_contour['var'] = d_conv_warming_ifs, xc_ifs
                label = f'\u25C6: {model}, {metric_text}: {d_doc_ifs.data:.2e} {units_text}'
        var_name = list(ds_map.data_vars.keys())[0]

        # -- format for specific subplots --
        if row == 0:
            move_row = 0.05
        if row == 1:
            move_row = 0.075 - 0.005

        if col == 0:
            hide_yticks =   False
            hide_ylabel =   True
            move_col = -0.055    
        if col == 1:
            move_col = -0.0165
        if col == 2:
            move_col = 0.0225   
        if row == nrows - 1:
            hide_xticks =   False
            hide_xlabel =   False
            hide_cbar =     False

        # -- plot subplot --
        ds_map.attrs.update({ 'scale': 1.3, 'move_row': move_row, 'move_col': move_col,                                                                    # format axes
        'name': f'{var_name}',                                                                                                                              # plot
        'vmin': -1.75, 'vmax': 1.75, 'cmap': cmap, 'cbar_height': 0.015, 'cbar_pad': 0.115,                                                                  # colorbar: position
        'cbar_label': f'd{x2_label} {x2_units}', 'cbar_fontsize': 8, 'cbar_numsize': 8, 'cbar_label_pad': 0.115,                                            # colorbar: label                
        'hide_xticks': hide_xticks, 'xticks': xticks, 'xticks_fontsize': 8,                                                                                 # x-axis:   ticks
        'hide_xlabel': hide_xlabel, 'xlabel_label': 'longitude', 'xlabel_pad': 0.09, 'xlabel_fontsize': 8,                                                 # x-axis:   label
        'hide_yticks': hide_yticks, 'yticks': yticks, 'yticks_fontsize':  8,                                                                                # y-axis:   ticks
        'hide_ylabel': hide_ylabel, 'ylabel_label': 'latitude', 'ylabel_pad': 0.065, 'ylabel_fontsize': 8,                                                  # y-axis:   label
        'axtitle_label': f'{label}', 'axtitle_xpad': -0.0025, 'axtitle_ypad': 0.0075, 'axtitle_fontsize': 8,
        'hide_colorbar': hide_cbar,         
        })                                                     # subplot   title
        ds_contour.attrs.update({'threshold': ds_contour[ds_map.attrs['name']].quantile(0.90, dim=('lat', 'lon')).data, 'color': 'k', 'linewidth': 0.5})    # contour
        axes[row, col] = pF_M.plot(fig, nrows, ncols, row, col, ax = axes[row, col], ds = ds_map, ds_contour = ds_contour, lines = [])
        # if i > 3:
        #     break

    # -- remove extra subplots --
    # for row, col in np.ndindex(nrows, ncols):
        # ''
        # if row > 1 and col > 0:
            # axes[row, col].remove()
        # if (row == 2 and col == 2):
        #     axes[row, col].remove()    
    # axes[6, 3].remove()

    # -- figure description --
    description_x_pos = 0.5
    description_y_pos = 0.885
    fontsize = 8
    fig.text(description_x_pos, description_y_pos,  f'Nino conditions vs all \n (CMIP, OBS, IFS)',  transform=fig.transFigure,  fontweight='bold', fontsize = fontsize, ha='center' )

    # description_x_pos = 0.5
    # description_y_pos = 0.525
    # fontsize = 8
    # fig.text(description_x_pos, description_y_pos,  f'Change with warming \n (CMIP, OBS, IFS)',  transform=fig.transFigure,  fontweight='bold', fontsize = fontsize, ha='center' )

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)           # user settings
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'nino_diff_cmip_obs_ifs'
    path = f'{folder}/{plot_name}.png'
    
    print(path)
    exit()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot()




