''' 
# ------------------------
#     Get cmip data
# ------------------------
For cloud variable, cloud interpolation takes a long time, so stored locally. Run utils/util_data/cmip/cl_cmip_regridVert.py to regenerate         
Currently saved here
path_to_saved_data = f'/g/data/k10/cb4968/temp_saved/data/cmip/cl/cmip6/{model}_cl_{timescale}_{experiment}_{resolution}.nc'
so change this line to the folder you save the data in

'''

# -- Packages --
import xarray as xr
import numpy as np
import warnings

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_cmip.model_institutes     as mI
import utils.util_cmip.conserv_interp       as cI


# == Deal with vertical coordinates ==
def regrid_vert(da, model = ''):                                                                                            # does the same thing as scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (cloud fraction is dealt with separately) '''
    da['plev'] = da['plev'].round(0)                if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else da['plev']             # plev coordinate is specified to a higher number of significant figures in these models
    p_new = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])      
    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    return da_p_new
                                                                

# == Specify nci folders ==
def folder_ensemble(source, model, experiment):
    ''' Some models don't have the ensemble most common amongst other models and some models don't have common ensembles for historical and warmer simulation '''
    if source == 'cmip5':
        ensemble = 'r6i1p1'    if model in ['EC-EARTH', 'CCSM4']                                                        else 'r1i1p1'
        ensemble = 'r6i1p1'    if model in ['GISS-E2-H'] and experiment == 'historical'                                 else ensemble
        ensemble = 'r2i1p1'    if model in ['GISS-E2-H'] and not experiment == 'historical'                             else ensemble
    if source == 'cmip6':
        ensemble = 'r1i1p1f2'  if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'MIROC-ES2L']  else 'r1i1p1f1'
        ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical'                                   else ensemble
    return ensemble

def folder_grid(model):
    ''' Some models have a different grid folder in the path to the files (only applicable for cmip6) '''
    folder = 'gn'
    folder = 'gr'  if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-f3-L', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM', 'GFDL-ESM4', 'INM-CM4-8'] else folder           
    return folder

def folder_latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def folder_var(var, model, experiment, ensemble, project, timeInterval):
    ''' ACCESS model is stored separately '''
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mI.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}'
        folder_to_grid = folder_grid(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mI.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}' 
        folder_to_grid = folder_grid(model)
        version = folder_latestVersion(os.path.join(path_gen, folder_to_grid))
    return f'{path_gen}/{folder_to_grid}/{version}'


# == Get data ==
def concat_data(path_folder, model, experiment):
    ''' Concatenate files between specified years '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if experiment == 'historical':
        year1 = '1970-1999'.split('-')[0]
        year2 = '1970-1999'.split('-')[1]
    else:
        year1 = '2070-2099'.split('-')[0]
        year2 = '2070-2099'.split('-')[1]
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder      else (17, 13)                                    # character index range for starting year of file (counting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  if 'Amon' in path_folder      else (8, 4)                                      #                             end
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    if model in ['IPSL-CM6A-LR']:              
        ds = xr.open_mfdataset(paths, combine='by_coords', use_cftime=True)                                                                         
    else:
        ds = xr.open_mfdataset(paths, combine='by_coords')
    ds = ds.sel(time=slice(str(year1), str(year2)))                                                                                 
    return ds

def get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval): 
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''       
    if var_name in ['cl']:                                                                                                          # cloud interpolation takes a long time, so stored locally. Run utils/util_data/cmip/cl_cmip_regridVert.py to regenerate              
        try:
            path_to_saved_data = f'/g/data/k10/cb4968/data/cmip/cl/cmip6/{model}_cl_{timescale}_{experiment}_{int(360/resolution)}x{int(180/resolution)}.nc'
            ds = xr.open_dataset(path_to_saved_data) 
        except:
            print('cloud interpolation takes a long time, so stored locally. Run utils/util_data/cmip/cl_cmip_regridVert.py to regenerate')
            print('exiting')
            exit()                                                                                   
        return ds         

def get_var_data(var_name, model, experiment, resolution, timescale, source, regrid_resolution):
    ''' concatenate files, interpolate grid, and mask '''
    ensemble = folder_ensemble(source, model, experiment)
    project =       'CMIP'      if experiment ==    'historical'    else 'ScenarioMIP'
    time_interval = 'day'       if timescale ==     'daily'         else 'Amon'         
    if var_name in ['cl']:                                                                                                          # cloud variable is on hybrid-sigma pressure coordiantes, so treated separately (uses saved version for the moment)
        da = get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval)         #
        da = ds[var_name]              
        da = cI.conservatively_interpolate(da_in =          da.load(), 
                                           res =            regrid_resolution, 
                                           switch_area =    None,                                                                   # regrids the whole globe
                                           simulation_id =  model
                                           )
        return xr.Dataset(data_vars = {f'{var_name}': da}, attrs = ds.attrs)                   
    else:                                                                                                                           # all other 2D or 3D pressure level variables     
        if var_name == 'ds_cl':
            var_name = 'cl'                                                                                                         # as it is named on nci
        folder_to_var = folder_var(var_name, model, experiment, ensemble, project, time_interval)
        ds = concat_data(folder_to_var, model, experiment)    
        if var_name == 'ds_cl':                                                                                                     # as it is named on nci
            return ds                                                                                                               # this is for interpolating cloud variable
        da = ds[var_name]              
        da = regrid_vert(da, model)                         if 'plev' in da.dims    else da                                         # vertically interpolate (some models have different number of vertical levels)
        da = cI.conservatively_interpolate(da_in =          da.load(), 
                                           res =            regrid_resolution, 
                                           switch_area =    None,                                                                   # regrids the whole globe
                                           simulation_id =  model
                                           )
        return xr.Dataset(data_vars = {f'{var_name}': da}, attrs = ds.attrs)                                                       

def convert_units(model, da, var_name):
    if var_name == 'pr':
        da = da * 60 * 60 * 24
    if var_name == 'tas':
        da = da - 273.15  
    if var_name == 'wap':
        da = da * 60 * 60 * 24 / 100
        if model in ['IITM-ESM']: 
            da = da * 1000
    if var_name == 'clwvi':
        if model in ['IITM-ESM']: 
            da = da/1000
    if var_name == 'zg':
        da = da * 9.8                                                                                                               # This variable is in m, but geeopotential height has units of m^2/s^2, so multiply with the gravitational acceleration (this  m^2/s^2 is the same as J/Kg)
        print('converted zg units from m to m**2/s**2')
    return da

def get_cmip_data(switch_var, model, experiment, resolution, timescale, source, regrid_resolution):
    var_name = next((key for key, value in switch_var.items() if value), None)
    da = get_var_data(var_name, model, experiment, resolution, timescale, source, regrid_resolution)[var_name] if not var_name == 'ds_cl' else get_var_data(var_name, model, experiment, resolution, timescale, source, regrid_resolution)
    da = convert_units(model, da, var_name)
    da = da.load()
    return da 

def get_data(process_request, process_data_further):
    var, dataset, t_freq, resolution, time_period = process_request
    # -- determine if historical or warm --
    start_year = int(time_period.split("-")[0])
    if 1970 <= start_year <= 1999:
        time_period_string = '1970_1-1999_12'
        experiment = 'historical'
    elif 2070 <= start_year <= 2099:
        time_period_string = '2070_1-2099_12'
        experiment = 'ssp585'
    else:
        time_period_string = None
        experiment = None
        print('pick a valid time period')
        print('exiting')
        exit()
    # -- get data --
    da = get_cmip_data(switch_var =     {var : True}, 
                  model =               dataset, 
                  experiment =          experiment, 
                  resolution =          'regridded', 
                  timescale =           t_freq, 
                  source =              'cmip6',
                  regrid_resolution =   resolution
                  )
    # -- custom process --
    da = process_data_further(da)
    return da


# == test ==
if __name__ == '__main__':
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f'running {script_name}')

    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'zg':       False,                                                                                      # Moist Static Energy
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,                                          # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,                      # Shortwave radiation
        'cl':       True,  'ds_cl':         False,                                                              # Cloudfraction (ds_cl is used for interpolating to pressure levels)
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    var_name = next((key for key, value in switch_var.items() if value), None)
    print(f'testing variable: {var_name}')
    process_request = [var_name, 'IITM-ESM', 'monthly', 2.8, '1970-01:1999-12']

    def process_data(da):
        ''
        return da

    da = get_data(process_request, process_data)
    print(da)
    exit()



