'''
# -----------------
#    Main_func
# -----------------

'''

# == imports ==
# -- Packages --
import itertools
from distutils.util import strtobool  
import numpy as np
import xarray as xr

# -- util- and local scripts --
import os
import sys
import importlib
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
        relative_path = os.path.relpath(file_path, cwd) # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                   'utils')
jS = import_relative_module('submit_as_job',                __file__)
mC = import_relative_module('calc_metric',                  __file__)
gD = import_relative_module('util_cmip.get_cmip_data',      'utils')


# == process data ==
def process_data(da):
    ''
    return da

def process_hus(da):
    da = da.sel(plev = 700e2, method='nearest')                                                                                         # lower troposphere
    return da

# == get metric ==
def get_metric(dataset, t_freq, lon_area, lat_area, resolution, time_period, years, months, r_folder, r_filename, section_range, test):    
    _, folder_scratch, _, _, _ = mS.get_user_specs()                                                                                    # for temporary saving
    # -- get data --                                                                                                                    #
    if test:                                                                                                                            # for quickly testing some timesteps
        print('getting test data')
        try:
            folder = f'{folder_scratch}/temp_data/{r_folder}/{r_filename}'
            filename = f'{r_filename}_var.nc'
            path = f'{folder}/{filename}'
            filename = f'{r_filename}_hus.nc'
            path_hus = f'{folder}/{filename}'
            da = xr.open_dataset(path)['var']
            hus = xr.open_dataset(path_hus)['var']
        except:
            print('no saved test data')
            print('getting data for saving ..')
            process_request = ['pr', dataset, t_freq, resolution, time_period]                                                          # specify data request (for test)
            da = gD.get_data(process_request, process_data_further = process_data)                                                      # get data (for test)
            time_str1 = f"{years[0]}-{int(months[0]):02d}"                                                                              # pick out the relevant section
            time_str2 = f"{years[-1]}-{int(months[-1]):02d}"                                                                            #
            da = da.sel(time=slice(f"{time_str1}", f"{time_str2}")).load()                                                              #
            print(da)
            process_request = ['hus', dataset, 'monthly', resolution, time_period]                                                      # specify data request
            hus = gD.get_data(process_request, process_data_further = process_hus)                                                      # get data
            time_str1 = f"{years[0]}-{int(months[0]):02d}"                                                                              # pick out the relevant section
            time_str2 = f"{years[-1]}-{int(months[-1]):02d}"                                                                            #
            hus = hus.sel(time=slice(f"{time_str1}", f"{time_str2}")).load()                                                            #
            print(hus)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            xr.Dataset({'var': da}).to_netcdf(path)
            xr.Dataset({'var': hus}).to_netcdf(path_hus)
            print('saved test data ..')
            print('now run again')
            print('exiting')
            exit()
    else:
        print('getting data ..')
        process_request = ['pr', dataset, t_freq, resolution, time_period]                                                              # specify data request
        da = gD.get_data(process_request, process_data_further = process_data)                                                          # get data
        time_str1 = f"{years[0]}-{int(months[0]):02d}"                                                                                  # pick out the relevant section
        time_str2 = f"{years[-1]}-{int(months[-1]):02d}"                                                                                #
        da = da.sel(time=slice(f"{time_str1}", f"{time_str2}")).load()                                                                  #
    
        process_request = ['hus', dataset, 'monthly', resolution, time_period]                                                          # specify data request
        hus = gD.get_data(process_request, process_data_further = process_hus)                                                          # get data
        time_str1 = f"{years[0]}-{int(months[0]):02d}"                                                                                  # pick out the relevant section
        time_str2 = f"{years[-1]}-{int(months[-1]):02d}"                                                                                #
        hus = hus.sel(time=slice(f"{time_str1}", f"{time_str2}")).load()                                                                #

    # -- get metric for given (years, months) --
    print('getting metric ..')
    data_objects = [da, lon_area, lat_area, dataset, years, hus]
    ds = mC.calculate_metric(data_objects)

    # -- save result from section --
    folder = f'{folder_scratch}/temp_calc/{r_folder}/{r_filename}'
    filename = f'{r_filename}_{section_range}.nc'
    path = f'{folder}/{filename}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path, mode="w")
    print('saved section result')


# == concatenate results ==
def concat_result(r_folder, r_filename, test):
    # -- load collection of partial results --
    print('finding temp files for section results')    
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs()
    folder = f'{folder_scratch}/temp_calc/{r_folder}/{r_filename}'
    temp_files = [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.nc')]

    # -- concatenate --
    print('concatenating results')    
    ds = xr.open_mfdataset(temp_files, combine="by_coords", engine="netcdf4", parallel=True).load()
    print(ds)

    if not test:
        # -- save result --
        folder = f'{folder_work}/metrics/{r_folder}'
        filename = f'{r_filename}.nc'
        path = f'{folder}/{filename}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.to_netcdf(path, mode="w")
        print('saved result')    

        # -- remove tempfiles --
        print('removing temp files')    
        [os.remove(path_temp) for path_temp in temp_files]


# == main ==
def main(switch, dataset, t_freq, lon_area, lat_area, resolution, time_period, years, months, r_folder, r_filename, section_range, test = False):
    if switch.get('calc'):
        get_metric(dataset, t_freq, lon_area, lat_area, resolution, time_period, years, months, r_folder, r_filename, section_range, test)
    if switch.get('concat'):
        concat_result(r_folder, r_filename, test)
    print('finished')        
    if os.environ.get('PBS_SCRIPT'):
        print(f'removing resource script')
        os.remove(os.environ.get('PBS_SCRIPT'))


# == when this script is ran / submitted ==
if __name__ == '__main__':
    if not os.environ.get("PBS_SCRIPT"):                                                                                                # when run interactively (test)
        datasets, t_freqs, lon_areas, lat_areas, resolutions, time_periods = jS.set_specs()                                             # all specs
        for i, (t, lat, lon, r, d, p) in enumerate(itertools.product(t_freqs,                                                           #
                                                                    lat_areas,                                                          #
                                                                    lon_areas,                                                          #
                                                                    resolutions,                                                        #
                                                                    datasets, time_periods)):                                           # loops over all specs (looped in input order)
            r_folder, r_filename = jS.get_path(d, t, lon, lat, r, p)
            print(f'Running metric for:')
            print(f'folder:     {r_folder}')
            print(f'filename:   {r_filename}')
            time_section = jS.get_timesections(n_jobs = 1, time_period = p)[0] 
            years_section, months_section = zip(*time_section)            
            year1_section, month1_section = time_section[0]
            year2_section, month2_section = time_section[-1]
            section_range =  f'{year1_section}_{month1_section}-{year2_section}_{month2_section}'   
            main(switch =           {'calc': True, 
                                     'concat': True},
                 dataset =          d,
                 t_freq =           t,
                 lon_area =         lon,
                 lat_area =         lat,
                 resolution =       r,
                 time_period =      p,
                 years =            years_section,
                 months =           months_section,
                 r_folder =         r_folder,
                 r_filename =       r_filename,
                 section_range =    section_range,
                 test =             True,
                 )
            exit()
    else:                                                                                                                               # when submitted (save)
            main(switch =           {'calc': strtobool(os.environ.get("SWITCH_CALC")), 
                                    'concat': strtobool(os.environ.get("SWITCH_CONCAT"))},
                 dataset =          os.environ.get('DATASET'),
                 t_freq =           os.environ.get('T_FREQ'),
                 lon_area =         os.environ.get('LON_AREA'),
                 lat_area =         os.environ.get('LAT_AREA'),
                 resolution =       float(os.environ.get('RESOLUTION')),
                 time_period =      os.environ.get('TIME_PERIOD'),
                 years =            os.environ.get("YEAR").split(':'),
                 months =           os.environ.get("MONTH").split(':'),
                 r_folder =         os.environ.get('R_FOLDER'),
                 r_filename =       os.environ.get('R_FILENAME'),
                 section_range =    os.environ.get('SECTION_RANGE'),
                 )



