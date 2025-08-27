'''
# -----------------
#    Main_func
# -----------------

'''

# == imports ==
# -- Packages --
import os
import sys
import importlib
import xarray as xr
import dask
from distributed import get_client
from distutils.util import strtobool        

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
        relative_path = os.path.relpath(file_path, cwd)                                                                                                 # ensures the path is relative to cwd
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                   'utils')
sL = import_relative_module('util_files.save_and_load',     'utils')
gD = import_relative_module('util_ifs.get_ifs_data',        'utils')
dF = import_relative_module('util_dask.dask_funcs',         'utils')
jS = import_relative_module('submit_as_job',                __file__)
mC = import_relative_module('calc_metric',                  __file__)


# == data process ==
def process_data(da):
    da = da * 24 * 1000                                                                                                                                 # convert units:    [m/h] -> [mm/day]
    return da


# == node task ==
def worker_task(years, months, worker_func, dask_adapted, dask_workers, folder_scratch, strings):
    ds_dask, temp_data_path, ds_list = None, None, []                                                                                                   # filled in later
    for idx, (year, month) in enumerate(zip(years, months)):                                                                                            # loop through months
        if dask_adapted:                                                                                                                                # if iteratively temp saving data
            temp_data_path = sL.get_temp_data_path(folder_scratch,                                                                                      # get temp data path    
                                                   sub_folder =     strings[2],                                                                         # folder 
                                                   file_name_base = strings[3], year = year, month = month                                              # filename
                                                   )                                                                                                    # temp data path generated
        switch_time         = {'year': int(year), 'month': int(month)}                                                                                  # process request: time section
        process_request     = ['tp', dictionaries[0], dictionaries[1], dictionaries[2], switch_time]                                                    # process request: area, resample, regrid 
        ds_dask, da = gD.get_data(ds_dask, year, process_request, dask_adapted, temp_data_path, process_data_further = process_data)                    # get data
        ds_list_month = []                                                                                                                              # initiate monthly calc
        for i, day in enumerate(gD.get_valid_days(year, month)):                                                                                        # loop through days
            ds_list_month.append(worker_func(da, year, month, day, i, folder_scratch, strings))                                                         # worker func execution
        print(f'finished year: {year}, month: {month}')                                                                                                 # month complete
        if dask_workers:                                                                                                                                # if dask distributed tasks
            dask.compute(*ds_list_month)                                                                                                                # let the workers finish a month at a time
        ds_list.extend(ds_list_month)                                                                                                                   # add result from month
        if dask_adapted:                                                                                                                                # if iteratively temp saving data
            os.remove(temp_data_path)                                                                                                                   # remove temp data for month
    ds = xr.concat(ds_list, dim='time')                                                                                                                 # concatenate to section resullt
    sL.save_temp_calc(folder_scratch,                                                                                                                   # save section result
                      sub_folder = strings[2],                                                                                                          # folder
                      filename = f'{strings[3]}_yearend_{years[-1]}_monthend_{months[-1]}.nc',                                                          # filename
                      ds = ds
                      )
    
# == concat ==
def concat_result(folder_scratch, strings, test, years, months, work_folder):
    section_range = f'{years[0]}_{months[0]}-{years[-1]}_{months[-1]}'                                                                                  # open files from requested range
    temp_paths = sL.find_tempfiles(temp_calc_folder = f'{folder_scratch}/temp_calc/{strings[2]}',                                                       # folder
                                rfilename = strings[3], date_range = section_range,                                                                     # filename
                                )
    ds = xr.open_mfdataset(temp_paths, combine="by_coords", engine="netcdf4", parallel=True).load()
    if not test:
        sL.save_data_result(work_folder = work_folder, 
                        sub_folder = strings[2], 
                        filename = f'{strings[3]}.nc', 
                        ds = ds
                        )
        sL.remove_tempfiles(temp_paths)

# == main ==
def main(switch, years, months, dask_adapted, dask_workers, test, folder_scratch, strings, work_folder):
    # -- timestep calc func --
    def worker_func(da, year, month, day, i, folder_scratch, strings):
        if da is None:                                                                                                                                  # either use temp saved data, or given data
            da = xr.open_dataset(sL.get_temp_data_path(folder_scratch, sub_folder = strings[2],                                                         # folder
                                                       file_name_base = strings[3], year = year, month = month)                                         # filename
                                                       )['var'].isel(time = i).load()                                                                   # saved
        else:                                                                                                                                           # or
            da = da.isel(i)                                                                                                                             # given data
        ds_metric = mC.get_metric(da, year, month, day, i)                                                                                              # execute metric calc
        return ds_metric
    # -- start execution --
    if dask_workers:
        dF.create_dask_cluster(nworkers = 6, cpus_per_worker = 'use_all', memory_per_worker = 'default', print_worker = True)    
    # -- section calc --
    if switch.get('calc'):
        worker_func = dF.apply_dask_delayed(apply_decorator = dask_workers)(worker_func)                                                                # apply @dask.delayed if requested
        worker_task(years, months, worker_func, dask_adapted, dask_workers, folder_scratch, strings)
    # -- section concat --
    if switch.get('concat'):
        concat_result(folder_scratch, strings, test, years, months, work_folder)
    # -- finish execution --
    print('finished')        
    if os.environ.get('SLURM_SCRIPT'):
        print(f'removing resource script')
        os.remove(os.environ.get('SLURM_SCRIPT'))
    if dask_workers:
        client = get_client()
        client.close()
        print("Dask client closed.")


# == when this script is ran / submitted ==
if __name__ == '__main__':
    # -- interactive run --
    if not os.environ.get("SLURM_SCRIPT"): 
        username, SU_project, folder_scratch = mS.get_user_specs()                                                                                      # user settings     (temp data folder)
        strings, dictionaries = jS.get_specs()                                                                                                          # metric settings   (from submit_as_job.py)
        work_folder = mS.get_work_folder()                                                                                                              # work folder
        for time_period in strings[4]:                                                                                                                  # for timesection in timesections
            time_section = gD.get_timesections(n_jobs = 1, time_period = time_period)[0]                                                                # pick out timesection (whole timeperiod for n_jobs = 1)
            years_section, months_section = zip(*time_section)                                                                                          # put in format the function can handle
            main(switch =           {'calc': True, 'concat': True},                                                                                     # choose operation
                years =            years_section,                                                                                                       # years
                months =           months_section,                                                                                                      # months
                dask_adapted =     True ,                                                                                                               # if progressively saving
                dask_workers =     False,                                                                                                               # if splitting calculation on dask workers
                test =             True,                                                                                                               # saving result as test
                folder_scratch =   folder_scratch,
                strings =          strings,
                work_folder =      work_folder
                )
    # -- when submitted --
    else:
        username, SU_project, folder_scratch = mS.get_user_specs()                                                                                      # user settings     (temp data folder)
        strings, dictionaries = jS.get_specs()                                                                                                          # metric settings   (from submit_as_job.py)
        work_folder = mS.get_work_folder()                                                                                                              # work folder
        main(switch =           {'calc':   strtobool(os.environ.get("SWITCH_CALC")), 
                                'concat': strtobool(os.environ.get("SWITCH_CONCAT"))},
             years =            os.environ.get("YEAR").split(':'), 
             months =           os.environ.get("MONTH").split(':'),
             dask_adapted =     strtobool(os.environ.get("DASK_ADAPTED")), 
             dask_workers =     strtobool(os.environ.get("DASK_WORKERS")),
             folder_scratch =   folder_scratch, 
             strings =          strings,
             work_folder =      work_folder,
             test =             False,
             )

