'''
# -----------------
#  submit_as_job
# -----------------

'''

# == Imports ==
# -- Packages --
import os
import sys
import importlib
from datetime import timedelta
from pathlib import Path

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
        relative_path = file_path.replace(os.getcwd(), "").lstrip("/")
        module_base = os.path.dirname(relative_path).replace("/", ".").strip(".")
        module_path = f"{module_base}.{module_name}"
    return importlib.import_module(module_path)
mS = import_relative_module('user_specs',                   'utils')
sJ = import_relative_module('util_slurm.submission_funcs',  'utils')
gD = import_relative_module('util_ifs.get_ifs_data',        'utils')


# == Set specs ==
def set_specs():
    dataset = (
        'IFS_9_FESOM_5',        # ~10 km resolution atm, 30 years   
        )
    lon_area = (
        '0:360',                # Full domain
        )
    resolution = (
        # 0.1,                  # 10 km
        # 0.25,                 # ERA5 resolution
        2.8,                    # CMIP resolution            
        )
    time_periods = (
        '2025-01:2049-12',      # full period
        )
    return dataset, lon_area, resolution, time_periods

def get_specs(show = False):
    dataset, lon_area, resolution, time_periods = set_specs()
    folder_0 = Path(__file__).resolve().parents[2].name                                                                      
    folder_1 = Path(__file__).resolve().parents[1].name                                                                     
    folder_2 = Path(__file__).resolve().parents[0].name                                                                      
    switch_area     = {'lon_area': list(map(int, lon_area[0].split(':'))), 'lat_area': list(map(int, '-30:30'.split(':'))),}
    switch_process  = {'resample_daily': True, 'latlon_grid': True, 'regridded': True,}
    switch_res      = {'lon_res':  resolution[0], 'lat_res':  resolution[0]}
    r_folder        = f'{folder_0}/{folder_1}/{folder_2}'
    r_filename      = (
                        f'{dataset[0]}_daily_{folder_2}'
                        f'_{switch_area["lon_area"][0]}-{switch_area["lon_area"][1]}'
                        f'_{switch_area["lat_area"][0]}-{switch_area["lat_area"][1]}'
                        f'_{int(360/switch_res.get("lon_res"))}x{int(180/switch_res.get("lat_res"))}'           
                        f'_{time_periods[0].split(":")[0]}_{time_periods[0].split(":")[1]}'               
                       )
    strings         = (folder_2, dataset, r_folder, r_filename, time_periods)
    dictionaries    = (switch_area, switch_process, switch_res)
    if show:
        print('strings:')
        [print(f'{i}. {f}') for i, f in enumerate(strings)]
        print('')
        print('dictionaries:')
        [print(f"{i}. {key}:\t{value}") for i, d in enumerate(dictionaries) for key, value in d.items()]
        print('')
    return strings, dictionaries


# == main ==
def main():
    username, SU_project, folder_scratch = mS.get_user_specs(show = True)   # user settings
    strings, dictionaries = get_specs(show = True)                          # metric settings
    python_script = f'{os.path.dirname(__file__)}/main_func.py'
    switch = {
        'calc':                 True,
        'concat':               True,
        }
    switch_job = {
        'dask_adapted':         True,  
        'dask_workers':         False
        }
    print('job queue before submission:')
    sJ.check_squeue(username, delay=0)
    job_ids = []
    walltime_tot = timedelta()
    # -- for each selected timeperiod --
    for time_period in strings[4]:
        n_jobs = 2
        # -- calc job for each section -- 
        if switch.get('calc'):    
            print(f'-- submitting calc jobs, timeperiod: {time_period} --')
            env_variables = {}
            env_variables["SWITCH_CALC"], env_variables["SWITCH_CONCAT"] = True, False
            env_variables["DASK_ADAPTED"], env_variables["DASK_WORKERS"] = switch_job.get('dask_adapted'), switch_job.get('dask_workers')
            walltime = '0:50:00'
            exclusive = True    # gets whole node
            for i, time_section in enumerate(gD.get_timesections(n_jobs, time_period)):
                year1_section, month1_section = time_section[0]
                year2_section, month2_section = time_section[-1]
                section_range =  f'{year1_section}_{month1_section}-{year2_section}_{month2_section}'
                years_section, months_section = zip(*time_section)
                env_variables["YEAR"] = ':'.join(map(str, years_section))
                env_variables["MONTH"] = ':'.join(map(str, months_section))
                job_ids.append(sJ.submit_job_slurm(python_script = python_script, 
                                    folder =        f'{folder_scratch}/oe_files/{strings[2]}', 
                                    filename =      f'calc_{strings[3]}_{section_range}', 
                                    walltime =      walltime,
                                    env_variables = env_variables, 
                                    SU_project =    SU_project,
                                    exclusive =     exclusive,
                                    ))
                print(f'calc job, submitted timesection: {section_range}')
                walltime_tot += timedelta(hours=int(walltime.split(':')[0]), minutes=int(walltime.split(':')[1]), seconds=int(walltime.split(':')[2]))    
        walltime_tot /= n_jobs 
        # -- concat job from section results --
        if switch.get('concat'):    
            print(f'-- submitting concat job, timeperiod: {time_period} --')
            env_variables = {}
            env_variables["SWITCH_CALC"], env_variables["SWITCH_CONCAT"] = False, True
            env_variables["DASK_ADAPTED"], env_variables["DASK_WORKERS"] = False, False         # don't need that here
            walltime = '0:10:00'
            time_section = gD.get_timesections(n_jobs = 1, time_period = time_period)[0]        # pick out first timesection (which is the whole timeperiod with n_jobs = 1)
            year1_section, month1_section = time_section[0]
            year2_section, month2_section = time_section[-1]
            section_range = f'{year1_section}_{month1_section}-{year2_section}_{month2_section}'
            years_section, months_section = zip(*time_section)
            env_variables["YEAR"] = ':'.join(map(str, years_section))
            env_variables["MONTH"] = ':'.join(map(str, months_section))
            job_ids.append(sJ.submit_job_slurm(python_script = python_script, 
                                folder =        f'{folder_scratch}/oe_files/{strings[2]}', 
                                filename =      f'concat_{strings[3]}_{section_range}', 
                                walltime =      walltime,
                                env_variables = env_variables, 
                                SU_project =    SU_project,
                                exclusive =     exclusive,
                                dependencies =  job_ids,
                                ))
            print(f'submitted concat job,  timesection: {section_range}')
            walltime_tot += timedelta(hours=int(walltime.split(':')[0]), minutes=int(walltime.split(':')[1]), seconds=int(walltime.split(':')[2]))        
    print(f'Number of jobs submitted:   {len(job_ids)}')
    print(f'total walltime:             {walltime_tot} (counting {n_jobs}) simultaneuos jobs')
    print('\nqueue after submission:')
    sJ.check_squeue(username, delay=1.5)


# == when this script is ran / submitted ==
if __name__ == '__main__':
    main()


