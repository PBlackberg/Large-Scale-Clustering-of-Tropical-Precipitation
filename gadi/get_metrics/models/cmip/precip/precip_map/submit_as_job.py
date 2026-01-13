'''
# -----------------
#  Submit_as_job
# -----------------

'''

# == Imports ==
# -- Packages --
import os
import sys
import importlib
from datetime import timedelta
from pathlib import Path
import numpy as np
import itertools

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
sJ = import_relative_module('util_qsub.submission_funcs',   'utils')


# == Set specs ==
def set_specs():
    datasets = (                                                                                                                    # Models ordered by change in temperature with warming    
        'INM-CM5-0',                                                                                                                # 1
        'IITM-ESM',                                                                                                                 # 2   
        'FGOALS-g3',                                                                                                                # 3    
        'INM-CM4-8',                                                                                                                # 4                                
        'MIROC6',                                                                                                                   # 5                                      
        'MPI-ESM1-2-LR',                                                                                                            # 6                         
        # 'KIOST-ESM',                                                                                                              # 7
        'BCC-CSM2-MR',                                                                                                              # 8           
        'GFDL-ESM4',                                                                                                                # 9         
        'MIROC-ES2L',                                                                                                               # 10 
        'NorESM2-LM',                                                                                                               # 11      
        # 'NorESM2-MM',                                                                                                             # 12                      
        'MRI-ESM2-0',                                                                                                               # 13                            
        'GFDL-CM4',                                                                                                                 # 14      
        'CMCC-CM2-SR5',                                                                                                             # 15                
        'CMCC-ESM2',                                                                                                                # 16                                    
        'NESM3',                                                                                                                    # 17     
        'ACCESS-ESM1-5',                                                                                                            # 18 
        'CNRM-ESM2-1',                                                                                                              # 19 
        'EC-Earth3',                                                                                                                # 20 
        'CNRM-CM6-1',                                                                                                               # 21
        'CNRM-CM6-1-HR',                                                                                                            # 22 
        'KACE-1-0-G',                                                                                                               # 23            
        'IPSL-CM6A-LR',                                                                                                             # 24
        'ACCESS-CM2',                                                                                                               # 25 
        'TaiESM1',                                                                                                                  # 26                      
        'CESM2-WACCM',                                                                                                              # 27   
        'CanESM5',                                                                                                                  # 28  
        'UKESM1-0-LL',                                                                                                              # 29
        )                                                                                                                           #
    t_freqs = (                                                                                                                     #
        # 'hrly',                                                                                                                   # animations
        # '3hrly',                                                                                                                  # 
        'daily',                                                                                                                    #
        # 'monthly',                                                                                                                #
        )                                                                                                                           #
    lon_areas = (                                                                                                                   # set lon extent
        '0:360',                                                                                                                    # Full domain
        # '0:49',                                                                                                                   # Africa
        # '50:99',                                                                                                                  # Indian Ocean
        # '100:149',                                                                                                                # Maritime Continent
        # '150:204',                                                                                                                # West / central Pacific    (55 degrees, 5 degrees wider)
        # '205:259',                                                                                                                # East Pacific              (55 degrees, 5 degrees wider)
        # '260:309',                                                                                                                # Amazon
        # '310:359',                                                                                                                # Atlantic
        )                                                                                                                           #
    lat_areas = (                                                                                                                   # set lat extent (can be looped)
        '-30:30',                                                                                                                   # Tropics
        # '-20:20',                                                                                                                 # Central tropics
        # '-10:10',                                                                                                                 # Equator
        )                                                                                                                           #
    resolutions = (                                                                                                                 #
        2.8,                                                                                                                        # CMIP, lowest common            
        )                                                                                                                           #
    time_periods = (                                                                                                                # time_periods for metric (can be looped)
        # '1970-01:1970-01',                                                                                                        # test
        '1970-01:1999-12',                                                                                                          # historical
        '2070-01:2099-12',                                                                                                          # warm
        )                                                                                                                           #
    return datasets, t_freqs, lon_areas, lat_areas, resolutions, time_periods

def get_timesections(n_jobs, time_period):
    year1, month1 = map(int, time_period.split(':')[0].split('-'))                                                                  #
    year2, month2 = map(int, time_period.split(':')[1].split('-'))                                                                  #
    timesteps = [(year, month) for year in range(int(year1), int(year2) + 1) for month in range(1, 13)                              # year, month pair
                 if not (year == year1 and month < month1) and not (year == year2 and month > month2)]                              # clipping months of first and last year
    time_sections = np.array_split(timesteps, n_jobs)                                                                               #
    return time_sections

def get_path(dataset, t_freq, lon_area, lat_area, resolution, time_period):
    folder_0 = Path(__file__).resolve().parents[3].name                                                                             # ex: models
    folder_1 = Path(__file__).resolve().parents[2].name                                                                             # ex: cmip
    folder_2 = Path(__file__).resolve().parents[1].name                                                                             # ex: metric_group
    folder_3 = Path(__file__).resolve().parents[0].name                                                                             # ex: metric
    folder_4 = dataset                                                                                                              # ex: dataset
    # [print(f) for f in [t_folder, mg_folder, dg_folder, m_folder]]                                                                #
    # exit()                                                                                                                        #
    r_folder        = f'{folder_0}/{folder_1}/{folder_2}/{folder_3}/{folder_4}'                                                     # result_folder dir
    r_filename      = (                                                                                                             # base result_filename
                      f'{folder_3}'   
                      f'_{dataset}'                                                                                                 #
                      f'_{t_freq}'                                                                                                  #
                      f'_{lon_area.split(":")[0]}-{lon_area.split(":")[1]}'                                                         #
                      f'_{lat_area.split(":")[0]}-{lat_area.split(":")[1]}'                                                         #
                      f'_{int(360/resolution)}x{int(180/resolution)}'                                                               #
                      f'_{time_period.split(":")[0]}_{time_period.split(":")[1]}'                                                   #
                      )                                                                                                             #
    return r_folder, r_filename


# == main func ==
def main():
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = True)                        # user settings
    switch =        {'calc': True, 'concat': True}                                                                                  # calculate section and then concat
    datasets, t_freqs, lon_areas, lat_areas, resolutions, time_periods = set_specs()                                                # all specs
    job_ids_concat = []                                                                                                             # make next spec combination dependent on completion of previous combination
    n_jobs_calc_tot = 0                                                                                                             #
    last_job = len(list(itertools.product(t_freqs, lat_areas, lon_areas, resolutions, datasets, time_periods))) - 1                 # send email for this
    for i, (t, lat, lon, r, d, p) in enumerate(itertools.product(t_freqs, 
                                                                 lat_areas, 
                                                                 lon_areas, 
                                                                 resolutions, 
                                                                 datasets, time_periods)):                                          # loops over all specs (looped in input order)
        print(f'\njob group {i}')
        [print(f) for f in [d, t, lon, lat, r, p]]
        r_folder, r_filename = get_path(d, t, lon, lat, r, p)
        # -- clear temp calc from associated folder --                                                                              #
        folder = f'{folder_scratch}/temo_calc/{r_folder}/{r_filename}'                                                              #
        os.makedirs(folder, exist_ok=True)                                                                                          #
        temp_files = [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.nc')]                                             # clearing the folder that is filled with partial results
        [os.remove(path_temp) for path_temp in temp_files]                                                                          #
        # -- job resources (calc) --
        n_jobs_calc = 1  
        walltime_calc = '0:30:00'                                                                                                   # job time for complete time_period
        mem_calc = '75GB'                                                                                                           #
        ncpus_calc = 1                                                                                                              # if parallelizing, do it on the months
        # -- divide time_period calc, in n sections -- 
        job_ids_calc = []                                                                                                           # make concat dependent on calc job completion
        if switch.get('calc'):    
            # print(f'\n-- submitting calc jobs, timeperiod: {p} --')
            walltime_calc = timedelta(hours=int(walltime_calc.split(':')[0]),                                                       #
                                        minutes=int(walltime_calc.split(':')[1])) / n_jobs_calc                                     # time for each job (max time given)
            env_variables = {}                                                                                                      # sJ.check_env_variables(env_variables)
            env_variables['DATASET'] =          d
            env_variables['T_FREQ'] =           t
            env_variables['LON_AREA'] =         lon
            env_variables['LAT_AREA'] =         lat
            env_variables['RESOLUTION'] =       r
            env_variables["TIME_PERIOD"] =      p
            env_variables['R_FOLDER'] =         r_folder
            env_variables['R_FILENAME'] =       r_filename
            env_variables["SWITCH_CALC"] =      True
            env_variables["SWITCH_CONCAT"] =    False
            for j, time_section in enumerate(get_timesections(n_jobs_calc, p)):                                                     # each time section becomes a job    
                year1_section, month1_section = time_section[0]
                year2_section, month2_section = time_section[-1]
                section_range =  f'{year1_section}_{month1_section}-{year2_section}_{month2_section}'   
                print(f'calc job, timesection: {section_range}')
                years_section, months_section = zip(*time_section)
                env_variables["YEAR"] = ':'.join(map(str, years_section))                                                           # year, month pair: year
                env_variables["MONTH"] = ':'.join(map(str, months_section))                                                         # year, month pair: month
                env_variables["SECTION_RANGE"] = section_range                                                                      # year, month pair: month
                email = 'b' if i == 0 and j == 0 else ''                                                                            # send email when first job starts
                job_ids_calc.append(sJ.submit_job(python_script = f'{os.path.dirname(__file__)}/main_func.py',                      # the main_func.py script in this folder 
                                            folder =            f'{folder_scratch}/oe_files/{r_folder}',                            # folder for terminal output
                                            filename =          f'{r_filename}_{section_range}',                                    # file for terminal output                
                                            walltime =          str(walltime_calc),                                                 # max job length
                                            mem =               mem_calc,                                                           # max mem
                                            ncpus =             str(ncpus_calc),                                                    # max cpus
                                            env_variables =     env_variables,                                                      # job specs                                                                
                                            SU_project =        SU_project,                                                         # resource project
                                            data_projects =     data_projects,                                                      # available directories
                                            scratch_project =   storage_project,                                                    # storage options
                                            # dependencies =      job_ids_concat,                                                   # job start dependency
                                            email =             email                                                               # email about job status
                                            ))
        n_jobs_calc_tot += len(job_ids_calc)
        # -- concat job from section results --
        if switch.get('concat'):                                                                                                    # concatenates result from calc
            # print(f'-- submitting concat job, timeperiod: {p} --')
            walltime_concat = '0:10:00'                                                                                             #
            mem_concat = '20GB'
            ncpus_concat = 1
            walltime_concat = timedelta(hours=int(walltime_concat.split(':')[0]), minutes=int(walltime_concat.split(':')[1]))       # time for each job (max time given)
            env_variables = {}                                                                                                      # sJ.check_env_variables(env_variables)
            env_variables['DATASET'] =          d
            env_variables['T_FREQ'] =           t
            env_variables['LON_AREA'] =         lon
            env_variables['LAT_AREA'] =         lat
            env_variables['RESOLUTION'] =       r
            env_variables['TIME_PERIOD'] =      p
            env_variables['R_FOLDER'] =         r_folder
            env_variables['R_FILENAME'] =       r_filename
            env_variables["SWITCH_CALC"] =      False
            env_variables["SWITCH_CONCAT"] =    True
            time_section = get_timesections(n_jobs = 1, time_period = p)[0]                                                         # Whole timeperiod for concatenation
            year1_section, month1_section = time_section[0]
            year2_section, month2_section = time_section[-1]
            section_range = f'{year1_section}_{month1_section}-{year2_section}_{month2_section}'
            print(f'concat job, timesection: {section_range}')
            years_section, months_section = zip(*time_section)
            env_variables["YEAR"] = ':'.join(map(str, years_section))
            env_variables["MONTH"] = ':'.join(map(str, months_section))
            email = 'ae' if i == last_job else ''                                                                                   # send email when last job finishes (or fails) (will fail if previous jobs fail)
            job_ids_concat.append(sJ.submit_job(python_script = f'{os.path.dirname(__file__)}/main_func.py',                        # the main_func.py script in this folder 
                                        folder =            f'{folder_scratch}/oe_files/{r_folder}',                                # folder for terminal output
                                        filename =          f'{r_filename}_concat_{section_range}',                                 # file for terminal output                
                                        walltime =          str(walltime_concat),                                                   # max job length
                                        mem =               mem_concat,                                                             # max mem
                                        ncpus =             str(ncpus_concat),                                                      # max cpus
                                        env_variables =     env_variables,                                                          # job specs                                                                
                                        SU_project =        SU_project,                                                             # resource project
                                        data_projects =     data_projects,                                                          # available directories
                                        scratch_project =   storage_project,                                                        # storage options
                                        dependencies =      job_ids_calc,                                                           # job start dependency
                                        email =             email                                                                   # email about job status
                                        ))                                                                                          #   
    n_job_groups = (n_jobs_calc_tot + len(job_ids_concat)) // 250 + 1                                                               # think it's like 250 simultaneous jobs
    walltime_eff = (walltime_calc + walltime_concat) * n_job_groups
    print(f'\nSubmitted:        {(n_jobs_calc_tot + len(job_ids_concat))} jobs')
    print(f'Estimated max time: {str(walltime_eff)} hh:mm:ss')


# == when this script is ran / submitted ==
if __name__ == '__main__':
    main()
    # print('job queue before submission:')
    # sJ.check_qstat(username, delay=0)
    # print('\nqueue after submission:')
    # sJ.check_squeue(username, delay=1.5)

    # -- sequential metric combintation submission (for many timestep splits) --
    # print(f'\nSubmitted:                {(n_jobs_calc_tot + len(job_ids_concat))} jobs')
    # print(f'Split into:                 {len(job_ids_concat)} job groups, with {n_jobs_calc + 1} jobs per group')
    # print(f'Estimated time:             {str(walltime_eff)} hh:mm:ss    , with max {str(walltime_eff / (len(job_ids_concat)))} hh:mm:ss per group')
    # walltime_eff += walltime_concat                                                                                         # accumulated time (separate as dependent)    
    # walltime_eff += walltime_calc                                                                                           # accumulated effective time
    # walltime_eff = timedelta()                                                                                                      # effective time (after parallelization)
