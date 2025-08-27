'''
# -----------------
#  save_and_load
# -----------------

'''

# == imports ==
# -- Packages --
import os
import re
from pathlib import Path


# == handle folders ==
def create_folder(directory, name):
    ''' Create folder if not already existing '''
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f'{name} created at: {directory}')
    else:
        pass
    return directory

def create_plot_folder(path):
    ''' cwd folder for plots '''
    plot_folder = f'{os.path.dirname(os.path.dirname(path))}/plots'
    plot_folder = create_folder(directory = plot_folder, name = 'cwd_plot folder')    
    return plot_folder


# == hadnle files ==
def get_temp_data_path(folder_scratch, sub_folder, file_name_base, year, month):
    temp_data_folder = create_folder(directory = f'{folder_scratch}/temp_data/{sub_folder}', name = 'temp_data folder')
    temp_data_file = f'{file_name_base}_year_{year}_month_{month}.nc'
    return f'{temp_data_folder}/{temp_data_file}' 

def save_temp_calc(folder_scratch, sub_folder, filename, ds):
    temp_data_folder = create_folder(directory = f'{folder_scratch}/temp_calc/{sub_folder}', name = 'temp_calc folder')
    path = f'{temp_data_folder}/{filename}'
    ds.to_netcdf(path, mode="w")
    print(f'temp calc saved at: {path}')

def find_tempfiles(temp_calc_folder, date_range, rfilename):
    start_year, start_month =   map(int, date_range.split('-')[0].split('_'))
    end_year, end_month =       map(int, date_range.split('-')[1].split('_'))
    pattern = re.compile(r'(.+)_yearend_(\d+)_monthend_(\d+)\.nc')
    temp_files = []
    for root, _, files in os.walk(temp_calc_folder):
        for file in files:
            if file.endswith(".nc") and rfilename in file:
                match = pattern.search(file)
                if match:
                    file_name, year, month = match.groups()                 # if different year sections of the same metric is run, only pick out the requested years
                    year, month = int(year), int(month)                      
                    if (start_year < year < end_year or
                        (year == start_year and month >= start_month) or           
                        (year == end_year and month <= end_month)):
                        full_path = os.path.join(root, file)
                        temp_files.append((year, month, full_path))
    temp_files = [file[2] for file in sorted(temp_files, key=lambda x: (x[0], x[1]))]
    return temp_files

def remove_tempfiles(temp_files):
    print('removing temp files')    
    for path_temp in temp_files:
        os.remove(path_temp)
        
def save_metric_result(work_folder, sub_folder, filename, ds):
    temp_data_folder = create_folder(directory = f'{work_folder}/metric/{sub_folder}', name = 'metric folder')
    path = f'{temp_data_folder}/{filename}'
    ds.to_netcdf(path, mode="w")
    print(f'metric saved at: {path}')

def save_data_result(work_folder, sub_folder, filename, ds):
    data_folder = create_folder(directory = f'{work_folder}/data/{sub_folder}', name = 'data folder')
    path = f'{data_folder}/{filename}'
    ds.to_netcdf(path, mode="w")
    print(f'data saved at: {path}')
