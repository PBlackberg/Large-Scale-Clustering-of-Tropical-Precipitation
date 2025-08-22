'''
# -----------------
#   user_specs
# -----------------
Common directories
project:    k10                     (standard)
scratch:    /scratch/k10/cb4968    
work:       /g/data/k10/cb4968

'''

# == imports ==
# -- packages --
import os


# == get user ==
def get_user_specs(show = False):
    username =          os.path.expanduser("~").split('/')[-1]                          # ex; 'cb4968'    
    # -- project for saved metrics --
    storage_project =   (
        'k10',
        # 'nf33',
        )[0] 
    folder_work =       (f'/g/data/{storage_project}/{username}')                       # metrics saved here

    # -- project for compute resources --
    SU_project =        (
        'if69',
        # 'gb02',
        # 'nf33',
        )[0]
    folder_scratch =    (f'/scratch/{storage_project}/{username}')                      # temp files and job output files go here
    
    # -- project for data saved on nci --    
    data_projects =     (
        'hh5',                                                                          # conda environment
        # 'gb02',                                                                       # general 21CW compute resources
        'if69',                                                                         # 21CW: Weather System Dynamics, Variability & Warmer World project
        'k10',                                                                          # my saved metrics
        'al33',                                                                         # CMIP5
        'oi10', 'fs38',                                                                 # CMIP6   
        'ia39',                                                                         # GPCP, IMERG
        'rt52',                                                                         # ERA5
        'xp65', 'qx55',                                                                 # Hackathon: storage, environment, high-res models ('nf33' no longer available)
        )
    if show:
        print('user specs:')
        [print(f) for f in [folder_work, folder_scratch, SU_project, storage_project, data_projects, username]]
    return folder_work, folder_scratch, SU_project, storage_project, data_projects


# == when this script is ran ==
if __name__ == '__main__':
    output = get_user_specs(show = True)
    print('\n' * 2)
    folder_work, folder_scratch, SU_project, storage_project, data_projects = output
    [print(f) for f in output]


