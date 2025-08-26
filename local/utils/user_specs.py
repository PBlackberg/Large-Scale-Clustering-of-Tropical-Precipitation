'''
# -----------------
#   user_specs
# -----------------
Put in your own paths to temporarily saved data (scratch) and data that is used for a longer time (work):
ex:
/Users/cbla0002/Desktop/work
/Users/cbla0002/Desktop/scratch
'''

# == imports ==
# -- packages --
import os


# == get user ==
def get_user_specs(show = False):
    username =          os.path.expanduser("~").split('/')[-1]                          # ex: 'cb4968'
    storage_project =   ''                                                              # storage project
    SU_project =        ''                                                              # resource project
    data_projects =     ()                                                              # directories available for job
    folder_scratch =    (f'/Users/{username}/Desktop/scratch')                          # temp files    (change this as needed)
    folder_work =       (f'/Users/{username}/Desktop/work')                             # saved         (change this as needed)
    if show:
        print('user specs:')
        [print(f) for f in [username, folder_scratch, folder_work]]
    return folder_work, folder_scratch, SU_project, storage_project, data_projects


# == when this script is ran ==
if __name__ == '__main__':
    output = get_user_specs()
    folder_work, folder_scratch, SU_project, storage_project, data_projects = output
    [print(f) for f in output]

