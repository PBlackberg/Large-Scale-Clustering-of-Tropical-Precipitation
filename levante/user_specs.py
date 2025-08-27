'''
# -----------------
#   my_settings
# -----------------
Specify levante username and projects, ex:
SU project: 'bb1153'
usename:    b382628
scratch:    
/scratch/b/b382628
work:       
/work/bb1153/b382628
'''

import os

def get_user_specs(show = False):
    username = os.path.expanduser("~").split('/')[-1]                                                                             
    SU_project = 'bb1153'                                          
    folder_scratch = (f'/scratch/b/{username}')     # /scratch/b/b382628   
    if show:
        print(f'''User specs:
0. username:        {username}
1. project:         {SU_project}
2. scratch_folder:  {folder_scratch}
              ''')                       
    return username, SU_project, folder_scratch

def get_work_folder():
    username, SU_project, folder_scratch = get_user_specs(show = False)
    return f'/work/{SU_project}/{username}'  # /work/bb1153/b382628

  
if __name__ == '__main__':
    username, SU_project, folder_scratch = get_user_specs()
    [print(f) for f in [username, SU_project, folder_scratch]]




