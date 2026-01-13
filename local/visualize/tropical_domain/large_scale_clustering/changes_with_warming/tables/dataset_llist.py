'''
# ---------------------------
#   list plotted in figure
# ---------------------------

'''

# == imports ==
# -- Packages --
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.stats import pearsonr

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
mS = import_relative_module('user_specs',                                           'utils')
cL = import_relative_module('util_cmip.model_letter_connection',                    'utils')


def plot():
    # -- create figure --    
    # width, height = 6.27, 9.69                                                                                                  # max size (for 1 inch margins)
    width, height = 8.5, 5                                                                                                        # max: 15.9, 24.5 for 1 inch margins [cm]
    width, height = [f / 2.54 for f in [width, height]]                                                                         # function takes inches
    ncols, nrows  = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize = (width, height))
    ax.axis('off')

    # -- model list text --
    x_position = 0.01 # 0.57
    y_position = 0.95 # 0.365
    for i, model in enumerate(cL.get_model_letters()):
        if model == 'ACCESS-ESM1-5_':
            fig.text(x_position, y_position, f"{cL.get_model_letters()[model]} - {model}", fontsize = 7.5, ha = 'left', va = 'top', fontweight = 'bold')
        else:
            fig.text(x_position, y_position, f"{cL.get_model_letters()[model]} - {model}", fontsize = 8, ha = 'left', va = 'top')
        y_position -= 0.075
        if i == 9:
            x_position = 0.34
            y_position = 0.95
        if i == 19:
            x_position = 0.69
            y_position = 0.95

    # -- obs text --
    x_position = 0.6875
    y_position = 0.425
    text = fig.text(x_position, y_position,                                             # data
                    '\u2605',                                                           # star                                   
                    color = 'w', ha='left', va='top',                                   #
                    fontsize = 8, fontweight='bold')                                    # 
    text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='g')])     # background effects
    text = fig.text(x_position + 0.035, y_position,                                     # data
                    '- GPCP',                                                           # star                                   
                    color = 'g', ha='left', va='top',                                   #
                    fontsize = 7.5)                                                     #
    
    # -- IFS text --
    x_position = 0.7025
    y_position = 0.362
    text = fig.text(x_position - 0.0005, y_position - 0.0575,                                             # data
                    '\u25C6',                                                                   # diamond                                   
                    color = 'w', ha='center', va='bottom',                                      #
                    fontsize = 7, fontweight='bold')                                            #
    text.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='purple')])        # background effects
    text = fig.text(x_position + 0.02, y_position - 0.01,                                      # data
                    '- IFS_9_FESOM_5',                                                          # star                                   
                    color = 'purple', ha='left', va='top',                                      #   
                    fontsize = 7.5)                                                             #
    return fig


def main():
    # -- plot --
    list_type = 'model_obs_ifs'
    fig = plot()

    # -- save figure --
    folder_work, folder_scratch, SU_project, storage_project, data_projects = mS.get_user_specs(show = False)
    filename = f'{Path(__file__).stem}'
    folder = f'{folder_scratch}/{Path(__file__).parents[2].name}/{Path(__file__).parents[1].name}/{Path(__file__).parents[0].name}/{filename}'
    plot_name = f'list_{list_type}'
    path = f'{folder}/{plot_name}.svg'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi = 150)
    print(f'plot saved at: {path}')
    plt.close(fig)




if __name__ == '__main__':
    main()
