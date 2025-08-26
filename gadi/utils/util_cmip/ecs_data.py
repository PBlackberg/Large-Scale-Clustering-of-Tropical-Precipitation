'''
# -------------------------
#  List of ECS from models
# -------------------------
ECS - Equilibrium Climate Sensitivity
# ECS taken from supplementary information at:
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085782#pane-pcw-figures
# https://www.nature.com/articles/d41586-022-01192-2 # most taken from here (used the column which had the same value for models existing in both sets)
'''



# -------------------------------------------------------------------------  CMIP5 ----------------------------------------------------------------------------------------------------- #
ecs_cmip5 = {
    'IPSL-CM5A-MR': 0, 
    'GFDL-CM3':     0, 
    'GISS-E2-H':    0, 
    'bcc-csm1-1':   0, 
    'CNRM-CM5':     0, 
    'CCSM4':        0, 
    'HadGEM2-AO':   0,        
    'BNU-ESM':      0, 
    'EC-EARTH':     0, 
    'FGOALS-g2':    0, 
    'MPI-ESM-MR':   0, 
    'CMCC-CM':      0, 
    'inmcm4':       0, 
    'NorESM1-M':    0, 
    'CanESM2':      0,           
    'MIROC5':       0, 
    'HadGEM2-CC':   0, 
    'MRI-CGCM3':    0, 
    'CESM1-BGC':    0
    }


# -------------------------------------------------------------------------  CMIP6 ----------------------------------------------------------------------------------------------------- #
ecs_cmip6 = {
    'INM-CM4-8':        1.83, 
    'INM-CM5-0':        1.92, 
    'IITM-ESM':         2.37, 
    'NorESM2-MM':       2.49, 
    'NorESM2-LM':       2.56, 
    'MIROC6':           2.6,
    'GFDL-ESM4':        2.65, 
    'MIROC-ES2L':       2.66, 
    'FGOALS-g3':        2.87, 
    'MPI-ESM1-2-LR':    3.02,
    'BCC-CSM2-MR':      3.02, 
    'MRI-ESM2-0':       3.13,  
    'KIOST-ESM':        3.36, 
    'CMCC-CM2-SR5':     3.56, 
    'CMCC-ESM2':        3.58, 
    'CMCC-ESM2':        3.58, 
    'ACCESS-ESM1-5':    3.88, 
    'GFDL-CM4':         3.89, 
    'EC-Earth3':        4.26, 
    'CNRM-CM6-1-HR':    4.34, 
    'TaiESM1':          4.36, 
    'ACCESS-CM2':       4.66, 
    'CESM2-WACCM':      4.68, 
    'IPSL-CM6A-LR':     4.70,   
    'NESM3':            4.72, 
    'KACE-1-0-G':       4.75, 
    'CNRM-ESM2-1':      4.79, 
    'CNRM-CM6-1':       4.90, 
    'CESM2':            5.15, 
    'UKESM1-0-LL':      5.36, 
    'CanESM5':          5.64
    }
ecs_list = {**ecs_cmip5, **ecs_cmip6}


