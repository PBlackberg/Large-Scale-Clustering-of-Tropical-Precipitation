'''
# -------------------------
#    Model institutes
# -------------------------
'''



# -------------------------------------------------------------------------  CMIP5 ----------------------------------------------------------------------------------------------------- #
institutes_cmip5 = {
    'IPSL-CM5A-MR': 'IPSL', 
    'GFDL-CM3':     'NOAA-GFDL', 
    'GISS-E2-H':    'NASA-GISS', 
    'bcc-csm1-1':   'BCC',
    'CNRM-CM5':     'CNRM-CERFACS', 
    'CCSM4':        'NCAR', 
    'HadGEM2-AO':   'NIMR-KMA', 
    'BNU-ESM':      'BNU',
    'EC-EARTH':     'ICHEC', 
    'FGOALS-g2':    'LASG-CESS', 
    'MPI-ESM-MR':   'MPI-M', 
    'CMCC-CM':      'CMCC',
    'inmcm4':       'INM',
    'NorESM1-M':    'NCC',
    'CanESM2':      'CCCma', 
    'MIROC5':       'MIROC', 
    'HadGEM2-CC':   'MOHC',
    'MRI-CGCM3':    'MRI', 
    'CESM1-BGC':    'NSF-DOE-NCAR'
    }


# -------------------------------------------------------------------------  CMIP6 ----------------------------------------------------------------------------------------------------- #
institutes_cmip6 = {
    'TaiESM1':          'AS-RCEC', 
    'BCC-CSM2-MR':      'BCC', 
    'FGOALS-g3':        'CAS', 
    'FGOALS-f3-L':      'CAS',
    'CanESM5':          'CCCma', 
    'CMCC-ESM2':        'CMCC', 
    'CMCC-CM2-SR5':     'CMCC', 
    'CNRM-CM6-1':       'CNRM-CERFACS',
    'CNRM-CM6-1-HR':    'CNRM-CERFACS', 
    'CNRM-ESM2-1':      'CNRM-CERFACS', 
    'MIROC6':           'MIROC', 
    'MIROC-ES2L':       'MIROC',
    'MPI-ESM1-2-LR':    'MPI-M', 
    'GISS-E2-1-H':      'NASA-GISS', 
    'GISS-E2-1-G':      'NASA-GISS', 
    'NorESM2-MM':       'NCC',
    'NorESM2-LM':       'NCC', 
    'GFDL-CM4':         'NOAA-GFDL', 
    'GFDL-ESM4':        'NOAA-GFDL', 
    'UKESM1-0-LL':      'MOHC', 
    'KACE-1-0-G':       'NIMS-KMA', 
    'MRI-ESM2-0':       'MRI', 
    'CESM2':            'NCAR', 
    'CESM2-WACCM':      'NCAR', 
    'NESM3':            'NUIST',
    'IITM-ESM':         'CCCR-IITM', 
    'EC-Earth3':        'EC-Earth-Consortium', 
    'INM-CM5-0':        'INM', 
    'INM-CM4-8':        'INM',
    'IPSL-CM6A-LR':     'IPSL', 
    'KIOST-ESM':        'KIOST', 
    'ACCESS-ESM1-5':    'CSIRO', 
    'ACCESS-CM2':       'CSIRO-ARCCSS'
    }

institutes = {**institutes_cmip5, **institutes_cmip6}