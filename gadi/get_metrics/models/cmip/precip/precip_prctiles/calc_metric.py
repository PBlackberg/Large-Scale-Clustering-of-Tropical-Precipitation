'''
# -----------------
#   Calc_metric
# -----------------

'''

# == imports ==
import xarray as xr
import numpy as np
from pathlib import Path


# == calculate metric ==
def calculate_metric(data_objects):
    # -- check data --
    da, lon_area, lat_area = data_objects
    da = da.sel(lon = slice(int(lon_area.split(':')[0]), int(lon_area.split(':')[1])), 
                lat = slice(int(lat_area.split(':')[0]), int(lat_area.split(':')[1]))
                )

    # -- calculate metric --
    quantile_thresholds = [
        0.5,                           
        0.9,                   
        0.95,                          # 5% of the domain 
        0.97,  
        0.99      
        ]
    quantiles = da.quantile(quantile_thresholds, dim = ('lat', 'lon'))

    # -- put in xr.Dataset --
    metric_name = Path(__file__).resolve().parents[0].name
    ds = xr.Dataset()
    for idx, quant in enumerate(quantile_thresholds):
        name = f'{metric_name}_{int(quant * 100)}'
        ds[name] = quantiles[idx]
        
    return ds



