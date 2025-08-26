'''
# ------------
#  util_calc
# ------------

'''

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import cftime
import warnings


def calculate_correlation_and_significance(da_1d, da_3d, nb_threshold = 1, zero2NaN = False):
    ''' Get correlation between tropics-wide metric and the timeseries of each gridbox '''
    if 'time' in da_3d.dims:
        if isinstance(da_3d.time.values[0], cftime.DatetimeGregorian) and not isinstance(da_1d.time.values[0], cftime.DatetimeGregorian):
            da_3d['time'] = da_1d['time']
    corr_coeff = np.empty((len(da_3d.lat), len(da_3d.lon)))
    significance_mask = np.zeros_like(corr_coeff, dtype=bool)
    regression_coeff = np.empty((len(da_3d.lat), len(da_3d.lon)))
    for i, _ in enumerate(da_3d.lat):
        for j, _ in enumerate(da_3d.lon):
            gridbox_timeseries = da_3d[:, i, j]
            if zero2NaN:                                        # if a lot of zeros hide correlation, turn the zeros to NaN first (case specific)
                gridbox_timeseries = gridbox_timeseries.where(gridbox_timeseries != 0) 
                da_1d_loop = da_1d.where(gridbox_timeseries != 0) 
            else:
                da_1d_loop = da_1d
            non_nan_mask = ~np.isnan(gridbox_timeseries)        # only correlate where there are non-NaN values (function cannot accept nan)
            clean_da_1d = da_1d_loop[non_nan_mask]
            clean_gridbox_timeseries = gridbox_timeseries[non_nan_mask]
            if clean_gridbox_timeseries.size > nb_threshold:    # threshold on number of datapoints for correlation to be calculated
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        corr, p_value = pearsonr(clean_da_1d, clean_gridbox_timeseries)
                        slope = np.cov(clean_da_1d, clean_gridbox_timeseries)[0,1] / np.var(clean_da_1d)
                    except Warning:                             # if all values in the timeseries is zero it will give a warning
                        corr, p_value, slope = np.nan, 1.0, np.nan
                corr_coeff[i, j] = corr
                significance_mask[i, j] = p_value < 0.05
                regression_coeff[i, j] = slope
            else:                                               # no correlation if not enough datapoints
                corr_coeff[i, j] = np.nan                       
                significance_mask[i, j] = False
                regression_coeff[i, j] = np.nan
    corr_da = xr.DataArray(corr_coeff, coords=[da_3d.lat, da_3d.lon], dims=["lat", "lon"])
    significance_mask = xr.DataArray(significance_mask, coords=[da_3d.lat, da_3d.lon], dims=["lat", "lon"])
    regression_coeff = xr.DataArray(regression_coeff, coords=[da_3d.lat, da_3d.lon], dims=["lat", "lon"])
    return corr_da, significance_mask, regression_coeff


# == when this script is ran ==
if __name__ == '__main__':
    print('executes')
    