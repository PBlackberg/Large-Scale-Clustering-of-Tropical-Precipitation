'''
# ------------------------------------------
#   util_calc - Multiple Linear Regression
# ------------------------------------------

'''

# == imports ==
# -- packages --
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats


# == different MLR assessments ==
# -- variance inflation factor: if explanatory variables are too closely related --
def check_variance_inflation_factor(x_list, show=False):
    x_names = [f'x{i+1}' for i in range(len(x_list))]
    X = pd.DataFrame({name: x for name, x in zip(x_names, x_list)})
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    if show:
        print('VIF assessment:')
        print(vif_data)
    if any(vif > 5 for vif in vif_data['VIF']):
        print('VIF value too large to create y_model')
        print(vif_data)
        print('check input')
        print('exiting')
        exit()
    return vif_data.loc[vif_data['Variable'] == 'x1', 'VIF'].values[0]


# == for 2 predictors ==
def get_pearson_partial_correlation(x_list, y, show = False):
    print('\npartial correlations (Pearson partial correlation):') if show else None
    x1 = x_list[0]
    x2 = x_list[1]
    x1x2_corr,      x1x2_p_value =      pearsonr(x1, x2)
    x1y_corr,       x1y_p_value =       pearsonr(x1, y)
    x2y_corr,       x2y_p_value =       pearsonr(x2, y)
    r_partial_x1 = (x1y_corr - x1x2_corr * x2y_corr) / np.sqrt((1 - x1x2_corr**2) * (1 - x2y_corr**2))
    r_partial_x2 = (x2y_corr - x1x2_corr * x1y_corr) / np.sqrt((1 - x1x2_corr**2) * (1 - x1y_corr**2))
    if show:
        if x1x2_p_value < 0.05:
            print(f'r(x1, x2) =             {x1x2_corr}')
        if x1y_p_value < 0.05:
            print(f'r(x1, y) =              {x1y_corr}')
        if x2y_p_value < 0.05:
            print(f'r(x2, y) =              {x2y_corr}')
        print(f'r_partial(x1, y | x2) = {r_partial_x1}')
        print(f'r_partial(x2, y | x1) = {r_partial_x2}')
    # t-test for partial correlation
    n = len(x1)
    t = r_partial_x2 * np.sqrt((n - 3) / (1 - r_partial_x2**2))
    p_partial_x2 = 2 * (1 - stats.t.cdf(abs(t), df=n - 3))
    return r_partial_x2, p_partial_x2


# == for 2 or more predictors ==
def get_linear_model_components(x_list, y, show = False, standardized = True):
    ''' Give it numpy arrays '''
    print('\nlinear model components (takes any number of predictors)') if show else None
    if standardized:
        x_list = [(x - np.mean(x)) / np.std(x, ddof=1) for x in x_list]
        y = (y - np.mean(y)) / np.std(y, ddof=1)
    X = pd.DataFrame({f'x{i+1}': x for i, x in enumerate(x_list)})                                  # linear model package needs dataframe
    y = pd.DataFrame({'y': y})        
    model = sm.OLS(y, X).fit()
    coeffs = [beta_i for beta_i in model.params]
    y_hat = 0
    for i, (beta, x) in enumerate(zip(coeffs, x_list)):
        y_hat += beta * x
    residual = y['y'].values - y_hat
    if show:
        print('linear model constants')
        [print(f'b_{i+1} =          {b}') for i, b in enumerate(coeffs)]
        y_hat_y_corr, y_hat_y_p_value = pearsonr(y_hat, y['y'])
        if y_hat_y_p_value < 0.05:
            print(f'r(y_hat, y) =  {y_hat_y_corr}')
    return y_hat, coeffs, residual




 