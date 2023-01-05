import pandas as pd
import numpy as np
from pytsproc.utils import add_water_year

def ML1_12(Q, month, mean_or_median='mean'):
    """ Mean (or median) minimum flows for each month
        across all years. Compute the minimums for each month over the entire flow
        record. For example, ML1 is the mean of the minimums of all January flow
        values over the entire record

    Args:
        Q (pandas DataFrame): Q data with time index
        month (int): month to evaluate
        mean_or_median (str): 'mean' or 'median', indicating which statistic to calculate.

    Returns:
        
    """
    monthly = Q.loc[Q.index.month==month]
    if mean_or_median.lower() == 'mean':
        return monthly.groupby(monthly.index.year)['discharge'].min().mean()
    elif mean_or_median.lower() == 'median':
        return monthly.groupby(monthly.index.year)['discharge'].min().mean()
    else:
        raise(f'{mean_or_median} must be "mean" or "median')


def ML13(Q):
    """ Variability (coefficient of variation)across minimum monthly flow values. 
        Compute the mean and standard deviation for the minimum monthly 
        flows over the entire flow record. 
        ML13 is the standard deviation times 100 divided by the 
        mean minimum monthly flow for all years (percent).
    Args:
        Q (pandas DataFrame): Q data with time index
    """
    m = Q.resample('M')['discharge'].min().mean()
    s = Q.resample('M')['discharge'].min().std()
    return s * 100. / m

def ML14(Q, wy=False):
    """ Compute the minimum annual flow for each year. 
    ML14 is the mean of the ratios of minimum annual flows to the 
    median flow for each year
    Args:
        Q (pandas DataFrame): Q data with time index
        wy (Bool): True if aggregate by water year, False if calendar year.
    """
    if wy:
        Q = add_water_year(Q)
        minflow = Q.groupby(Q.water_year)['discharge'].min()
        medflow = Q.groupby(Q.water_year)['discharge'].median()
    else:
        minflow = Q.groupby(Q.index.year)['discharge'].min()
        medflow = Q.groupby(Q.index.year)['discharge'].median()

    return np.mean(minflow/medflow)

def ML15(Q, wy=False):
    """ Compute the minimum annual flow for each year. 
    ML14 is the mean of the ratios of minimum annual flows to the 
    mean flow for each year
    Args:
        Q (pandas DataFrame): Q data with time index
        wy (Bool): True if aggregate by water year, False if calendar year.
    """
    if wy:
        Q = add_water_year(Q)
        minflow = Q.groupby(Q.water_year)['discharge'].min()
        medflow = Q.groupby(Q.water_year)['discharge'].mean()
    else:
        minflow = Q.groupby(Q.index.year)['discharge'].min()
        medflow = Q.groupby(Q.index.year)['discharge'].mean()

    return np.mean(minflow/medflow)

def ML16(Q, wy=False):
    """ Compute the minimum annual flow for each year. 
    ML14 is the median of the ratios of minimum annual flows to the 
    median flow for each year
    Args:
        Q (pandas DataFrame): Q data with time index
        wy (Bool): True if aggregate by water year, False if calendar year.
    """
    if wy:
        Q = add_water_year(Q)
        minflow = Q.groupby(Q.water_year)['discharge'].min()
        medflow = Q.groupby(Q.water_year)['discharge'].median()
    else:
        minflow = Q.groupby(Q.index.year)['discharge'].min()
        medflow = Q.groupby(Q.index.year)['discharge'].median()

    return np.median(minflow/medflow)
    
def ML17(Q, mean_or_median='mean', wy=False):
    """ Base flow. Compute the mean annual flows. Compute the minimum of a
        7-day moving average flow for each year and divide them by the mean annual flow for that year.
        ML17 is the mean (or median - Use Preference option) of those ratios
    Args:
        Q (pandas DataFrame): Q data with time index - must be on a daily step.
        wy (Bool): True if aggregate by water year, False if calendar year.
        mean_or_median (str): 'mean' or 'median', indicating which statistic to calculate.
 
    """
    # calculate the 7 day rolling mean
    Q['r7mean'] = Q.discharge.rolling(7).mean()
    if wy:
        Q=add_water_year(Q)
        tmp= Q.groupby(Q.water_year).r7mean.min()/Q.groupby(Q.water_year).discharge.mean()
    else:
        tmp =  Q.groupby(Q.index.year).r7mean.min()/Q.groupby(Q.index.year).discharge.mean()
    if mean_or_median.lower() == 'mean':
        return np.nanmean(tmp)
    elif mean_or_median == 'median':
        return np.nanmedian(tmp)
    else:
        raise(f'{mean_or_median} must be "mean" or "median')
    
def ML18(Q, wy=False):
    """ Variability in base flow. Compute the standard deviation for the ratios of
    7-day moving average flows to mean annual flows for each year. ML18 is the 
    standard deviation times 100 divided by the mean of the ratios
    Args:
        Q (pandas DataFrame): Q data with time index - must be on a daily step.
        wy (Bool): True if aggregate by water year, False if calendar year.
    
    """
    # calculate the 7 day rolling mean
    Q['r7mean'] = Q.discharge.rolling(7).mean()
    if wy:
        Q=add_water_year(Q)
        tmp = Q.groupby(Q.water_year).r7mean.mean()/Q.groupby(Q.water_year).discharge.mean()
    else:
        tmp = Q.groupby(Q.index.year).r7mean.mean()/Q.groupby(Q.index.year).discharge.mean()
    
    return np.nanstd(tmp)*100

def ML19(Q, mean_or_median='mean', wy=False):
    """ Base flow. Compute the ratios of the minimum annual flow to mean annual
        flow for each year. ML19 is the mean (or median) of these ratios times 100 
    Args:
        Q (pandas DataFrame): Q data with time index - must be on a daily step.
        wy (Bool): True if aggregate by water year, False if calendar year.
        mean_or_median (str): 'mean' or 'median', indicating which statistic to calculate.

    """
    # calculate the 7 day rolling mean
    if wy:
        Q=add_water_year(Q)
        tmp= Q.groupby(Q.water_year).discharge.min()/Q.groupby(Q.water_year).discharge.mean()
    else:
        tmp =  Q.groupby(Q.index.year).discharge.min()/Q.groupby(Q.index.year).discharge.mean()
    if mean_or_median.lower() == 'mean':
        return np.nanmean(tmp) * 100
    elif mean_or_median == 'median':
        return np.nanmedian(tmp) * 100
    else:
        raise(f'{mean_or_median} must be "mean" or "median')





    