import pandas as pd
import numpy as np
from pytsproc.utils import add_water_year, add_calendar_year

def MA1(Q):
    """
    Mean of the daily mean flow values for the entire flow record 

    Args:
        Q (pandas DataFrame): daily Q data with time index
    """
    return np.nanmean(Q.discharge.values)


def MA2(Q):
    """
    Median of the daily mean flow values for the entire flow record        

    Args:
        Q (pandas DataFrame): daily Q data with time index
    """
    return np.nanmedian(Q.discharge.values)

def MA3(Q, method='median', year_basis='water_year'):
    """
    ma3 Mean (or median - use method option) of the coefficients of variation (standard deviation/mean) for each year. Compute the coefficient of variation for each year of daily flows. Compute the mean of the annual coefficients of variation.

    'method'     : can be 'mean' or 'median'
    'year_basis' : can be 'water_year' or 'year' (calendar year)

    """

    if (year_basis == 'water_year'):
        Q_df = add_water_year(Q)
    else:
        Q_df = add_calendar_year(Q)

    ann_mean = Q_df.groupby(year_basis).mean()
    ann_stdev = Q_df.groupby(year_basis).std()
    ann_CV = ann_stdev / ann_mean

    if ( method == 'mean'):
        MA3 = ann_CV.discharge.mean() * 100.
    elif ( method == 'median'):
        MA3 = ann_CV.discharge.median() * 100.
    else:
        MA3 = np.nan

    return MA3   

def MA6(Q):
    """
    ma6 Range in daily flows is the ratio of the 10-percent to 90-percent exceedence values for the entire flow record. 
    """
    Q90 = np.nanpercentile(a=Q.discharge, q=90.)
    Q10 = np.nanpercentile(a=Q.discharge, q=10.)

    try:
        MA6 = Q90 / Q10
    except:
        MA6 = np.nan

    return MA6    


