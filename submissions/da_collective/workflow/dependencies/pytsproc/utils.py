import pandas as pd
import numpy as np

TIME_UNITS = {'year':'y',
              'month': 'M',
              'day':'D',
              'hour':'h',
              'minute':'min',
              'sec':'s'
}



def add_water_year(Q):
    """Calculate the water year and add as a column to a time-indexed dataframe.
    If water_year is already a column in the DataFrame, the DataFrame is returned unchanged.

    Args:
        Q (pandas DataFrame):DataFrame with a time index
    """

    #make water year by shifting forward the number of days in Oct., Nov., and Dec.
    # NOTE --> shifting by months is less precise
    if 'water_year' not in Q.columns:
        Qout = Q.copy()
        Qout['water_year'] = Qout.index.shift(31+30+31,freq='d').year
    else:
        Qout = Q
    return Qout

def add_calendar_year(Q):
    """Extract the calendar year and add as a column to a time-indexed dataframe

    Args:
        Q (pandas DataFrame):DataFrame with a time index
    """
    
    Qout = Q.copy()
    Qout['year'] = Qout.index.year

    return Qout
    
def add_month(Q):
    """Extract the month number and add as a column to a time-indexed dataframe

    Args:
        Q (pandas DataFrame):DataFrame with a time index
    """
    
    Qout = Q.copy()
    Qout['month'] = Qout.index.month

    return Qout
    
def calc_bfi(Q):
    """

    Args:
        Q (_type_): _description_
    """
    return np.nanmin(Q['discharge'].rolling(7).mean())/Q['discharge'].mean()


def linear_interpolate_to_time(Q, time_units='day'):
    """_summary_

    Args:
        Q (_type_): _description_
        time_units (_type_): _description_
    """
    Qtmp = Q.copy()
    Qtmp=Qtmp.asfreq(TIME_UNITS[time_units]).interpolate() # using linear interpolation
    if 'water_year' in Q.columns:
        Qtmp = add_water_year(Qtmp) # need to recalculate water year after interpolation
    
    return Qtmp

def read_ssf(filepath, return_col='discharge'):
    """
    Function to read a PEST/TSPROC .ssf file into a pandas DataFrame.  

    Args:
        filepath (string): File Path
        return_col (string): column name for the data column. Default is 'discharge'
    returns:
        pandas DataFrame with timestamp as the index and includes basename 
        column as the site name (first column in the ssf file)
        which allows for sorting by location. The data column is labelled 
        according to return_col
    """
    Q = pd.read_csv(filepath, delim_whitespace=True, header=None, 
                names = ['basename', 'date','time',return_col], parse_dates = {'datetime': [1,2]})
    Q.set_index(Q.datetime, drop=True, inplace=True)
    Q.drop(columns=['datetime'], inplace=True)

    return Q

def hysep_interpolation(Qdf, idx_turn):
    Q = Qdf.discharge.values
    n = 0
    b = np.zeros_like(Q)
    for i in range(idx_turn[0], idx_turn[-1]+1):
        if i == idx_turn[n + 1]:
            n += 1
            b[i] = Q[i]
        else:
            b[i] = Q[idx_turn[n]] + ((Q[idx_turn[n + 1]] - Q[idx_turn[n]]) /  
                (idx_turn[n + 1] - idx_turn[n]) * (i - idx_turn[n]))
        if b[i] > Q[i]:
            b[i] = Q[i]
    return b