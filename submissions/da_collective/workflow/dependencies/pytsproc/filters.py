from calendar import c
from re import I
import pandas as pd
import numpy as np
from pytsproc.utils import linear_interpolate_to_time, hysep_interpolation
from scipy.signal import butter, sosfilt
from scipy.ndimage import minimum_filter1d, generic_filter

#from scipy.ndimage import label

def minimum_filter(Q):
    """Return stationary base flow
    
    The base flow is set to the minimum observed flow.
    adapted from https://github.com/dadelforge/baseflow-separation
    :param ts: 
    :return: 
    """
    Qout = Q.copy()
    minimum = min(Qout.discharge)
    out_values = minimum * np.ones(len(Q))
    Qout['discharge_bf'] = out_values
    Qout['discharge_qf'] = Qout.discharge - out_values
    
    return Qout


def hysep_fixed_interval_filter(Q, size):
    """USGS HYSEP fixed interval method
    
    The USGS HYSEP fixed interval method as described in `Sloto & Crouse, 1996`_.
    
    .. _Slot & Crouse, 1996:
        Sloto, Ronald A., and Michele Y. Crouse. “HYSEP: A Computer Program for Streamflow Hydrograph Separation and 
        Analysis.” USGS Numbered Series. Water-Resources Investigations Report. Geological Survey (U.S.), 1996. 
        http://pubs.er.usgs.gov/publication/wri964040.

    adapted from https://github.com/dadelforge/baseflow-separation

    :param size: 
    :param ts: 
    :return: 
    """
    Qout = Q.copy()

    intervals = np.arange(len(Qout)) // size
    Qout['discharge_bf'] = Qout.groupby(intervals).discharge.transform('min')    
    Qout['discharge_qf'] = Qout.discharge - Qout['discharge_bf']
    
    return Qout

def hysep_sliding_interval_filter(Q, size):
    """USGS HYSEP sliding interval method
    
        The USGS HYSEP sliding interval method as described in `Sloto & Crouse, 1996`_.
        
        The flow series is filter with scipy.ndimage.genericfilter1D using numpy.nanmin function
        over a window of size `size`
    
    .. _Slot & Crouse, 1996:
        Sloto, Ronald A., and Michele Y. Crouse. “HYSEP: A Computer Program for Streamflow Hydrograph Separation and 
        Analysis.” USGS Numbered Series. Water-Resources Investigations Report. Geological Survey (U.S.), 1996. 
        http://pubs.er.usgs.gov/publication/wri964040.
       
        adapted from https://github.com/dadelforge/baseflow-separation

    :param size: 
    :param ts: 
    :return: 
    """
    # TODO: decide whether require even spacing or interpolate in place
    Qout = Q.copy()

    Qout['discharge_bf'] = minimum_filter1d(Qout.discharge, size, mode='reflect')
    Qout['discharge_qf'] = Qout.discharge - Qout['discharge_bf']
    return Qout

def hysep_local_minimum_filter2(Q,size):
    """Local minimum graphical method from HYSEP program (Sloto & Crouse, 1996)
    Args:
        Q (np.array): streamflow
        size (float): width of filter - odd number between 3 and 11
    """
    Qtmp = Q.copy()
    idx_turn = _Local_turn(Qtmp.discharge.values, size)
    # get the font and back bounds into the turning points
    idx_turn = np.insert(idx_turn,0,0)
    idx_turn = np.append(idx_turn, len(Qtmp)-1)
    bf = hysep_interpolation(Qtmp, idx_turn)
    Qout = Q.copy()
    Qout['discharge_bf'] = bf
    Qout['discharge_qf'] = Qtmp.discharge - Qout['discharge_bf']
    del Qtmp
    return Qout

def _Local_turn(Q, inN):
    """find turning points adapted from 

    Args:
        Q (_type_): _description_
        inN (_type_): _description_

    Returns:linear_interpolate_to_time
        _type_: _description_
    """
    idx_turn = np.zeros(Q.shape[0], dtype=int)
    for i in np.arange(int((2*inN - 1) / 2), int(Q.shape[0] - (2*inN - 1) / 2)):
        if Q[i] == np.min(Q[int(i - (2*inN - 1) / 2):int(i + (2*inN + 1) / 2)]):
            idx_turn[i] = i
    return idx_turn[idx_turn != 0]

def hysep_local_minimum_filter(Q, size):
    """USGS HYSEP local minimum method
    
        The USGS HYSEP local minimum method as described in `Sloto & Crouse, 1996`_.
    
    .. _Slot & Crouse, 1996:
        Sloto, Ronald A., and Michele Y. Crouse. “HYSEP: A Computer Program for Streamflow Hydrograph Separation and 
        Analysis.” USGS Numbered Series. Water-Resources Investigations Report. Geological Survey (U.S.), 1996. 
        http://pubs.er.usgs.gov/publication/wri964040.
        
        adapted from https://github.com/dadelforge/baseflow-separation

    :param size: 
    :param ts: 
    :return: 
    """
    Qout = Q.copy()
    Qout = Q.iloc[:350].copy()
    baseflow_min = pd.Series(generic_filter(Qout.discharge, _local_minimum, footprint=np.ones(size)), index=Qout.index)
    Qout['discharge_bf'] = baseflow_min.interpolate(method='linear')

    # ensure that no baseflow values are > Q measured
    QBgreaterthanQ = Qout['discharge_bf'].values > Qout.discharge.values
    Qout.loc[QBgreaterthanQ, 'discharge_bf'] = Qout.loc[QBgreaterthanQ, 'discharge']
    Qout['discharge_qf'] = Qout.discharge = Qout['discharge_bf']
    return Qout

def _local_minimum(window):
    """
    adapted from https://github.com/dadelforge/baseflow-separation
    """
    win_center_ix = len(window) / 2
    win_center_val = window[win_center_ix]
    win_minimum = np.min(window)
    if win_center_val == win_minimum:
        return win_center_val
    else:
        return np.nan

def butterworth_filter(Q, Wn=None, N=1, btype='lowpass', fs=None):
    """
        Butterworth filter, as implemented by 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    Args:
        Q (pandas DataFrame): 
            Q data with time index and flow in ``discharge`` column. Must be on a regular time interval.
        N (int): Order of filter. Defaults to 1
        Wn (float): 
            The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; 
            for bandpass and bandstop filters, Wn is a length-2 sequence. For a Butterworth filter, this 
            is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).

            For digital filters, if fs is not specified, Wn units are normalized from 0 to 1, 
            where 1 is the Nyquist frequency (Wn is thus in half cycles / sample and 
            defined as 2*critical frequencies / fs). If fs is specified, Wn is in the same units as fs.

            For analog filters, Wn is an angular frequency (e.g. rad/s).
        btype (str): {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            The type of filter. Default is ‘lowpass’. 
        fs (float): The sampling frequency of the digital system. Default is ``None``
    Returns:
        Q : pandas DataFrame
            discharge_butter : filtered discharge values
            discharge : original discharge values 
    """
    Qout = Q.copy()
    pars = {'N':N,
            'Wn':Wn,
            'btype':btype,
            'fs':fs,
            'output': 'sos'}
    pars = {k:v for k, v in pars.items() if v is not None}
   
    sos=butter(**pars)
    Qout['discharge_butter'] = sosfilt(sos,Q.discharge.values)
    return Qout

def baseflow_IHmethod(Qseries, block_length=5, tp=0.9, interp_semilog=True, freq='D', limit=100):
    """
    Implementation taken from Andrew Leaf, pydrograph code 
    https://github.com/aleaf/pydrograph/blob/develop/pydrograph/baseflow.py
    Baseflow separation using the Institute of Hydrology method, as documented in
    Institute of Hydrology, 1980b, Low flow studies report no. 3--Research report: 
    Wallingford, Oxon, United Kingdom, Institute of Hydrology Report no. 3, p. 12-19,
    and
    Wahl, K.L and Wahl, T.L., 1988. Effects of regional ground-water level declines
    on streamflow in the Oklahoma Panhandle. In Proceedings of the Symposium on 
    Water-Use Data for Water Resources Management, American Water Resources Association. 
    
    Args:
    Qseries : pandas Series or DataFrame
        Pandas time series (with datetime index) containing measured streamflow values. If a DataFrame, column
        ``discharge`` must contain the flow values.
    block_length : int
        N parameter in IH method. Streamflow is partitioned into N-day intervals;
        a minimum flow is recorded for each interval.
    tp : float
        f parameter in IH method. For each three N-day minima, if f * the central value
        is less than the adjacent two values, the central value is considered a 
        turning point. Baseflow is interpolated between the turning points.
    interp_semilog : boolean
        If False, linear interpolation is used to compute baseflow between  turning points
        (as documented in the IH method). If True, the base-10 logs of the turning points
        are interpolated, and the interpolated values are transformed back to 
        linear space (producing a curved hydrograph). Semi-logarithmic interpolation
        as documented in Wahl and Wahl (1988), is used in the Base-Flow Index (BFI)
        fortran program. This method reassigns zero values to -2 in log space (0.01)
        for the interpolation.
    freq : str or DateOffset, default ‘D’
        Any `pandas frequency alias <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_
        Regular time interval that forms the basis for base-flow separation. Input data are
        resampled to this frequency, and block lengths represent the number of time increments
        of the frequency. By default, days ('D'), which is what all previous BFI methods
        are based on. Note that this is therefore an experimental option; it is up to the user t
        o verify any results produced by other frequencies.
    limit : int
        Maximum number of timesteps allowed during linear interploation between baseflow 
        ordinances. Must be greater than zero.
    
    Returns
    -------
    Q : pandas DataFrame
        DataFrame containing the following columns:
        minima : N-day minima
        ordinate : selected turning points
        n : block number for each N-day minima
        QB : computed baseflow
        Q : discharge values
    
    Notes
    -----
    Whereas this program only selects turning points following the methodology above, 
    the BFI fortran program adds artificial turning points at the start and end of
    each calendar year. Therefore results for datasets consisting of multiple years
    will differ from those produced by the BFI program.
    
    """
    if len(Qseries) < 2 * block_length:
        raise ValueError('Input Series must be at '
                         'least two block lengths\nblock_length: '
                         '{}\n{}'.format(block_length, Qseries))

    # convert to Series if whole DataFrame is passed
    if isinstance(Qseries, pd.DataFrame):
        Qseries = Qseries.discharge.copy()
    # convert flow values to numeric if they are objects
    # (pandas will cast column as objects if there are strings such as "ICE")
    # coerce any strings into np.nan values
    if Qseries.dtype.name == 'object':
        Qseries = pd.to_numeric(Qseries, errors='coerce')

    # convert the series to a dataframe; resample to daily values
    # missing days will be filled with nan values
    df = pd.DataFrame(Qseries).resample(freq).mean()
    df.columns = ['Q']

    # compute block numbers for grouping values on blocks
    nblocks = int(np.floor(len(df) / float(block_length)))

    # make list of ints, one per measurement, denoting the block
    # eg [1,1,1,1,1,2,2,2,2,2...] for block_length = 5
    n = []
    for i in range(nblocks):
        n += [i + 1] * block_length
    n += [np.nan] * (len(df) - len(n))  # pad any leftover values with nans
    df['n'] = n

    # compute the minimum for each block
    # create dataframe Q, which only has minimums for each block
    blocks = df[['Q', 'n']].reset_index(drop=True).dropna(axis=0).groupby('n')
    Q = blocks.min()
    Q = Q.rename(columns={'Q': 'block_Qmin'})
    Q['n'] = Q.index
    # get the index position of the minimum Q within each block
    idx_Qmins = blocks.idxmin()['Q'].values.astype(int)
    # get the date associated with each Q minimum
    Q['datetime'] = df.index[idx_Qmins]

    # compute baseflow ordinates
    Q['ordinate'] = [np.nan] * len(Q)
    Qlist = Q.block_Qmin.tolist()
    Q['Qi-1'] = [np.nan] + Qlist[:-2] + [np.nan]
    Q['Qi'] = [np.nan] + Qlist[1:-1] + [np.nan]
    Q['Qi+1'] = [np.nan] + Qlist[2:] + [np.nan]
    isordinate = tp * Q.Qi < Q[['Qi-1', 'Qi+1']].min(axis=1)
    Q.loc[isordinate, 'ordinate'] = Q.loc[isordinate, 'block_Qmin']

    # reset the index of Q to datetime
    Q.index = Q.datetime

    # expand Q dataframe back out to include row for each day
    Q = Q.dropna(subset=['datetime'], axis=0).resample(freq).mean()

    # interpolate between baseflow ordinates
    if interp_semilog:
        iszero = Q.ordinate.values == 0
        logQ = np.log10(Q.ordinate)
        logQ[iszero] = -2
        QB = np.power(10.0, logQ.interpolate(limit=limit).values)
    else:
        QB = Q.ordinate.interpolate(limit=limit).values
    Q['QB'] = QB

    # reassign the original flow values back to Q
    Q['discharge'] = df.Q.loc[Q.index]

    # ensure that no baseflow values are > Q measured
    QBgreaterthanQ = Q.QB.values > Q.discharge.values
    Q.loc[QBgreaterthanQ, 'QB'] = Q.loc[QBgreaterthanQ, 'discharge']
    return Q





if __name__ == '__main__':
    Q_test = pd.read_csv('data/sampleData_q_only__long.csv', index_col=0, parse_dates=True)
    
    # some Athens tests
    Q_bf = baseflow_IHmethod(Q_test)
    
    j=2