import pandas as pd
import numpy as np
from pytsproc.utils import linear_interpolate_to_time
from scipy.signal import find_peaks


def exceedance_time(Q, time_units, under_over, threshold_flow, delay=None):
    """ Calculates the amount of time a flow is above or below a threshold
    Args:
        Q (pandas DataFrame): Q data with time index
        time_units (str): time units for exceedance, must be one of 'year','month','day','hour', 'minute', or 'sec'
        under_over (str): 'over' indicates to return time above the threshold_flow, 'under' indicates to return
            time below the threshold_flow
        threshold_flow (real): Flow (in common units as the Q discharge data) of the threshold
        delay (datetime, str): Time before which exceedance should not be calculated
    Returns:
        exceedance_time (int): amount of time, in `time_units` time above or below `threshold_flow`
    """
    # make a copy to not tweak the original
    Qout = Q.copy()

    # drop times before "delay" parameter if included
    if delay != None:
        Qout = Qout.loc[Qout.index>delay]

    # first interpolate to time unit indicated
    Qout = linear_interpolate_to_time(Qout, time_units)

    # now count values above or below the threshold, as appropriate
    if under_over.lower() == 'under':
        et = len(Qout.loc[Qout.discharge<=threshold_flow])
    elif under_over.lower() == 'over':
        et = len(Qout.loc[Qout.discharge>=threshold_flow])
    
    # now return the number of times over or under
    return et

    
def flow_duration(Q, exceedance_probabilities=[99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5],
    start_datetime=None, end_datetime=None, time_resolution='day'):
    """_summary_

    Args:
        Q (pandas DataFrame): Q data with time index
        exceedance_probabilities (arraylike, float, optional):Probability values at which to 
            evaluate exceedance. Range in 0-100. 
            Defaults to [99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5].
        start_datetime (datetime or str, optional): starting time by which to constrain evaluation. 
            Defaults to None.
        end_datetime (datetime or str, optional): ending time by which to constrain evaluation. 
            Defaults to None.
        time_resolution (str, optional): Unit of time over which to evaluate the flow_duration.
            The discharge signal will be linearly interpolated to this resolution prior to calcualtions. 
            Defaults to 'day'
    returns:
        pandas DataFrame with index of exceedance_probability vector and column of flow_duration values
    """
    # be sure all exceedance probabilities are in the range 0-100
    if np.sum([True if (i >= 0) & (i<=100) 
        else False for i in exceedance_probabilities]) != len(exceedance_probabilities):
        raise('exceedance_probabilities values must range between 0 and 100')

    # handle the time range
    if start_datetime == None:
        start_datetime = pd.Timestamp.min
    if end_datetime == None:
        end_datetime = pd.Timestamp.max
    Qout = Q.loc[(Q.index>=start_datetime) & (Q.index<=end_datetime)].copy()
    
    Qout =linear_interpolate_to_time(Qout, time_resolution)

    percs = np.nanpercentile(Qout.discharge, [100-i for i in exceedance_probabilities])

    return pd.DataFrame(index=exceedance_probabilities, data=percs, columns = ['flow_duration'])

def hydro_events(Q, prominence=None, distance=None, height=None, threshold=None, wlen=None,
    start_datetime=None, end_datetime=None, **kwargs):
    """_summary_

    Args:
        Q (pandas DataFrame): 
            Q data with time index and flow in ``discharge`` column. Must be on a regular time interval.
        prominence : number or ndarray or sequence, optional
            Required prominence of peaks. Either a number, ``None``, an array
            matching `Q` or a 2-element sequence of the former. The first
            element is always interpreted as the  minimal and the second, if
            supplied, as the maximal required prominence. Defaults to None.
        distance : number, optional
            Required minimal horizontal distance (>= 1) in samples between
            neighbouring peaks. Smaller peaks are removed first until the condition
            is fulfilled for all remaining peaks. Defaults to None.
        height : number or ndarray or sequence, optional
            Required height of peaks. Either a number, ``None``, an array matching
            `Q` or a 2-element sequence of the former. The first element is
            always interpreted as the  minimal and the second, if supplied, as the
            maximal required height. Defaults to None.
        threshold : number or ndarray or sequence, optional
            Required threshold of peaks, the vertical distance to its neighboring
            samples. Either a number, ``None``, an array matching `x` or a
            2-element sequence of the former. The first element is always
            interpreted as the  minimal and the second, if supplied, as the maximal
            required threshold. Defaults to None.
        wlen : int, optional
            Approximate width (asymetrical) of events in units of timescale. When provided, the start
            and end times of each event are calculated and reported. Note: if no prominence parameter
            is supplied, this is not used.
        start_datetime (datetime or str, optional): starting time by which to constrain evaluation. 
            Defaults to None.
        end_datetime (datetime or str, optional): ending time by which to constrain evaluation. 
            Defaults to None.
        kwargs: Other parameters to pass to scipy.signal.find_peaks
    Returns:
        events (numpy array, datetime): Datetimes of event peak locations from the discharge record
        event_meta (dict, various): Starting and ending points for peaks. This is only available if
            prominence is not None and should be used in conjunction with wlen. 


    Notes:
        Hydrologic events are detected as peaks in the regularly-spaced discharge signal using the
        scipy.signal function find_peaks. Values for threshold, height, and prominence are in units
        of discharge, and distance is in number of regulaly-spaced index values.
        More details are available below:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        https://en.wikipedia.org/wiki/Topographic_prominence
        kwargs are passed to the find_peaks function for advanced use.
    """

    # handle the time range
    if start_datetime == None:
        start_datetime = pd.Timestamp.min
    if end_datetime == None:
        end_datetime = pd.Timestamp.max
    Qout = Q.loc[(Q.index>=start_datetime) & (Q.index<=end_datetime)].copy()
    peaks = find_peaks(Qout.discharge,prominence=prominence,
                        distance=distance,
                        threshold=threshold,
                        wlen=wlen, height=height, **kwargs)
    event_meta = {}
    if prominence != None:
        event_meta['event_starts'] = Qout.index[peaks[1]['left_bases']]
        event_meta['event_ends'] = Qout.index[peaks[1]['right_bases']]
        
    return Qout.index[peaks[0]], event_meta

if __name__=='__main__':
    Q_test = pd.read_csv('data/sampleData_q_only__long.csv', index_col=0, parse_dates=True)
    # some Athens tests
    e, em = hydro_events(Q_test.iloc[:100],prominence=5,wlen=None,height=None,distance=None,threshold=None)
    Q_exceed_u = exceedance_time(Q_test, 'day', 'under', 15000, '1/1/2020')
    Q_exceed_o = exceedance_time(Q_test, 'day', 'over', 15000, '1/1/2020')
    exprob = flow_duration(Q_test)

    j=2