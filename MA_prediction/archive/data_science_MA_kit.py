"""     head imports
## import the two lines to make display better on jupyter notebook. No need for jupyter lab. ##
from IPython.display import display, HTML
display(HTML("<style>.container { width:92% !important; }</style>"))

%load_ext autoreload
%autoreload 2
import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path = "~/Documents/G3/MA-prediction"
path = expanduser(path)
sys.path.append(path)

import data_science_MA_kit as dsk
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import re
# import wrds

pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn-v0_8")
plt.rcParams['figure.figsize'] = [9, 9]
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.size'] = 30
"""

""" old plot imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn")
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['figure.titlesize'] = 25
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
"""

"""
# pandas Timestamp to datetime.date
x.date()

# pandas Timestamp series to datetime.date series
x.dt.date

# DatetimeIndex to datetime.date np array
x.date
"""

"""
import pickle

with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
"""

from os.path import expanduser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from datetime import timedelta
import pandas_market_calendars as mcal
import re

pd.options.mode.chained_assignment = None

import statsmodels.api as sm
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

import locale
from locale import atof, setlocale

setlocale(locale.LC_ALL, 'en_US')


##############################
## basic transformations
##############################
def extract_colnames_from_report_file(filepath):
    """
    extract the column names of the dataset from the report file, as a list of lowercase strings.
    Each line after 'Custom Report Contents:\n' is a column name.
    """
    # open file as a list of strings of each line 
    lines = open(expanduser(filepath), 'r').readlines()

    # only need lines after "Custom Report Contents:\n"
    ind = 0
    while not lines[ind].startswith('Custom Report Contents:'):
        ind += 1
    # discard newline character. return lower cases.
    return list(map(lambda x: x.strip().lower(), lines[ind+1:]))


# checked
def convert_date_str_ser_to_datetime(ser):
    """
    convert a series of date-like strings to datetime.date objects.
    NA is allowed.
    
    Parameters:
    ---------------------
    ser: Series
        a series of date-like strings.
        
    Returns:
    ---------------------
    a series of datetime.date.
    """
    return pd.to_datetime(ser).dt.date


# checked
def convert_num_str_ser_to_float(ser):
    """
    convert a series of numeric-like strings (e.g. '1,000') to floats. NA is allowed in <ser>.
    
    Parameters:
    ---------------------
    ser: Series
        a series of numeric-like strings.
        
    Returns:
    ---------------------
    a series of floats.
    """
    return ser.map(atof, na_action='ignore')



##################
## correct dataset
##################

def get_delete_index(df):
    """
    get the index of wrong data entries to be deleted manually.
    """
    index_del = [274214020, 227680020, 243448020]
    return df.index.intersection(index_del)
    

def correct_consid(dff):
    """
    fix data error manually. <dff> should contain columns <consido> and <consid>.
    """ 
    # create copy
    df = dff.copy()
    
    # 
    df.loc[1040793020, ['consido', 'consid']] = ['Cash Only', '$7 cash/sh com']
    
    # fix `consid` manually
    index = [1018291020,
 1752953020,
 1871799020,
 2474162020,
 2684890020,
 2770926020,
 2942217020,
 2634022020,
 3037577020,
 3090980020,
 3098396020,
 3121972020,
 3238285020,
 3416138020,
 3453292020,
 3473708020,
 3664650020,
 3711080020,
 3700733020,
 3728695020,
 3761157020,
 3306644020,
 3846599020,
1889692020,
        1610526020,
        1425273020,
        1594235020,
        2724775020,
        2530559020,
        2309478020,
        3100226020,
        3731057020,
        1181437020,
        1581335020,
        2333741020,
        1074016020,
        2952899020,
        2732707020,
        3213676020,
        2701755020,
        2229022020,
        2291770020,
        2919836020,
        934761020,
            569766020,
            170805020,
            938046020,
            166546020,
            197336020,
            238913020,
            325340020,
            330520020,
            338719020,
            344267020,
            389787020,
            414062020,
            405440020,
            420869020,
            436959020,
            692819020,
            679983020,
            616158020,
            565173020,
            553658020,
            542491020,
            490778020,
            483432020,
            476416020,
            2382542020] +\
    [125504020, 146736020, 147638020, 163620020, 164842020, 174041020, 179091020, 180234020, 754797020, 806241020, 806318020, 1259809020]
    
    correction_consid = ['.105 shs com/sh com',
 '$6 cash and $2.5 com/sh com',
 '$40 cash plus $35 com/sh com',
 '$25 cash plus 0.6531 shs com/sh com',
 '$1.26 cash plus 0.5846 shs com/sh com',
 '$62.93 cash plus 0.6019 shs com/sh com',
 '$6.45 cash plus 0.75 shs com/sh com',
 '$34.1 cash and $27.9 com/sh com',
 '$1.91 cash plus 0.14864 shs com/sh com',
 '$95.63 cash/sh com',
 '$35 cash plus 0.23 shs com/sh com',
 '$3.78 cash plus 0.7884 shs com/sh com',
 '$2.3 cash plus 0.738 shs com/sh com',
 '$6.28 cash plus 0.528 shs com/sh com',
 '$6.8 cash and .7275 shs com/sh com',
 '$2.5 cash plus 0.8 shs com/sh com',
 '$26.790 cash plus 0.0776 shs com/sh com',
 '$1.39 cash plus 0.42 shs com/sh com',
 '$41.75 cash plus 0.907 shs com/sh com',
 '$2.89 cash plus 1.408 shs com/sh com',
 '0.4 shs com/sh com',
 '$1.46 cash and .297 shs com/sh com',
 '$133 cash plus .4506 shs com/sh com',
             '$12.5 cash/sh com',
             '$1.50 cash plus $13.875 com/sh com',
             '$17.5 cash/sh com',
             '$11.375 cash plus .2917 shs com/sh com',
             '$6.25 cash plus 0.3521 shs com/sh com',
             '$83 cash plus $49.5 shs com/sh com',
             '$5 cash plus 0.588 shs com/sh com',
             '1.123 shs com/sh com',
             '.124 shs com/sh com',
             '1.14 shs com/sh com',
             '0.89 shs com/sh com',
             '0.46 shs com/sh com',
             '1.7 shs com/sh com',
             '1.63 shs com/sh com',
             '0.73 shs com/sh com',
             '0.27 shs com/sh com',
             '0.2413 shs com/sh com',
             '0.93 shs com/sh com',
             '1.02 shs com/sh com',
                 '4.4 shs com/sh com',
                 '.4626 shs com/sh com',
                 '$65 cash/sh com',
                 '1.752 shs com/sh com',
                 '.322 shs com/sh com',
                 '1.12 shs com/sh com',
                 '.384 shs com/sh com',
                  '$30.25 com/sh com',
                 '$17.5 com/sh com',
                 '$20 cash plus .45 shs com/sh com',
                 '2.13 shs com/sh com',
                 '1 shs com/sh com',
                 '.2 shs com/sh com',
                 '.6803 shs com/sh com',
                 '.438 shs com/sh com',
                 '$5.875 cash plus $6.125 com/sh com',
                 '$16 cash/sh com',
                 '$28.85 cash/sh com',
                 '.55 shs com/sh com',
                 '.65 shs com/sh com',
                 '$12.75 com/sh com',
                 '.933 shs com/sh com',
                 '$8 cash and $32 com/sh com',
                 '1.05 shs com/sh com',
                 '.53 shs com/sh com',
                 '.845 shs com/sh com',
                 '$21.75 cash/sh com'] + \
    ['$17 cash/sh com', '$17 cash plus 1.07195 shs com/sh com', '$45 cash/sh com', '1.3889 shs com/sh com', '$16.2 cash/sh com', '$9.25 cash/sh com', '.2 shs com/sh com', '1 shs com/sh com',
'$2.23 cash plus 1 shs com/sh com', '.34 shs com/sh com', '1.347 shs com/sh com', '$2.21 cash/sh com']
    
    # create the mapping series
    correction_ser = pd.Series(correction_consid, index=index, name='consid')
    
    # find the indices also in <df>
    index_intersection = df.index.intersection(index)
    # do the correction
    df.consid[index_intersection] = correction_ser[index_intersection]
    return df












######################
## market calendar
######################
# checked
def get_trading_day_range(start_date = '1900-01-01', end_date = '2030-01-01', return_as_series = False):
    """
    return the valid trading days (of NYSE) within the date range. 
    return either a Series or an array.
    
    Parameters:
    -----------------------------------------
    start_date: 
    
    end_date: 
    
    return_as_series: boolean
        whether to return as Series. Otherwise return as array.
    
    Returns:
    -----------------------------------------
    Series
    """
    
    nyse = mcal.get_calendar('NYSE')
    days = nyse.valid_days(start_date, end_date).date
    if return_as_series:
        return pd.Series(days)
    return days

# checked
def get_trading_day_offset(ser, offset):
    """
    get the trading days with a certain offset from the input dates.
    offset=0 is the nearest next trading day, including the day itself. 
    
    Parameters:
    -----------------------------------------
    ser: Series of, or a single `datetime.date` or string `YY-mm-dd`
        NA allowed
    offset: int
        number of days in the offset.
    
    
    Returns:
    -----------------------------------------
    a series of, or a single `datetime.date`
    
    
    Examples:
    ----------------------------------------------
    input: ser = datetime.date(2022, 9, 4) or datetime.date(2022, 9, 5) or datetime.date(2022, 9, 6), offset = 0
    output: datetime.date(2022, 9, 6).
    """
    if isinstance(ser, pd.Series):
        if ser.empty:   # edge case
            return pd.Series(dtype=object)
        if ser.isna().all():   # all missing
            return pd.Series(dtype=object, index=ser.index)

        # change to series of `datetime.date` objects
        ser_new = convert_date_str_ser_to_datetime(ser.dropna())#pd.to_datetime().dt.date
        # get date range
        start, end = ser_new.min(), ser_new.max()
    else: # supposedly ser is a single datetime-like object
        if pd.isna(ser):     # edge case
            return np.nan
        # convert to `datetime.date`
        ser_new = pd.to_datetime(ser).date()
        # get date range
        start, end = ser_new, ser_new
    
    # some tolerance    
    start -= timedelta(days=2*abs(offset) + 5)
    end += timedelta(days=2*abs(offset) + 5)
#     return start, end
    
    # get `NYSE` calendar
    trading_days = get_trading_day_range(start, end)

    # find index in the series of all trading days
    ind_arr = np.searchsorted(trading_days, ser_new)
    days_with_offset = trading_days[ind_arr + offset]
    
    if isinstance(ser, pd.Series):  # fill back those missing numbers
        return pd.Series(days_with_offset, index=ser_new.index).reindex(ser.index)
    return days_with_offset

# checked
def get_num_trading_days_between(left_date_ser, right_date_ser):
    """
    calculate the number of trading days between two series or two single dates.
    NA is allowed.
    """
    
    if isinstance(left_date_ser, pd.Series) and isinstance(right_date_ser, pd.Series):
        # dropna
        dates_combined = pd.concat([convert_date_str_ser_to_datetime(left_date_ser), convert_date_str_ser_to_datetime(right_date_ser)], axis=1).dropna()
        if dates_combined.empty:
            return pd.Series(np.nan, index=left_date_ser.index)
        # get the ser without na
        left, right = dates_combined.iloc[:, 0], dates_combined.iloc[:, 1]
        # start and end date for pulling trading days
        start, end = left.min(), right.max()
    else:    # two single date-likes
        if pd.isna(left_date_ser) or pd.isna(right_date_ser):
            return np.nan
        # get the ser without na
        left, right = pd.to_datetime(left_date_ser).date(), pd.to_datetime(right_date_ser).date()
        # start and end date for pulling trading days
        start, end = left, right
    
    start -= timedelta(5)
    end += timedelta(5)
        
    trading_days = get_trading_day_range(start, end)
    # difference of indice of left and right
    days_diff = np.searchsorted(trading_days, right) - np.searchsorted(trading_days, left)
    
    if isinstance(left_date_ser, pd.Series):
        return pd.Series(days_diff, index=left.index).reindex(left_date_ser.index)
    return days_diff



def count_num_bewteen(ser, left_ends, right_ends, inclusive='both'):
    """
    for each value in series, count how many intervals it is included in. The end points of intervals is included in left/right_ends.
    
    Parameters:
    ------------------------------------------
    ser:
    
    left_ends: Series.
        the series of all the left ends of the intervals
    right_ends: Series
    
    inclusive: {“both”, “neither”, “left”, “right”} 
    
    
    Returns:
    ----------------------------------------
    Series, with index being the values of ser.
    """
    counts = pd.Series(0, index=ser.index)
    for left, right in tqdm(list(zip(left_ends, right_ends))):
        counts[ser.between(left, right, inclusive=inclusive)] += 1
    counts.index=ser
    return counts




#####################
## CRSP helpers
#####################

# checked
def get_delisting_information(permno, 
                              cols=['dlstcd', 'dlstdt', 'dlpdt', 'dlamt', 'dlret'],
                              db=None):
    """
    search for delisting information in crsp_m_stock.dsedelist database, by permno of the stock.
    
    All the columns are 
    ['permno', 'last_trade_date', 'delist_code', 'nwperm', 'nwcomp',
       'nextdt', 'delist_amount', 'dlretx', 'dlprc', 'delist_date',
       'delist_return', 'permco', 'compno', 'issuno', 'hexcd', 'hsiccd',
       'cusip', 'acperm', 'accomp']
       
    Return a series.
       
    NA is allowed.
    
    Parameters:
    ----------------------------------------
    permno:
    
    cols: list
    
    db:
    
    Returns:
    ----------------------------------------
    Series.
    """
    cols_lst = ['dlstcd', 'dlstdt', 'dlpdt', 'dlamt', 'dlret']
    name_lst = ['delist_code', 'last_trade_date', 'delist_date', 'delist_amount', 'delist_return']
    rename_dict = dict(zip(cols_lst, name_lst))

    if pd.isna(permno):  # missing
        return pd.Series(dtype=object, index=cols).rename(index=rename_dict)
    # not missing permno
    command = f"select {', '.join(cols)} from crsp_m_stock.dsedelist where permno = {permno}"   
    df_delist = db.raw_sql(command)
#     return df_delist
    if len(df_delist)!= 1:
        return pd.Series(dtype=object, index=cols).rename(index=rename_dict)
    df_delist = df_delist.iloc[0]
    df_delist.name = None
    return df_delist.rename(index=rename_dict)

# checked
def get_stock_stocknames_CRSP(id_no, id_type='permno', date = None, 
                              cols=['permno', 'permco', 'ticker', 'comnam', 'namedt', 'nameenddt', 'cusip', 'ncusip'], 
                              db=None):
    """
    get stocknames file of a stock from the `crsp_m_stock.stocknames`, by its permno / ticker / cusip.
    
    all the available columns are 
    ['permno', 'namedt', 'nameenddt', 'shrcd', 'exchcd', 'siccd', 'ncusip',
       'ticker', 'comnam', 'shrcls', 'permco', 'hexcd', 'cusip', 'st_date',
       'end_date', 'namedum']
    
    can add the filter that the company is listed on a given date.
       
    Parameters:
    -----------------------------
    id_no: supports permno / ticker / cusip.
    
    id_type: {'permno', 'ticker', 'cusip'}
    
    date:
        The stock is listed on this date.
    cols: list
        
    db: 
    
    Returns:
    -----------------------------
    DataFrame or Series.
    
    """
    command = f"select {', '.join(cols)} from crsp_m_stock.stocknames where "
    
    if id_type == 'permno':
        command += f" permno = {id_no}"
    elif id_type == 'ticker':
        command += f" ticker = '{id_no}'"
    elif id_type == 'cusip':
        command += f" (substring(cusip, 1, {len(id_no)}) = '{id_no}' or substring(ncusip, 1, {len(id_no)}) = '{id_no}')"
    else:
        print('Unrecognized identifier.')
        return None
    
    if date != None:
        command += f" and namedt <= '{str(date)}' and nameenddt >= '{str(date)}'"
#     return command    
    return db.raw_sql(command)

# checked
def convert_stocknames_to_permno(stocknames, return_names=False):
    """
    extract permno and stock information from a stocknames file.
    
    Pay attention to 'Multiple permnos'. 
    Output this string if return a value, and the string appears in comnam if return a series.
    
    Parameters:
    ------------------------------------
    stocknames:
    
    return_names: boolean, default False
        if True return a series of ['permno', 'ticker', 'cusip', 'comnam'], otherwise just the permno number
    
    Returns:
    ------------------------------------
    float or Series
    
    """
    
    index = ['permno', 'ticker', 'cusip', 'comnam']
    if stocknames.empty:   # no stocknames
        return pd.Series(dtype=object, index=index) if return_names else np.nan
    elif len(stocknames.permno.unique()) > 1:    # multiple permnos
#         print('Multiple permnos. Check.')
        return pd.Series([np.nan, np.nan, np.nan, 'Multiple permnos'], index=index) if return_names else 'Multiple permnos'
    else:     # unique permno
        if return_names:
            ser = stocknames.iloc[-1][index]
            ser.name = None
            return ser
        return stocknames.iloc[-1]['permno']
    
# checked    
def get_stock_permno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):
    """
    get the permno of a stock by its ticker or cusip.
    
    Parameters:
    ------------------------------------
    id_no: support ticker / cusip
    
    id_type: {'ticker', 'cusip'}
    
    date: 
        The stock is listed on this date.
    return_names: boolean, default False
        if True return a series of ['permno', 'ticker', 'cusip', 'comnam'], otherwise just the permno number
    db:
    
    
    Returns:
    ------------------------------------
    float or Series
    """
    stocknames = get_stock_stocknames_CRSP(id_no, id_type=id_type, date = date, db=db)
    return convert_stocknames_to_permno(stocknames, return_names)

# checked
def get_stock_permno_by_ticker_and_cusip_CRSP(ticker, cusip, date=None, return_names=False, db=None):
    """
    get the permno of a stock by its ticker, and then cusip.
    
    Returns:
    ------------------------------------
    permno if return_id is False, else a series of (permno, ticker, cusip, comnam)
    """
    def check_permno_notnull(permno, return_names):
        return ((not return_names) and pd.notna(permno) and isinstance(permno, float)) or (return_names and pd.notna(permno.permno))
    def check_multiple_permnos(permno, return_names):
        return ((not return_names) and permno=='Multiple permnos') or (return_names and permno.comnam=='Multiple permnos')
    # search by ticker first
    if pd.notna(ticker):
        permno = get_stock_permno_CRSP(ticker, 'ticker', date, return_names, db)
        if check_permno_notnull(permno, return_names):
            return permno
    # by cusip then
    if pd.notna(cusip):
        permno = get_stock_permno_CRSP(cusip, 'cusip', date, return_names, db)
        if check_permno_notnull(permno, return_names):
            return permno
        
    # multiple permnos
    if check_multiple_permnos(permno, return_names):
        return permno
    # cannot match by either ticker or cusip
    index = ['permno', 'ticker', 'cusip', 'comnam']
    return pd.Series(dtype=object, index=index) if return_names else np.nan


def get_stock_market_data_daily_CRSP(id_no, id_type='permno', start_date='1900-01-01', end_date='2030-01-01', 
                                cols=['permno', 'prc', 'ret', 'vol', 'shrout', 'cfacpr', 'cfacshr'], 
                                db=None):
    """
    get the daily time series of equity market data from the `crsp_m_stock.dsf` database, by its permno or ticker.
    
    all the available columns are 
    ['cusip', 'permno', 'permco', 'issuno', 'hexcd', 'hsiccd', 'bidlo',
       'askhi', 'prc', 'vol', 'ret', 'bid', 'ask', 'shrout', 'cfacpr',
       'cfacshr', 'openprc', 'numtrd', 'retx']
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'permno', 'ticker'}, default 'permno'
    
    start_date: datetime.date, or string. default '1900-01-01'.
    
    end_date: datetime.date, or string. default '2030-01-01'. 
    
    cols: list
        ['*'] or any combination. Would automatically add 'date' if it isn't included.
    db: 
    
    Returns:
    -----------------------------
    DataFrame or Series, depending on whether there is only one column.
    
    """

    # convert ticker to permno
    if pd.isna(id_no):
        return np.nan
    if id_type == 'ticker':
        date = start_date if start_date != '1900-01-01' else None
        permno = get_stock_permno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if permno == None:
            print("cannot find permno.")
            return pd.DataFrame(dtype=float, columns=cols)
    elif id_type == 'permno':
        permno = id_no
    else:
        print('Unrecognized identifier.')
        return pd.DataFrame(dtype=float, columns=cols)
    
    # add 'date'
    if 'date' not in cols and '*' not in cols:
        cols += ['date']
    #    
    start_date, end_date = str(start_date), str(end_date)
    
    #
    command = f"select {', '.join(cols)} from crsp_m_stock.dsf where permno = " + \
    f"{str(permno)} and date >= '{start_date}' and date <= '{end_date}'"
    
    #
    market_data = db.raw_sql(command).set_index('date').sort_index() 
    return market_data if len(market_data.columns) > 1 else market_data.iloc[:, 0]


# checked

def combine_ser_of_ser_into_df(ser_of_ser, use_new_cols=None):
    """
    combine a series of series into dataframe.
    
    single na in the values of <ser_of_ser> is allowed.
    """
    ser_of_ser_dropna = ser_of_ser.dropna()
    if use_new_cols == None:   # don't use new column names
        return pd.DataFrame(list(ser_of_ser_dropna.values), index=ser_of_ser_dropna.index).reindex(ser_of_ser.index)
    else:        # use new column names
        return pd.DataFrame(list(ser_of_ser_dropna.map(lambda x: x.values).values),
                            index=ser_of_ser_dropna.index,
                           columns=use_new_cols).reindex(ser_of_ser.index)
    
    
    
def apply_func_to_ser_df(ser, func, *args, return_as_df = False, use_new_cols=None, **kwargs):
    """
    apply function to each value in a series, or each row (as a series) in a DataFrame.
        
    The output can be either a DataFrame or series, by parameter <return_as_df>. 
    If a DataFrame is desired, it is assumed the output of the function is a Series whose index is the same for every row.
    Otherwise it is recommended to output as a series, and further process out of this function.
    
    The function has the same functionality as `ser.map()` or `df.apply(axis=1)`, but you can monitor progress by tqdm in this function.
    
    NA is allowed.
    
    Parameters:
    --------------------------------
    ser: Series or DataFrame
        If DataFrame, apply function to each row.
    func:
    
    args:
        for <func>
    return_as_df: boolean, default False
        return as a DataFrame if True.
    kwargs:
        for <func>
        
    Returns:
    -------------------------------------
    Series or DataFrame
    
    """
    ret_lst_dropna = []
    # index without NA values
    index_dropna = ser.index[ser.notnull()] if isinstance(ser, pd.Series) else ser.index[ser.notnull().any(axis=1)]
#     return index_notna
    for ind in tqdm(index_dropna):
        ret_lst_dropna.append(func(ser.loc[ind], *args, **kwargs))
    # series of series
    ret_ser_of_ser_dropna = pd.Series(ret_lst_dropna, index=index_dropna)
    
    if not return_as_df: # return as series
        return ret_ser_of_ser_dropna.reindex(ser.index)
    else:                # return as df
        ret_df_dropna = combine_ser_of_ser_into_df(ret_ser_of_ser_dropna, use_new_cols)
        return ret_df_dropna.reindex(ser.index) 
    

########################################
## process equity market data
########################################

def add_delisting_return(df, market_data_tgt):
    for ind in tqdm(df.index):
        last_trade_date, delist_code = df.loc[ind, ['last_trade_date', 'delist_code']]
        # delisting due to MA, and last trade date matches last date in the market data
        if 200<=delist_code<300 and pd.notna(last_trade_date) and last_trade_date==market_data_tgt[ind].index[-1]:
            delist_date = df.delist_date[ind]
            # add a new row
            market_data_tgt[ind].loc[delist_date] = np.nan
            cols = ['permno', 'shrout', 'cfacpr', 'cfacshr']
            # coppy these numbers from last row
            market_data_tgt[ind].iloc[-1][cols] = market_data_tgt[ind].iloc[-2][cols]
            # add the delisting amount and return
            market_data_tgt[ind].iloc[-1][['prc', 'ret']]=[df.delist_amount[ind], df.delist_return[ind]]    
    return market_data_tgt

def correct_prc_ret(values):
    """
    correct the returns and prices from CRSP. Problems include:
    - For all None prices and returns in the period, replace None by nans.
    - For subperiod of None prices and returns, fill prices by previous value, and returns by zero.
    - negative price: take absolute value.
    """
    # missing price
    if 'prc' in values.columns:
        # if all none, replace by nan
        if values.prc.isna().all():
            values.prc = np.nan
        else:
            values.prc = abs(values.prc)
            values.prc = values.prc.fillna(method='ffill')
    # missing returns
    if 'ret' in values.columns:
        # if all none, replace by nan
        if values.ret.isna().all():
            values.ret = np.nan
        else:
            values.ret = values.ret.fillna(0.)            
    return values

def calculate_mktcap(market_data_df):
    """
    given market data, calculate mktcap, and shifted (previous day) mktcap
    """
#     if market_data_df.prc.notnull().any():
#         market_data_df['mktcap'] = market_data_df.prc.mul(market_data_df.shrout)
#     else:
#         market_data_df['mktcap'] = np.nan
    market_data_df['mktcap'] = market_data_df.prc.mul(market_data_df.shrout)
    market_data_df['mktcap_prev'] = market_data_df.mktcap.shift()
    return market_data_df  


def adjust_price_to_ann(market_data_df, da):
    """
    adjust price by accumulative factor, based on the announcement day.
    """
    if da in market_data_df.index:
        market_data_df['prc_adj_ann'] = market_data_df.prc.div(market_data_df.cfacpr)*market_data_df.cfacpr[da]
    else:
        market_data_df['prc_adj_ann'] = np.nan
    market_data_df['prc_adj_ann_prev'] = market_data_df.prc_adj_ann.shift()
    return market_data_df  


# def get_target_acquiror_values_CRSP(df, 
#                                     cols = ['permno', 'prc', 'ret', 'shrout', 'cfacpr', 'cfacshr', 'vol'],
#                                     db=None):
#     """
#     get the market price and return data for targets and acquirors during the deal process
    
#     """
#     values_target = pd.Series(index=df.index, dtype=object)
#     values_acquiror = pd.Series(index=df.index, dtype=object)
#     for ind in tqdm(df.index):
#         tpermno, apermno, start_date, end_date = df.loc[ind, ['tpermno', 'apermno', 'da_5days_prior', 'dr_4days_after']]
#         if pd.notnull(tpermno):
#             values_target[ind] = get_stock_values_daily_CRSP(tpermno, start_date=start_date, end_date=end_date, cols=cols, db=db)
#         if pd.notnull(apermno):
#             values_acquiror[ind] = get_stock_values_daily_CRSP(apermno, start_date=start_date, end_date=end_date, cols=cols, db=db)
            
#     return values_target, values_acquiror    

################################
## backtesting
################################

def zip_two_sers(ser1, ser2):
    """
    given two series, zip them into one series in the form of [(value1, value2)]
    """
    return pd.Series(zip(ser1, ser2), index=ser1.index).map(lambda x: [x])



from datetime import timedelta

def correct_holding_periods_compete_overlap(holding_periods, df):
    """
    eliminate overlapping in holding periods for competing deals
    """
    holding_periods_new = pd.Series(dtype=object, index = holding_periods.index)
    for ind in holding_periods.index:
        holding_periods_new[ind] = []
        for period in holding_periods[ind]:
            holding_periods_new[ind].append(period)
#     holding_periods_new = holding_periods.copy()
    for group_no in range(int(df.compete_group_no.max())+1):
        # index sorted by announcement date
        index = df.da_corrected[df.compete_group_no.eq(group_no)].sort_values().index
        for i in range(len(index) - 1):
            _, end_prev = holding_periods_new[index[i]][0]
            end_prev += timedelta(1)    #dsk.get_trading_day_offset(end_prev, 1)
            for j in range(i+1, len(index)):
                start_now, end_now = holding_periods_new[index[j]][0]
                holding_periods_new[index[j]][0] = (max(start_now, end_prev), end_now)    
    return holding_periods_new

def correct_holding_periods_delete_invalid(holding_periods):
    """
    delete period where start is later than end
    """
    holding_periods_new = pd.Series(dtype=object, index = holding_periods.index)
    for ind in holding_periods.index:
        holding_periods_new[ind] = []
        for period in holding_periods[ind]:
            if period[0] < period[1]:
                holding_periods_new[ind].append(period)
    return holding_periods_new

def correct_holding_periods(holding_periods, df):
    return correct_holding_periods_delete_invalid(correct_holding_periods_compete_overlap(holding_periods, df))


def compound_daily_return_to_monthly(ret_ser):
    """
    Compound a daily return series into a monthly return series.
    """
    ret_ser_copy = ret_ser.copy()
    ret_ser_copy.index = pd.PeriodIndex(ret_ser_copy.index, freq='d')
    monthly_ret = ret_ser_copy.resample('M').apply(lambda x: (1+x).prod()-1)
    return monthly_ret


import gc

def compute_weighted_average(return_df, weight='weight'):
    """
    <return_df> contains columns `returns` and `weights`. computes the weighted returns.
    """
    if return_df.empty:   # no positions
        return 0.
    else:
        if weight == 'equal':
            return return_df.returns.mean()
        elif weight == 'weight':
            return np.average(return_df.returns, weights=return_df.weights)
        else:
            print('Unsupported weighting scheme!')
            return np.nan
        
        
        
        
def backtest_daily_returns(df, holding_periods, market_data_tgt_corrected, market_data_acq_corrected, deal_no_lst, weight='value', return_raw = False):
    """
    The main function to backtest the daily returns of any strategy.
    
    
    Parameters:
    --------------------------------
    df: DataFrame
        
    holding_periods: Series
        for each deal, holding periods is a list of tuple (start_date, end_date)
    market_data_tgt_corrected:
        
    market_data_acq_corrected: 
    
    deal_no_lst:
        deal picked by any strategy
    weight:
        weighting scheme. now only support value weighted or equally weighted.
    return_raw:
        whether to return the returns and weight of each deal specific on each day.
        
    Returns:
    -------------------------------------
    
    
    """
    # outputs: a series of dataframes, with dates being the index, elements being a dataframe with deal_no being index and return & weights as columns
    # initialize outputs
    start_date = df.da_corrected[deal_no_lst].min()
    end_date = df.dr_corrected[deal_no_lst].dropna().max()
    trading_days = get_trading_day_range(start_date, end_date)
    backtest_res = pd.Series(dtype=object, index=trading_days)
    for day in trading_days:  # initialization
        backtest_res[day] = pd.DataFrame(dtype=float, columns=['returns', 'weights'])
        
    # inputs: market data for tgt and acq, holding period for each deal as a list of (start, end)
    # inputs: df (deal_no, stock, terms), market_data, deal_no_lst, holding_periods
    ## modify holding periods for competing deals?
    
    for deal_no in tqdm(deal_no_lst):
        if not df.stock[deal_no]: # cash deals
            if not holding_periods[deal_no]:  # no holding periods
                continue
            # there are holding periods
            for (start, end) in holding_periods[deal_no]:
                # get slice for market data within this period
                market_data_tgt_single = market_data_tgt_corrected[deal_no][start:end]
                for date in market_data_tgt_single.index:
                    # weight is now mktcap 
                    try:
                        backtest_res[date].loc[deal_no] = [market_data_tgt_single.ret[date], market_data_tgt_single.mktcap_prev[date]]
                    except:
                        continue
        else: # stock deals
            if not holding_periods[deal_no]:  # no holding periods
                continue
            # number of shares
            stock_term = df.stock_term[deal_no]
            # there are holding periods
            for (start, end) in holding_periods[deal_no]:
                # get slice for market data within this period
                market_data_tgt_single = market_data_tgt_corrected[deal_no][start:end]
                market_data_acq_single = market_data_acq_corrected[deal_no][start:end]
                for date in market_data_tgt_single.index:
                    try:
                        return_tgt, return_acq = market_data_tgt_single.ret[date], market_data_acq_single.ret[date]
                        prc_tgt_adj_prev, prc_acq_adj_prev = market_data_tgt_single.prc_adj_ann_prev[date], market_data_acq_single.prc_adj_ann_prev[date]
                        # r_t = r_t^T - (delta * P_{t-1}^A) / P_{t-1}^T * r_t^A
                        return_long_short = return_tgt - return_acq * (stock_term * prc_acq_adj_prev) / prc_tgt_adj_prev
                        backtest_res[date].loc[deal_no] = [return_long_short, market_data_tgt_single.mktcap_prev[date]]
                    except:
                        continue
    if weight == 'value':
        daily_ret_vw = apply_func_to_ser_df(backtest_res, compute_weighted_average)
    elif weight == 'equal':
        daily_ret_vw = apply_func_to_ser_df(backtest_res, compute_weighted_average, weight='equal')
    gc.collect()
    return daily_ret_vw if not return_raw else (daily_ret_vw, backtest_res)


################################
## mutual funds
################################

def get_fund_fundnames_CRSP(id_no, id_type='crsp_fundno', date = None, 
                              cols=['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], 
                              db=None):
    """
    get fundnames file of a mutual fund from the `crsp_q_mutualfunds.fund_names`, by its ticker / crsp_fundno.
    
    all the available columns are 
    ['cusip8', 'crsp_fundno', 'chgdt', 'chgenddt', 'crsp_portno',
       'crsp_cl_grp', 'fund_name', 'ticker', 'ncusip', 'mgmt_name', 'mgmt_cd',
       'mgr_name', 'mgr_dt', 'adv_name', 'open_to_inv', 'retail_fund',
       'inst_fund', 'm_fund', 'index_fund_flag', 'vau_fund', 'et_flag',
       'delist_cd', 'header', 'first_offer_dt', 'end_dt', 'dead_flag',
       'merge_fundno']
    
    can add the filter that the fund is listed on a given date.
       
    Parameters:
    -----------------------------
    id_no: supports ticker / crsp_fundno.
    
    id_type: {'ticker', 'crsp_fundno'}
    
    date:
        The fund is listed on this date.
    cols: list
        
    db: 
    
    Returns:
    -----------------------------
    DataFrame
    
    """
    command = f"select {', '.join(cols)} from crsp_q_mutualfunds.fund_names where {id_type} = "
    if id_type == 'crsp_fundno':
        command += f" {id_no}"
    elif id_type == 'ticker':
        command += f" '{id_no}'"
    else:
        print('Unsupported identifier.')
        return None
    
    if pd.notna(date):
        command += f" and chgdt <= '{str(date)}' and chgenddt >= '{str(date)}'"
    return db.raw_sql(command)

# checked
def convert_fundnames_to_crsp_fundno(fundnames, return_names=False):
    """
    extract crsp_fundno and fund information from a fundnames file.
    
    Pay attention to 'Multiple permnos'. 
    Output this string if return a value, and the string appears in comnam if return a series.
    
    Parameters:
    ------------------------------------
    fundnames:
    
    return_names: boolean, default False
        if True return a series of ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], otherwise just the crsp_fundno number
    
    Returns:
    ------------------------------------
    float or Series
    """
    
    index = ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt']
    if fundnames.empty:   # no stocknames
        return pd.Series(dtype=object, index=index) if return_names else np.nan
    elif len(fundnames.crsp_fundno.unique()) > 1:    # multiple permnos
#         print('Multiple permnos. Check.')
        return pd.Series([np.nan, np.nan, 'Multiple permnos', np.nan, np.nan, np.nan, np.nan], index=index) if return_names else 'Multiple permnos'
    else:     # unique permno
        if return_names:
            ser = fundnames.iloc[-1][index]
            ser.name = None
            return ser
        return fundnames.iloc[-1]['crsp_fundno']
    
def get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):
    """
    get the crsp_fundno of a fund by its ticker.
    
    Parameters:
    ------------------------------------
    id_no: support ticker
    
    id_type: {'ticker'}
    
    date: 
        The fund is listed on this date.
    return_names: boolean, default False
        if True return a series of ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], otherwise just the crsp_fundno number
    db:
    
    
    Returns:
    ------------------------------------
    float or Series
    """
    fundnames = get_fund_fundnames_CRSP(id_no, id_type=id_type, date = date, db=db)
    return convert_fundnames_to_crsp_fundno(fundnames, return_names)

def get_fund_monthly_return_CRSP(id_no, id_type='crsp_fundno', start_month='1900-01', end_month='2030-01', db=None):
    """
    get the monthly return series of a mutual fund, by its permno or ticker.
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'crsp_fundno', 'ticker'}, default 'crsp_fundno'
    
    start_month: pd.Period, or string. default '1900-01'.
    
    end_month: pd.Period, or string. default '2030-01'. 

    db: 
    
    Returns:
    -----------------------------
    Series
    """
    start, end = pd.Period(start_month, 'M'), pd.Period(end_month, 'M')
    
    # convert ticker to crsp_fundno
    if id_type == 'ticker':
        date = str(start+1)+'-01' if start_month != '1900-01' else None
        crsp_fundno = get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if pd.isna(crsp_fundno):
            print("cannot find crsp_fundno.")
            return pd.DataFrame(dtype=float, columns=['mret'])
    elif id_type == 'crsp_fundno':
        crsp_fundno = id_no
    else:
        print('Unrecognized identifier.')
        return pd.DataFrame(dtype=float, columns=['mret'])
    
    #    
    start_date, end_date = str(start)+'-01', str(end+1)+'-01'
    
    #
    command = f"select caldt, mret from crsp_q_mutualfunds.monthly_returns where crsp_fundno = {crsp_fundno} and caldt >= '{start_date}' and caldt <='{end_date}'"
    
    #
    market_data = db.raw_sql(command).sort_values('caldt')#.set_index('date').sort_index() 
    return to_monthly_period_index(market_data, 'caldt').mret

def get_fund_daily_return_CRSP(id_no, id_type='crsp_fundno', start_date='1900-01-01', end_date='2030-01-01', db=None):
    """
    get the daily return series of a mutual fund, by its permno or ticker.
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'crsp_fundno', 'ticker'}, default 'crsp_fundno'
    
    start_date: datetime.date, or string. default '1900-01-01'.
    
    end_date: datetime.date, or string. default '2030-01-01'. 

    db: 
    
    Returns:
    -----------------------------
    Series
    """
    # convert ticker to crsp_fundno
    if id_type == 'ticker':
        date = start_date if start_date != '1900-01-01' else None
        crsp_fundno = get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if pd.isna(crsp_fundno):
            print("cannot find crsp_fundno.")
            return pd.DataFrame(dtype=float, columns=['dret'])
    elif id_type == 'crsp_fundno':
        crsp_fundno = id_no
    else:
        print('Unrecognized identifier.')
        return pd.DataFrame(dtype=float, columns=['dret'])
    
    #
    command = f"select caldt, dret from crsp_q_mutualfunds.daily_returns where crsp_fundno = {crsp_fundno} and caldt >= '{start_date}' and caldt <='{end_date}'"
    
    #
    market_data = db.raw_sql(command).rename(columns={'caldt':'date'}).set_index('date').sort_values('date').dret#.set_index('date').sort_index() 
    return market_data



###################
## competing 
###################
def create_compete_group_no(df):
    """
    create competing deal group number. <df> should include columns <ttic> <cha> <competecode>.
    deals in the same group must have the same target company.
    
    Returns:
    -----------------------------------
    a series with the competing group numbers.
    """
    compete_group_no = pd.Series(np.nan, index=df.index, name='compete_group_no')
    no = 0
    for i in df.index[df.cha.eq(1)]:
        if not pd.isna(compete_group_no[i]): # has been assigned a group
            continue
        code_lst = [int(code) for code in df.competecode[i].split('\n')]
        # competing deal in the dataset, and the same target
        code_lst = [code for code in code_lst if code in df.index and df.ttic[i]==df.ttic[code]]
        if len(code_lst)==0:  
            continue
        code_lst.append(i)   # the compeing group
        if len(compete_group_no[code_lst].value_counts())==0: # all the deals in the group has not been assigned no
            compete_group_no[code_lst] = no
            no += 1
        elif len(compete_group_no[code_lst].value_counts())==1: # some deals in the group has been assigned no
            compete_group_no[code_lst] = compete_group_no[code_lst].dropna().iloc[0]
        else:    # deals in the same group are assigned different group numbers. error
            compete_group_no[code_lst] = -1
    return compete_group_no


def create_compete_status_code_single_group(statc_series, dw_series):
    """
    assign competing deal status codes to one single competing group.
    
    compete status codes:
    - 0: winner in a competition and completes the deal
    - 1: winner in a competition and not completes the deal
    - 2: loser in a competition
    - 3: winner in a competition, still pending
    - 4: still competing
    - 9: error code
    
    Parameters:
    ----------------------------------------
    statc_series: Series
        a series of the status of the deals in a group
    dw_series: Series
        a series of the withdrawl date of the deals in a group
    
    Returns:
    ----------------------------------------
    a series of competing deal status codes in a group.
    """
    
    if 'C' in statc_series.values and statc_series.value_counts().loc['C'] >= 2: # more than one completed deals in a competing group. error
        return pd.Series(9, index=statc_series.index)
    elif 'C' in statc_series.values and statc_series.value_counts().loc['C'] == 1: # just one completed deal in the group
        return statc_series.replace({'C':0,'W':2, 'P':2})
    elif 'P' in statc_series.values and statc_series.value_counts().loc['P'] >= 2: # no completed and more than 2 pending. still compeing
        return statc_series.replace({'P':4,'W':2})
    elif 'P' in statc_series.values and  statc_series.value_counts().loc['P'] == 1: # no completed and only one pending.
        return statc_series.replace({'P':3,'W':2})
    else:       # statc_series.value_counts().loc['P'] == 0: # no completed and no pending, all withdrawn
        result = pd.Series(2, index=statc_series.index)
        result.iloc[np.argmax(dw_series)] = 1    # the last withdrawn deal is the winner of the competition.
        return result
    return pd.Series(9, index=statc_series.index)

def create_compete_status_code(df):
    """
    create competing deal status codes for the dataset. <df> should include columns <compete_group_no> <dw> <statc>.
    
    compete status codes:
    - 0: winner in a competition and completes the deal
    - 1: winner in a competition and not completes the deal
    - 2: loser in a competition
    - 3: winner in a competition, still pending
    - 4: still competing
    - 9: error code
    
    Returns:
    -----------------------------------
    a series with the competing deal status codes.
    """
    compete_statc_code = pd.Series(np.nan, index=df.index, name='compete_statc_code')
    # map from group no to the indices of deals in that group
    no_to_index = df.groupby('compete_group_no').apply(lambda x:x.index)
    # exclude groups with only one deal
    count = df.compete_group_no.value_counts()
    count = count[count.ge(2)]
    for group_no in count.index: # each group
        index_group = no_to_index[group_no]
        compete_statc_code[index_group] = \
        create_compete_status_code_single_group(df.statc[index_group], df.dw[index_group])
    return compete_statc_code



#################
## payment type
#################

def check_list_of_keys_in_list(key_lst, lst):
    """
    check whether any of the key in <key_lst> is contained in the list <lst>.
    """
    for key in key_lst:
        if key in lst:
            return True
    return False


def create_amend(df):
    """
    create the column of whether the deal is amended. <df> should contain columns <consid> and <synop>
    
    Returns:
    ----------------------------------------
    a Series of 0/1 indicating whether the deal is amended or not. 
    """
    amend_lst = ['sweet', 'amend']    # keys of amendment to search in synopsis
    amend_lst_more = amend_lst + ['Original', 'original', 'previous', 'Previous']   # keys of amendment to search in consid
    amend_synop = df.synop.map(lambda x: check_list_of_keys_in_list(amend_lst, x))
    amend_consid = df.consid.map(lambda x: check_list_of_keys_in_list(amend_lst_more, x), na_action='ignore').fillna(False)
    return (amend_consid | amend_synop).astype(int)  


def create_choice(df):
    """
    create the column of whether the deal term consists of a choice. <df> should contain columns <consid> and <synop>
    
    Returns:
    ----------------------------------------
    a Series of 0/1 indicating whether whether the deal term consists of a choice.
    """
    choice_lst = ['choice', 'Choice', 'choose', 'Choose']    # keys of amendment to search in synopsis
    choice_synop = df.synop.map(lambda x: check_list_of_keys_in_list(choice_lst, x))
    choice_consid = df.consid.map(lambda x: check_list_of_keys_in_list(choice_lst, x), na_action='ignore').fillna(False)
    return (choice_consid | choice_synop).astype(int) 



def extract_all_payment_types(ser):
    """
    returns all the possible payment types in the dataset.
    each element in <ser> is a string of payment types, delimited by \newline
    """
    return np.unique(ser.map(lambda x: x.split('\n'), na_action='ignore').dropna().sum())



def transform_payment_str(string, lst_cash, lst_stock):
    """
    categorize one `consido` string into four groups:
    - 'Cash'
    - 'Common Stock'
    - 'Cash and Common Stock'
    - 'No Cash or Common Stock'.

    strings indicating cash or stock payments are included in <lst_cash> and <lst_stock>
    """
    str_lst = string.split('\n')
    if check_list_of_keys_in_list(lst_cash, str_lst):   # cash payment is included
        if check_list_of_keys_in_list(lst_stock, str_lst):  # stock payment is also included
            return 'Cash and Common Stock'
        else:    # only cash
            return 'Cash'
    elif check_list_of_keys_in_list(lst_stock, str_lst):   # no cash, have stock payment
        return 'Common Stock'
    else:      # no cash or stock
        return 'No Cash or Stock'
    
    

################################
## extract terms from consid
################################



def convert_consid_to_readable(string):
    """
    convert the consid string to more readable format.
    
    - replace any whitespace by one space
    - replace '/ ' and '/  ' by '/'. 
    - replace '. ' by '.'
    - replace ', ' by ' '
    - replace 'USD ' by '$'
    """
    string = " ".join(string.split())
    return string.replace('/ ', '/').replace('/  ', '/').replace('. ', '.').replace(', ', ' ').replace('USD ', '$')

def convert_consid_single_to_easy_to_parse(consid_single):
    """
    convert a single consideration string to an easy-to-parse format
    """
    consid_single = consid_single.strip()
    consid_single = consid_single.replace(' sh ', ' shs ')
    consid_single = consid_single.replace(' sh/', ' shs com/')
    consid_single = consid_single.replace(' and ', ' plus ')
    consid_single = consid_single.replace(' ord ', ' com ')
    consid_single = consid_single.replace(' ord/', ' com/')
    consid_single = consid_single.replace(' com sh/', ' shs com/')
    consid_single = consid_single.replace(' com shs/', ' shs com/')
    consid_single = consid_single.replace(' ADRs', ' shs com')
    consid_single = consid_single.replace(' ADR', ' shs com')
    consid_single = consid_single.replace(' American depositary shares', ' shs com')
    consid_single = consid_single.replace(' American depositary share', ' shs com')
    consid_single = consid_single.replace('Cl A ', '')
    consid_single = consid_single.replace(' Series A ', '')
    consid_single = consid_single.replace('An estimated ', '')
    consid_single = consid_single.replace('Class A ', '')
    consid_single = consid_single.replace(' in ', ' ')
#     consid_single = consid_single.replace('UAI ', '')
    consid_single = consid_single.replace(' per share ', ' ')
    consid_single = consid_single.replace(' per ', ' ')
    consid_single = consid_single.removeprefix("plus ")
    consid_single = consid_single.replace(' newly issued ', ' ')
    consid_single = consid_single.replace(' newly-issued ', ' ')
    consid_single = consid_single.replace(' new ', ' ')
    consid_single = consid_single.replace(' co ', ' ')
    consid_single = consid_single.replace(' of ', ' ')
    consid_single = consid_single.replace(' US ', ' ')
    consid_single = consid_single.replace(' sh comA/', ' shs com/')
    consid_single = consid_single.replace(' including ', ' plus ')
    consid_single = consid_single.replace(' A', '')
    consid_single = consid_single.replace(' B', '')
    consid_single = consid_single.replace(' Class', '')
    consid_single = consid_single.replace(' class', '')
    
#     consid_single = consid_single.replace(' C', '')
    consid_single_lst = consid_single.split()
    if len(consid_single_lst) >= 2 and consid_single_lst[1]=='cash' and consid_single_lst[0][0]!='$':
        consid_single = '$' + consid_single
        
    if ' plus the assumption' in consid_single:
        consid_single = consid_single.split(" plus the assumption")[0]
    return consid_single


def extract_substr_before_key(key, string):
    """
    if <key> is contained in <string>, returns the substring before the first <key> appearance.
    """
    if key in string:
        return string[:string.find(key)]
    return False

def extract_substr_before_list_of_keys(key_lst, string):
    """
    for the first key in <key_lst> that is contained in the <string>, returns  the substring before the first key appearance.
    """
    for key in key_lst:
        temp = extract_substr_before_key(key, string)
        if temp:
            return temp
    return False


key_list = ['/sh com','/ sh com', '/shs com', '/ shs com', '/com', '/ com', '/sh', '/sh ', '/coma', '/sh ord', '/Class A sh com', '/ Class A sh com'] # '/sh com,',  '/sh com A;', '/ com;',, '/sh ,'  '/sh com A','/sh comA', '/com A',
def extract_terms_from_list_of_consids(key_list, consid_list):
    """
    for a list of considerations, extract the terms
    """
    terms_list = []
    for consid_single in consid_list:
        temp = extract_substr_before_list_of_keys(key_list, consid_single)
        if temp:
            terms_list.append(convert_consid_single_to_easy_to_parse(temp))
    return terms_list

def extract_term_from_consid(consid_ser):
    """
    extract easy-to-parse term string from consid
    """
    consid_lst_ser = consid_ser.str.split(";")
    terms_lst_ser = consid_lst_ser.map(lambda x: extract_terms_from_list_of_consids(key_list, x), na_action='ignore')
    terms_ser = terms_lst_ser.map(lambda x: x[0] if len(x)>0 else np.nan, na_action='ignore')
    return terms_ser


def extract_cash_stock_from_term(term):
    """
    extract cash and stock terms from term string
    """
    index = ['cash_term', 'stock_term', 'payment_type']
    if pd.isna(term):
        return pd.Series([np.nan, np.nan, np.nan], index=index)
    # cash only, "$2.2 cash"
    if len(term.split()) <= 2 and re.search("^\$.* cash$", term) != None:
        cash = term.removeprefix('$').removesuffix(' cash')
        try:
            return pd.Series([float(cash), 0, 'Cash'], index=index)
        except:
            pass
        
    # cash only, '$21'
    if (re.search('\s', term) == None) and (term[0]=='$'):
        cash = term[1:]
        try:
            return pd.Series([float(cash), 0, 'Cash'], index=index)
        except:
            pass
        
    # stock only, '2.1 shs com', '$10 shs com'
    if len(term.split()) <= 3 and (re.search('.* shs com$', term) != None or re.search('.* com$', term) != None or re.search('.* shs$', term) != None):
        stock = term.removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        try:
            return pd.Series([0, float(stock), 'Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([0, float(stock[1:]), 'Common Stock, fixed dollar'], index=index)
        except:
            pass

    # combination, '$8.5 cash plus .85 shs (or shs com, or com)'
    if (re.search('^\$.* cash plus .* shs$', term) != None) or (re.search('^\$.* cash plus .* shs com$', term) != None) or (re.search('^\$.* cash plus .* com$', term) != None):
        term_new = term.removeprefix('$').removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        cash = term_new.split()[0]
        stock = term_new.split()[-1]
        try:
            return pd.Series([float(cash), float(stock), 'Cash and Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar'], index=index)
        except:
            pass  
        
    # combination, '0.2109 shs com plus $9 cash', '$12.546 com plus $12.054 cash' (fixed dollar)
    if (re.search('.* com plus \$.* cash$', term) != None) or (re.search('.* shs com plus \$.* cash$', term) != None) or (re.search('.* shs plus \$.* cash$', term) != None):
        term_new = term.removesuffix(" cash")
        cash = term_new.split()[-1][1:]
        stock = term_new.split()[0]
        try:
            return pd.Series([float(cash), float(stock), 'Cash and Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar'], index=index)
        except:
            pass   
        
    # combination, fixed dollar, '$15 cash plus com'
    if (re.search('^\$.* cash plus com$', term) != None) or (re.search('^\$.* cash plus sh com$', term) != None) or (re.search('^\$.* cash plus shs com$', term) != None):
        cash = term.split()[0][1:]
        try:
            return pd.Series([float(cash), 0, 'Cash and Common Stock, fixed dollar'], index=index)# if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
        except:
            pass 
        
    return pd.Series([np.nan, np.nan, 'parse failed'], index=index)


# def extract_cash_stock_from_term(term):
#     """
#     extract cash and stock terms from term string
#     """
#     # cash only, "$2.2 cash"
#     if len(term.split()) <= 2 and re.search("^\$.* cash$", term) != None:
#         cash = term.removeprefix('$').removesuffix(' cash')
#         try:
#             return (float(cash), 0, 'Cash')
#         except:
#             pass
        
#     # cash only, '$21'
#     if (re.search('\s', term) == None) and (term[0]=='$'):
#         cash = term[1:]
#         try:
#             return (float(cash), 0, 'Cash')
#         except:
#             pass
        
#     # stock only, '2.1 shs com', '$10 shs com'
#     if len(term.split()) <= 3 and (re.search('.* shs com$', term) != None or re.search('.* com$', term) != None or re.search('.* shs$', term) != None):
#         stock = term.removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
#         try:
#             return (0, float(stock), 'Common Stock') if stock[0]!='$' else (0, float(stock[1:]), 'Common Stock, fixed dollar')
#         except:
#             pass

#     # combination, '$8.5 cash plus .85 shs (or shs com, or com)'
#     if (re.search('^\$.* cash plus .* shs$', term) != None) or (re.search('^\$.* cash plus .* shs com$', term) != None) or (re.search('^\$.* cash plus .* com$', term) != None):
#         term_new = term.removeprefix('$').removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
#         cash = term_new.split()[0]
#         stock = term_new.split()[-1]
#         try:
#             return (float(cash), float(stock), 'Cash and Common Stock') if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
#         except:
#             pass  
        
#     # combination, '0.2109 shs com plus $9 cash', '$12.546 com plus $12.054 cash' (fixed dollar)
#     if (re.search('.* com plus \$.* cash$', term) != None) or (re.search('.* shs com plus \$.* cash$', term) != None) or (re.search('.* shs plus \$.* cash$', term) != None):
#         term_new = term.removesuffix(" cash")
#         cash = term_new.split()[-1][1:]
#         stock = term_new.split()[0]
#         try:
#             return (float(cash), float(stock), 'Cash and Common Stock') if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
#         except:
#             pass   
        
#     # combination, fixed dollar, '$15 cash plus com'
#     if (re.search('^\$.* cash plus com$', term) != None) or (re.search('^\$.* cash plus sh com$', term) != None) or (re.search('^\$.* cash plus shs com$', term) != None):
#         cash = term.split()[0][1:]
#         try:
#             return (float(cash), 0, 'Cash and Common Stock, fixed dollar')# if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
#         except:
#             pass 
        
#     return (np.nan, np.nan, 'parse failed')






# def index_higher_value(df):
#     """
#     for a two-column dataFrame, return a tuple (index_1, index_2), where for rows in index_1, the first column is no smaller than the second column,
#     and for rows in index_2, the first column is smaller than the second column.
#     """
#     dff = df.dropna()
#     return dff.index[dff.iloc[:, 0].ge(dff.iloc[:, 1])], dff.index[dff.iloc[:, 0]<(dff.iloc[:, 1])]




############################
## CRSP
############################



def find_mutual_fund_name_CRSP(db, ticker):
    """
    
    """
    command = """select ticker, crsp_fundno,fund_name, mgmt_name,
    first_offer_dt, chgdt, chgenddt, et_flag, ncusip
    from crsp_q_mutualfunds.fund_names where ticker = """ +\
    f"'{ticker}'"
    return db.raw_sql(command)


def get_mutual_fund_ret_mothly_CRSP(db, fundno, start_month='1000-01', end_month='3000-01'):
    """
    
    """
    # convert month to date
    start_month = str(start_month) + '-01'
    end_month = str(end_month) + '-31'
    #
    command = f'select caldt, mret from crsp_q_mutualfunds.monthly_returns where crsp_fundno = {str(fundno)}' + \
    f" and caldt >= '{start_month}' and caldt <= '{end_month}'"
    #
    df_ret = db.raw_sql(command).dropna()
    return to_monthly_period_index(df_ret, 'caldt')['mret']    


def get_stock_ret_monthly_CRSP(db, permno, start_month='1000-01', end_month='3000-01'):
    """
    
    """
    # convert month to date
    start_month = str(start_month) + '-01'
    end_month = str(end_month) + '-31'
    #
    command = """
    select date, permno, ret, vol, shrout, prc, cfacshr, cfacpr from crsp_m_stock.msf where permno = 
    """ + f"{str(permno)} and date >= '{start_month}' and date <= '{end_month}'"
    #
    df_ret = to_monthly_period_index(db.raw_sql(command), 'date') #.dropna()
    df_ret.ret = df_ret.ret.fillna(0)
    df_ret.prc = abs(df_ret.prc.fillna(method='ffill'))
    return df_ret

def get_stock_ret_daily_CRSP(db, permno, start_date='1000-01-01', end_date='3000-01-01'):
    """
    
    """
    # convert month to date
    start_date = str(start_date)
    end_date = str(end_date)
    #
    command = """
    select date, permno, ret, vol, shrout, prc, cfacshr, cfacpr from crsp_m_stock.dsf where permno = 
    """ + f"{str(permno)} and date >= '{start_date}' and date <= '{end_date}'"
    #
    df_ret = db.raw_sql(command)  #.dropna()
    df_ret.ret = df_ret.ret.fillna(0)
    df_ret.prc = abs(df_ret.prc.fillna(method='ffill'))
    return df_ret.set_index('date')


def get_stock_ret_adjprc_mktcap_daily_CRSP(db, permno, start_date='1000-01-01', end_date='3000-01-01'):
    """
    
    """
    # convert month to date
    start_date = str(start_date)
    end_date = str(end_date)
    #
    command = """
    select date, ret, shrout, prc, cfacpr from crsp_m_stock.dsf where permno = 
    """ + f"{str(permno)} and date >= '{start_date}' and date <= '{end_date}'"
    #
    df_ret = db.raw_sql(command)  #.dropna()
    df_ret.ret = df_ret.ret.fillna(0)
    df_ret.prc = abs(df_ret.prc.fillna(method='ffill'))
    df_ret['mktcap'] = df_ret.prc * df_ret.shrout / 1e3
    df_ret['prc_adj'] = df_ret.cfacpr.iloc[0] * df_ret.prc / df_ret.cfacpr
    return df_ret.set_index('date')[['ret', 'prc_adj', 'mktcap']]








####################
## factor models
####################

# checked
def get_ff_factors_monthly_CRSP(start_month='1900-01', end_month='2050-01', db=None):
    """
    get FF four factors from CRSP.
    <start_month> and <end_month> can be str '2000-01' or monthly period data type.
    
    Parameters:
    -----------------------------------------
    start_month: str 'YYYY-MM' or Period
    
    end_month
    
    db:
    
    Returns:
    -----------------------------------------
    DataFrame with columns rf, mktrf, smb, hml, umd
    
    """
    # convert month to date
    start_month = str(start_month) + '-01'
    end_month = str(end_month) + '-01'
    # date in `ff_all.factors_monthly` is the 1st day of each month
    command = 'select date, rf, mktrf, smb, hml, umd from ff_all.factors_monthly where ' + \
    f"date >= '{start_month}' and date <= '{end_month}'"
    #
    df_ff = db.raw_sql(command).dropna()
    return to_monthly_period_index(df_ff, 'date')


def get_mkt_ret_monthly_CRSP(start_month='1900-01', end_month='2050-01', db=None):
    """
    get CRSP total market return.
    <start_month> and <end_month> can be str '2000-01' or monthly period data type.
    
    Parameters:
    -----------------------------------------
    start_month: str 'YYYY-MM' or Period
    
    end_month:
    
    db:
    
    Returns:
    -----------------------------------------
    
    """
    # convert month to date
    start_month = str(start_month) + '-01'
    end_month = str(pd.Period(end_month, 'M')+1) + '-01'
    # date in `ff_all.factors_monthly` is the 1st day of each month
    command = 'select date, vwretd from crsp_m_stock.msi where ' + \
    f"date >= '{start_month}' and date <= '{end_month}'"
    #
    mkt_ret = db.raw_sql(command).dropna()
    return dsk.to_monthly_period_index(mkt_ret, 'date').squeeze()

# checked
def to_monthly_period_index(df, col_dt):
    """
    convert a column <col_dt> of dates, into monthly Period dtype, and set it as the index of <df>.
    The original column is dropped
    
    Parameters:
    ---------------------------------
    df: DataFrame

    col_dt: string
        column name of <df>. The column is a date series to be converted to monthly Periods.
        
    Returns:
    ---------------------------------
    a DataFrame with monthly Period as index.
    """
    return (
        df.assign(month = pd.to_datetime(df[col_dt]).dt.to_period('M'))
        .drop(columns = [col_dt])
        .set_index('month')
    )


def estimate_factor_model(ret_ser, model='four-factor', db=None):
    """
    estimate the parameters of FF factor models for a monthly return series <ret_ser>
    supports model = 'CAPM', 'three-factor', 'four-factor'
    
    Parameters:
    --------------------------------------------------
    ret_ser: Series
        monthly return series
    model: {'CAPM', 'three-factor', 'four-factor'}
    
    db:
    
    
    Returns:
    --------------------------------------------------
    statsmodel fit result
    
    """
    # factor correspondence
    cols_dict = {'CAPM': ['mktrf'], 'three-factor': ['mktrf', 'smb', 'hml'], 'four-factor': ['mktrf', 'smb', 'hml', 'umd']}
    # 
    df_ff = get_ff_factors_monthly_CRSP(ret_ser.index[0], ret_ser.index[-1], db)
    #
    if ret_ser.name == None:
        ret_ser.name = 'return'
    df_ff_ret = pd.concat([df_ff, ret_ser], axis=1)
    name = df_ff_ret.columns[-1]

    # minus rf rate
    df_ff_ret[name+'rf'] = df_ff_ret[name].sub(df_ff.rf)
    #
    res = sm.OLS(df_ff_ret[name+'rf'], sm.add_constant(df_ff_ret[cols_dict[model]], has_constant='add'), missing = 'drop').fit()
#     print(res.summary())
    return res

# def estimate_factor_model(ret_ser, model='four-factor', db=None):
#     """
#     estimate the parameters of FF factor models for a monthly return series <ret_ser>
#     supports model = 'CAPM', 'three-factor', 'four-factor'
    
#     Parameters:
#     --------------------------------------------------
#     ret_ser: Series
#         monthly return series
#     model: {'CAPM', 'three-factor', 'four-factor'}
    
#     db:
    
    
#     Returns:
#     --------------------------------------------------
#     statsmodel fit result
    
#     """
#     df_ff = get_ff_factors_monthly_CRSP(ret_ser.index[0], ret_ser.index[-1], db)
#     if len(df_ff) != len(ret_ser):
#         print("different length. maybe the end month is too near.")
#         return
#     y = ret_ser - df_ff['rf']
#     if model == 'CAPM':
#         X = df_ff[['mktrf']]
#     elif model == 'three-factor':
#         X = df_ff[['mktrf', 'smb', 'hml']]
#     elif model == 'four-factor':
#         X = df_ff[['mktrf', 'smb', 'hml', 'umd']]
#     res = sm.OLS(y, sm.add_constant(X, has_constant='add'), missing = 'drop').fit()
#     print(res.summary())
#     return res





def compute_mean_divided_by_std(ser):
    """
    compute mean divided by std for a series.
    """
    return ser.mean()/ser.std()


def compute_sharpe_of_monthly_return(ret_ser, db=None):
    """
    for a monthly return series or dataFrame. Compute the sharpe ratio.
    risk-free rate and market return is pulled from 'ff_all_monthly_factors'
    
    Parameters:
    --------------------------------------------------
    ret_ser: Series or DataFrame
        if a DataFrame, each column is a return series.
    db:
    

    Returns:
    --------------------------------------------------
    """
    df_ff = get_ff_factors_monthly_CRSP(ret_ser.index[0], ret_ser.index[-1], db=db)
#     if len(df_ff) != len(ret_ser):
#         print("different length. maybe the end month is too near.")
#         return
    # convert series to df
    ret_df = pd.DataFrame(ret_ser) if isinstance(ret_ser, pd.Series) else ret_ser
    # minus rf
    ret_df_minus_rf = ret_df.sub(df_ff.rf, axis=0)
    # include mktrf
    ret_df_mkt_minus_rf = pd.concat([ret_df_minus_rf, df_ff.mktrf], axis=1)
    
    return ret_df_mkt_minus_rf.apply(compute_mean_divided_by_std)*np.sqrt(12)
           
#     if isinstance(ser, pd.Series):
#         ser
#         df = ser - df_ff.rf
#         df.name = ser.name
#     elif isinstance(df, pd.DataFrame):
#         dff = ser.sub(df.rf, axis=0)
#         df = ser.sub(df_ff.rf, axis=0) - df_ff[['rf']].values
    
#     df = pd.concat([df, df_ff.mktrf], axis=1)
#     return df.apply(compute_mean_divided_by_std, axis=0)*np.sqrt(12)




###################
## Other helpers
##################

def transform_tuple_ser_to_Dataframe(ser):
    return pd.DataFrame(list(ser.values), index=ser.index)



def insert_cols(df, loc_names, new_colnames, value):
    """
    insert columns to <df>. 
    insert each column of <value> at the location in <loc_names> (as column name), with new column name in <new_colnames>
    
    <loc_names>, <new_colnames> are strings, value is a Series; 
    or <loc_names>, <new_colnames> are lists of strings, value is a DataFrame.
    
    Examples:
    ------------------------------------
    >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['Mon', 'Tues'], columns=['a', 'b'])
    >>> df
          a  b
    Mon   1  2
    Tues  3  4
    >>> insert_cols(df, 'b', 'new_col', pd.Series([5,6], index=['Mon', 'Tues']))
    >>> df
          a  new_col  b
    Mon   1        5  2
    Tues  3        6  4
    >>> insert_cols(df, ['new_col', 'b'], ['new_col_1', 'new_col_2'], pd.DataFrame([[7, 8], [9, 10]], index=['Mon', 'Tues']))
    >>> df
          a  new_col_1  new_col  new_col_2  b
    Mon   1          7        5          8  2
    Tues  3          9        6         10  4
    """
    if isinstance(loc_names, str):
        df.insert(df.columns.get_loc(loc_names), new_colnames, value)
    else:
        for i, (loc_name, new_colname) in enumerate(zip(loc_names, new_colnames)):
            df.insert(df.columns.get_loc(loc_name), new_colname, value.iloc[:, i])
    return 



def yearly_summary(df_input):
    """
    Given status <statc> and date of announcement <da>, 
    output the annual total/failure count and failure rate.
    """
    df = df_input.copy()
    if not is_integer_dtype(df['statc']):
        df['statc'].replace({'C':0, 'W':1, 'P':0}, inplace=True)
    df['ann_yr'] = [date.year for date in df['da']]
    ret = pd.DataFrame({'total':df['ann_yr'].value_counts().sort_index(),\
                        'fail': df.groupby('ann_yr')['statc'].sum().sort_index(),\
                        'fail_rate':df.groupby('ann_yr')['statc'].mean().sort_index()})
    return ret



def set_from_series(series):
    """
    Every element of series is a string delimited by a new line. Return a set of all the possible values.
    """
    ret = set()
    for items in series:
        for item in items.split('\n'):
            ret.add(item)
    return ret
    
    
def trans_names(series, neglect_lst, delete_lst, replace_dict):
    """
    split by '\n'. Cannot have element in the <delete_lst>. Neglect the words in <neglect_lst>.
    """
    ret = []
    for names in series:
        # replace
        new = set(pd.Series(names.split('\n')).replace(replace_dict))
        # delete
        if len(new.intersection(set(delete_lst))) > 0:
            ret.append(np.nan)
            continue
        # check cash and common stock
        if len(new.intersection(set(['Cash', 'Common Stock']))) == 0:
            ret.append(np.nan)
            continue
        # neglect
        new = list(new.difference(set(neglect_lst)))
        new.sort()
        if len(new) > 1:
            new[:-1] = [item + ", " for item in new[:-1]]
        ret.append("".join(new))
    return ret


def check_subst(st, subst_lst):
    """
    check whether the string <st> contains any of the word in the list <subst_lst> of string 
    """
    for subst in subst_lst:
        if subst in st:
            return True
    return False




def generate_timeline_series(date_series, days_list):
    """
    Given a pd.Series <date_series> of datetime.date object (ex. an array of announce dates), 
    and an int list <days_list> denoting the time points on the whole timeline that we need 
    (days=1 denotes the nearest next trading day, days=0 denotes the nearest trading day before),
    
    returns the timeline <df> as a DataFrame. Each row of <df> corrsponds to one timeline.
    
    Used for pulling out pricing information in the database.
    
    For example, <date_series> = announce date, <days_list> = [0, 1, 5], gives us the date 
    before announcement, post announcement, and 1 wk after announcement.
    """
    # to numpy array
    days_arr = np.array(days_list)
    date_series = pd.Series(date_series)
    # determine the scope of dates. Translating trading days to calendar days by ( * 1.6 +- 5 )
    # should be (* 7 / 5 = * 1.4). ( * 1.6 +- 5 ) is just for satefy
    start_date = date_series.min() + timedelta(days=np.min([days_arr.min(),0])*1.6-5)
    end_date = date_series.max() + timedelta(days=np.max([days_arr.max(),0])*1.6+5)
    # get `NYSE` calendar
    nyse = mcal.get_calendar('NYSE')
    # returns a pd.Series of datetime.date objects
    days_series = nyse.valid_days(start_date=start_date, end_date=end_date).date
    
    # find day 0, i.e. the neareast next trading day
    ind_arr = np.searchsorted(days_series, date_series) - 1
    # plus the date offset
    ind = ind_arr.reshape((-1, 1)) + days_arr
    
    df = pd.DataFrame(days_series[ind])
    df.index = date_series.index
    df.columns = days_arr
    return df


def generate_timeline_single(date_single, days_list):
    """
    Given a datetime.date object <date_single> (ex. an array of announce dates), 
    and an int list <days_list> denoting the time points on the whole timeline that we need 
    (days=1 denotes the nearest next trading day, days=0 denotes the nearest trading day before),
    
    returns the timeline <df> as a DataFrame, which has only one row.
    
    Used for pulling out pricing information in the database.
    
    For example, <date_single> = announce date, <days_list> = [0, 1, 5], gives us the date 
    before announcement, post announcement, and 1 wk after announcement.
    """
    # to numpy array
    days_arr = np.array(days_list)
    # determine the scope of dates. Translating trading days to calendar days by ( * 1.6 +- 5 )
    # should be (* 7 / 5 = * 1.4). ( * 1.6 +- 5 ) is just for satefy
    start_date = date_single + timedelta(days=days_arr.min()*1.6-5)
    end_date = date_single + timedelta(days=days_arr.max()*1.6+5)
    # get `NYSE` calendar
    nyse = mcal.get_calendar('NYSE')
    # returns a pd.Series of datetime.date objects
    days_series = nyse.valid_days(start_date=start_date, end_date=end_date).date
    
    # find day 0, i.e. the neareast next trading day
    ind = np.searchsorted(days_series, date_single) - 1
    # plus the date offset
    ind = ind.reshape((-1, 1)) + days_arr
    
    df = pd.DataFrame(days_series[ind])
    df.columns = days_arr
    return df


def get_n_trailing_average_single(cusip_single, date_single, n, conn):
    """
    Given the cusip of a stock, the date as a datetime.date object, window size <n>,
    and CRSP access,
    returns the n trailing average of the closed price before the date.
    """
    #announce_date = date(announce_date)
    [start, end] = generate_timeline_single(date_single, [-(n-1), 0]).values[0]
    sql_command = "select prc from crsp_a_stock.dsf where cusip = '" + cusip_single +"' and date between '" +\
                str(start) + "' and '" + str(end) + "'"
    return conn.raw_sql(sql_command).mean()[0]

"""
def generate_date_timeline(announce_date):
    \"""
    given a pandas datetime Series 'announce_date', 
    returns the 20 (trading) days before, the last trading day before, 
    and the next trading day after announcement
    \"""
    start_date = announce_date.min() - DateOffset(months = 2)
    end_date = announce_date.max() + DateOffset(years = 2)
    nyse = mcal.get_calendar('NYSE')
    days_series = nyse.valid_days(start_date=start_date, end_date=end_date)
    days_series = days_series.tz_localize(None)
    index_arr = np.zeros(len(announce_date), dtype = np.int16)
    for i, date in enumerate(announce_date):
        index_arr[i] = days_series.get_loc(date, method='bfill')
    post_announce_date = days_series[index_arr]
    pre_announce_date = days_series[index_arr-1]
    pre_20_announce_date = days_series[index_arr-20]
    df = pd.DataFrame({'post_announce':post_announce_date, 
                  'pre_announce':pre_announce_date, 
                  'pre_20_announce':pre_20_announce_date})
    df.index = announce_date.index
    return df
"""


def change_order(list_all, list_first):
    """
    change the order of <list_all> so that the items in <list_first> appears first
    """
    list_others = [x for x in list_all if x not in list_first]
    return list_first + list_others


def print_shape(df):
    """
    Print the shape of a dataset.
    """
    print(f'The dataset is of size {df.shape}.')
    

def print_colnames(df):
    """
    Print the column names of a dataset.
    """
    print('The columns of the dataset are: ' + str(list(df.columns)))
    

def print_cv_scores(scores):
    """
    print the CV scores for different models applied on the IRIS dataset
    """
    print('The cv scores are ' + str(scores))
    print(f'mean: {scores.mean():.4f}, std: {scores.std():.4f}.')


def na_value_counts(df):
    """
    Print the number and percent of missing values in every column.
    """
    count = df.isnull().sum().sort_values(ascending = False)
    count = count[count > 0]
    if count.shape[0] == 0:
        print('There is no missing value.')
        return 
    count.name = 'Total'
    percent = count / df.shape[0]
    percent.name = 'Percent'
    na_frame = pd.concat([count, percent], axis = 1)
    return na_frame


def ROC_curve_display_CV(model, X, y, cv, confid_band = False):
    """
    Plot the ROC curves, AUC metrics in all the cross validation folds. 
    Also includes the average curve, with an optional 1 std confidence band, and average AUC.
    Parameter <cv> can only be a sklearn CV iterator.
    Adapted from examples on the scikit-learn website.
    """
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    _, ax = plt.subplots(figsize = (8, 8))
    for i, (index_train, index_valid) in enumerate(cv.split(X, y)):
        X_train, X_valid, y_train, y_valid = \
        X.iloc[index_train], X.iloc[index_valid], y.iloc[index_train], y.iloc[index_valid]
        model.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(model, 
                                            X_valid,
                                            y_valid,
                                            ax = ax,
                                            name = f'fold {i+1}',
                                            alpha = .3)
        # store the auc score
        aucs.append(viz.roc_auc)
        # interpolate the ROC curve
        tprs.append(np.interp(mean_fpr, viz.fpr, viz.tpr))
    
    aucs = np.array(aucs)
    tprs = np.array(tprs)
    # averaged tpr
    mean_tpr = tprs.mean(axis = 0)
    def insert_0_1(arr, a, b): 
        # insert one <a> at the first element, and append one <b> at the end of a numpy array.
        arr = np.insert(arr, 0, a)
        arr = np.append(arr, b)
        return arr
    mean_fpr = insert_0_1(mean_fpr, 0, 1)
    mean_tpr = insert_0_1(mean_tpr, 0, 1)
    ax.plot(mean_fpr, mean_tpr, 'k', lw = 2,
           label = f'Mean ROC (AUC = {aucs.mean():.2f} $\pm$ {aucs.std():.2f})')
    
    ax.plot([0, 1], [0, 1], 'r--', label = 'Random Guess')
    
    if confid_band:
        std_tpr = tprs.std(axis = 0)
        std_tpr = insert_0_1(std_tpr, 0, 0)
        ax.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0), 
                        np.minimum(mean_tpr + std_tpr, 1),
                        alpha = .3,
                        color = 'gray',
                        label = '$\pm$ 1 std'
                       )
    ax.legend()
    
    return ax


def ROC_curve_display_CV_from_estimator(model, X, y, cv, confid_band = False):
    """
    Plot the ROC curves, AUC metrics in all the cross validation folds. 
    Also includes the average curve, with an optional 1 std confidence band, and average AUC.
    Parameter <cv> can only be a sklearn CV iterator.
    Adapted from examples on the scikit-learn website.
    """
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    _, ax = plt.subplots(figsize = (8, 8))
    for i, (index_train, index_valid) in enumerate(cv.split(X, y)):
        X_train, X_valid, y_train, y_valid = \
        X.iloc[index_train], X.iloc[index_valid], y.iloc[index_train], y.iloc[index_valid]
        #model.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(model[i], 
                                            X_valid,
                                            y_valid,
                                            ax = ax,
                                            name = f'fold {i+1}',
                                            alpha = .3)
        # store the auc score
        aucs.append(viz.roc_auc)
        # interpolate the ROC curve
        tprs.append(np.interp(mean_fpr, viz.fpr, viz.tpr))
    
    aucs = np.array(aucs)
    tprs = np.array(tprs)
    # averaged tpr
    mean_tpr = tprs.mean(axis = 0)
    def insert_0_1(arr, a, b): 
        # insert one <a> at the first element, and append one <b> at the end of a numpy array.
        arr = np.insert(arr, 0, a)
        arr = np.append(arr, b)
        return arr
    mean_fpr = insert_0_1(mean_fpr, 0, 1)
    mean_tpr = insert_0_1(mean_tpr, 0, 1)
    ax.plot(mean_fpr, mean_tpr, 'k', lw = 2,
           label = f'Mean ROC (AUC = {aucs.mean():.2f} $\pm$ {aucs.std():.2f})')
    
    ax.plot([0, 1], [0, 1], 'r--', label = 'Random Guess')
    
    if confid_band:
        std_tpr = tprs.std(axis = 0)
        std_tpr = insert_0_1(std_tpr, 0, 0)
        ax.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0), 
                        np.minimum(mean_tpr + std_tpr, 1),
                        alpha = .3,
                        color = 'gray',
                        label = '$\pm$ 1 std'
                       )
    ax.legend()
    
    return ax







# str_lst = ['/sh com', '/sh com,', '/sh com A', '/sh com A;', '/sh comA', '/ sh com', '/shs com', '/ shs com',
#            '/com', '/ com', '/ com;', '/coma', '/com A',
#            '/sh', '/sh ', '/sh ,', '/sh ord', '/Class A sh com', '/ Class A sh com']

def check_endswith(st, subst_lst):
    for subst in subst_lst:
        if st.endswith(subst):
            return True
    return False

def find_subst_lst(st, subst_lst):
    for subst in subst_lst:
        if st.find(subst) > 0:
            return st.find(subst)
    return -1





# def check_st_lst(st, st_lst):
#     return st in st_lst

# def check_cash_only(st):
#     """
#     $41 cash
#     $21
#     $16 cash and com
#     $25.5 cash plus shs com
#     $5 cash plus the assumption of $171 mil in liabilities plus 0.65 shs com
#     """
#     lst2 = ['and', 'plus']
#     st = st.split()
#     if len(st) == 2 and st[1] == 'cash' and st[0][0] == '$':
#         return float(st[0][1:])
#     if len(st) >= 5 and st[1] == 'cash' and st[0][0] == '$' and st[4]=='assumption':
#         return float(st[0][1:])
#     if len(st) == 1 and st[0][0] == '$':
#         return float(st[0][1:])
#     if len(st)>=4 and st[1] == 'cash' and st[0][0] == '$' and check_st_lst(st[2], lst2)\
#     and check_st_lst(st[3], ['com', 'shs']):
#         return float(st[0][1:])
#     return False
    
# def check_stock_only(st):
#     """
#     2.1 shs(/sh) com
#     .649 shs com, based on a 2- for-1 pre-split basis
#     """
#     lst1 = ['shs', 'sh']
#     st = st.split()
#     if len(st) == 3 and check_st_lst(st[1], lst1) and st[2].startswith('com') and st[0][0]!='$':
#         return float(st[0])
#     if len(st) >= 3 and 'basis' in st and check_st_lst(st[1], lst1) and st[2].startswith('com') and st[0][0]!='$':
#         return float(st[0])
#     return False
    
# def check_stock_only_fixed_dollar(st):
#     """
#     $12 com
#     $10 shs com
#     """
#     #print(st)
#     lst1 = ['shs', 'sh']
#     st = st.split()
#     if len(st) == 2 and st[1] == 'com' and st[0][0]=='$':
#         return float(st[0][1:])
#     if len(st) == 3 and check_st_lst(st[1], lst1) and st[2] == 'com' and st[0][0]=='$':
#         return float(st[0][1:])
#     return False
    
# def check_combination_fixed_dollar(st):
#     """
#     $11 cash plus $11 shs com
#     $11 cash plus $11 com
#     $2.625 com plus $1.125 cash
#     """
#     lst1 = ['shs', 'sh']
#     lst2 = ['and', 'plus']
#     st = st.split()
#     if len(st) == 6 and st[1] == 'cash' and check_st_lst(st[2], lst2)\
#     and check_st_lst(st[4], lst1) and st[5] == 'com':
#         if st[0][0]!= '$' or st[3][0] != '$':
#             return False
#         else:
#             return (float(st[0][1:]), float(st[3][1:]))
#     if len(st) >= 5 and st[1] == 'cash' and check_st_lst(st[2], lst2)\
#     and st[4].startswith('com'):
#         if st[0][0]!= '$' or st[3][0] != '$':
#             return False
#         else:
#             return (float(st[0][1:]), float(st[3][1:]))
#     if len(st)>=5 and st[1]=='com' and st[2] == 'plus' and st[4] == 'cash':
#         if st[0][0]!= '$' or st[3][0] != '$':
#             return False
#         else:
#             return (float(st[3][1:]), float(st[0][1:]))
#     return False
    
# def check_combination(st):
#     """
#     $2 cash plus(/and) .9 shs(/sh) com
#     $8.50 cash and .85 shs
#     $16.17 cash plus .636 com
#     0.2109 shs(/sh) com plus(/and) $9 cash
#     $1.3 cash plus 0.56385 new sh com
#     """
#     lst1 = ['shs', 'sh']
#     lst2 = ['and', 'plus']
#     st = st.split()
#     if len(st) >= 5 and st[1] == 'cash' and check_st_lst(st[2], lst2)\
#     and (check_st_lst(st[4], lst1) or st[4].startswith('com')):
#         if st[0][0]!= '$' or st[3][0] == '$':
#             return False
#         else:
#             return (float(st[0][1:]), float(st[3]))
#     if len(st) >= 7 and st[1] == 'cash' and check_st_lst(st[2], lst2)\
#     and st[4]=='new':
#         if st[0][0]!= '$' or st[3][0] == '$':
#             return False
#         else:
#             return (float(st[0][1:]), float(st[3]))
#     if len(st) >= 6 and check_st_lst(st[1], lst1) and st[2] == 'com' and check_st_lst(st[3], lst2)\
#     and st[5] == 'cash':
#         if st[0][0] == '$' or st[4][0] != '$':
#             return False
#         else:
#             return (float(st[4][1:]), float(st[0]))
#     return False


# def extract_consid_c(series):
#     # extract sentence
#     #series = df_c['consid']
#     ret = []
#     for consids in series:
#         flag = False
#         consids = consids.replace('\n', ' ').split(';')
#         for consid in consids:
#             temp = find_subst_lst(consid, str_lst)
#             if temp >= 0:
#                 if 'choice' in consid or 'Choice' in consid:
#                     break
#                 consid = consid[:temp]
#                 consid = consid.replace('USD ', '$')
#                 consid = consid.replace('An estimated ', '')
#                 consid = consid.replace('Class A ', '')
#                 consid = consid.replace('ord', 'com')
#                 consid = consid.replace('in ', '')
#                 consid = consid.replace('UAI ', '')
#                 consid_lst = consid.split()
#                 if len(consid_lst) >= 2 and consid.split()[1]=='cash' and consid.split()[0][0]!='$':
#                     consid = '$' + consid
#                 ret.append(consid)
#                 flag = True
#                 break
#         if not flag:
#             ret.append(np.nan)
#     return ret
#     #df_c['consid_'] = ret
    
# def extract_terms_c(series):   
#     # extract terms
#     #df_cc = df_c[~df_c['consid_'].isnull()].copy()
#     #series = df_cc['consid_']
#     ret_st = []
#     ret_ch = []
#     ret_struc = []
#     for consid in series:
#         #print(consid)
#         if pd.isna(consid):
#             ret_st.append(np.nan)
#             ret_ch.append(np.nan)
#             ret_struc.append(np.nan)
#         elif check_cash_only(consid):
#             ret_st.append(0)
#             ret_ch.append(check_cash_only(consid))
#             ret_struc.append('Cash Only')
#         elif check_stock_only(consid):
#             ret_st.append(check_stock_only(consid))
#             ret_ch.append(0)
#             ret_struc.append('Stock Only')
#         elif check_stock_only_fixed_dollar(consid):
#             ret_st.append(check_stock_only_fixed_dollar(consid))
#             ret_ch.append(0)
#             ret_struc.append('Stock Only fixed dollar')
#         elif check_combination(consid):
#             ret_st.append(check_combination(consid)[1])
#             ret_ch.append(check_combination(consid)[0])
#             ret_struc.append('Combination')
#         elif check_combination_fixed_dollar(consid):
#             ret_st.append(check_combination_fixed_dollar(consid)[1])
#             ret_ch.append(check_combination_fixed_dollar(consid)[0])
#             ret_struc.append('Combination fixed dollar')
#         else:
#             ret_st.append(np.nan)
#             ret_ch.append(np.nan)
#             ret_struc.append(np.nan)
#     return ret_struc, ret_ch, ret_st


# def extract_consid_s(series):    
#     # extract sentence
#     #series = df_s['consid']
#     ret = []
#     for consids in series:
#         flag = False
#         consids = consids.replace('\n', ' ').split(';')
#         for consid in consids:
#             temp = find_subst_lst(consid, str_lst)
#             if temp >= 0:
#                 if 'choice' in consid or 'Choice' in consid:
#                     break
#                 consid = consid[:temp]
#                 consid = consid.replace('Cl A ', '')
#                 consid = consid.replace('USD ', '$')
#                 consid = consid.replace('An estimated ', '')
#                 consid = consid.replace(' A', '')
#                 consid = consid.replace(' B', '')
#                 consid = consid.replace(' Class', '')
#                 consid = consid.replace(' class', '')
#                 if consid.startswith(' plus '):
#                     consid = consid[6:]
#                 #consid = consid.replace(' plus ', '')
#                 consid = consid.replace('newly issued ', '')
#                 consid = consid.replace('newly-issued ', '')
#                 consid = consid.replace('new ', '')
#                 consid = consid.replace('ord', 'com')
#                 consid = consid.replace(' C', '')
#                 ret.append(consid)
#                 flag = True
#                 break
#         if not flag:
#             ret.append(np.nan)
#     return ret
#     #df_s['consid_'] = ret
    
    
# def extract_terms_s(series):
#     # extract terms
#     #df_ss = df_s[~df_s['consid_'].isnull()].copy()
#     #series = df_ss['consid_']
#     ret_st = []
#     ret_ch = []
#     ret_struc = []
#     for consid in series:
# #         print(consid)
#         if pd.isna(consid):
#             ret_st.append(np.nan)
#             ret_ch.append(np.nan)
#             ret_struc.append(np.nan)
#         elif check_stock_only(consid):
#             ret_st.append(check_stock_only(consid))
#             ret_ch.append(0)
#             ret_struc.append('Stock Only')
#         elif check_stock_only_fixed_dollar(consid):
#             ret_st.append(check_stock_only_fixed_dollar(consid))
#             ret_ch.append(0)
#             ret_struc.append('Stock Only fixed dollar')
#         elif check_cash_only(consid):
#             ret_st.append(0)
#             ret_ch.append(check_cash_only(consid))
#             ret_struc.append('Cash Only')
#         elif check_combination(consid):
#             ret_st.append(check_combination(consid)[1])
#             ret_ch.append(check_combination(consid)[0])
#             ret_struc.append('Combination')
#         elif check_combination_fixed_dollar(consid):
#             ret_st.append(check_combination_fixed_dollar(consid)[1])
#             ret_ch.append(check_combination_fixed_dollar(consid)[0])
#             ret_struc.append('Combination fixed dollar')
#         else:
#             ret_st.append(np.nan)
#             ret_ch.append(np.nan)
#             ret_struc.append(np.nan)

#     return ret_struc, ret_ch, ret_st 



# def amend(series):
#     val_lst = ['amend', 'sweet', 'original', 'Previous', 'previous', 'Original']
#     ret = []
#     for consid in series:
#         ret.append(int(check_subst(consid, val_lst)))
#     #df['valamend_new'] = ret
#     return ret


# def extract_terms(df_input):
#     df = df_input.copy()
    
#     ret = pd.DataFrame({'consid_struc':np.nan, 'consid_':np.nan,
#                         'ch':np.nan, 'stk':np.nan}, index = df.index)
    
#     df['valamend_new'] = amend(df['consid'])
#     ret.loc[df['valamend_new']==1, 'consid_struc'] = 'amend'
    
#     ret.loc[df['consid_struc_new']=='Cash Only', 'consid_struc'] = 'Cash Only'
#     index0 = ret[ret['consid_struc'].isna()].index
#     df1 = df.loc[index0]
#     df_s = df1[df1['consid_struc_new']=='Stock Only']
#     df_c = df1[df1['consid_struc_new']=='Cash and Stock Combination']
    
#     ret.loc[df_c.index, 'consid_'] = extract_consid_c(df_c['consid'])
#     ret.loc[df_c.index, 'consid_struc'], ret.loc[df_c.index, 'ch'], ret.loc[df_c.index, 'stk'] = \
#     extract_terms_c(ret.loc[df_c.index, 'consid_'])
#     ret.loc[df_s.index, 'consid_'] = extract_consid_s(df_s['consid'])
#     ret.loc[df_s.index, 'consid_struc'], ret.loc[df_s.index, 'ch'], ret.loc[df_s.index, 'stk'] = \
#     extract_terms_s(ret.loc[df_s.index, 'consid_'])
#     return ret#, df_s, df_c


def get_price(permno_list, date_list, db):
    pr_list = []
    for i in tqdm(permno_list.index):
        permno, date = permno_list.loc[i], date_list.loc[i]
        pr_df = db.raw_sql("select prc from crsp_m_stock.dsf where permno = "\
                           + str(permno) + " and date = '" + str(date) + "'")
        if len(pr_df)==1:
            pr_list.append(pr_df.iloc[0, 0])
        else:
            pr_list.append(np.nan)
    return pr_list




def get_num_of_deals_each_day(df, start_date=datetime.date(1990, 1, 1), end_date=datetime.date(2021, 12, 31)):
    """
    compute the number of ongoing deals in each day
    """
    days = get_trading_days_range(start_date, end_date)
    num_of_deals = pd.Series(days).map(lambda x: (df.da_corrected.le(x) & df.dr.gt(x)).sum())
    num_of_deals.index = days
    return num_of_deals


