"""     head imports
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
import wrds

pd.options.mode.chained_assignment = None
"""


"""
# pandas Timestamp to datetime.date
x.date()

# pandas Timestamp series to datetime.date series
x.dt.date

# DatetimeIndex to datetime.date np array
x.date
"""

from os.path import expanduser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import date, timedelta
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


def convert_date_str_ser_to_datetime(ser):
    """
    convert a series of date-like strings to datetime.date objects.
    
    Parameters:
    ---------------------
    ser: Series
        a series of date-like strings.
        
    Returns:
    ---------------------
    a series of datetime.date.
    """
    return pd.to_datetime(ser).dt.date





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

def get_trading_day_offset(ser, offset):
    """
    compute the date of trading days with certain days of offset, for a series or a single date. 
    e.g. offset=0 is the next trading day of <ser>, including the day itself. 
    input: ser = datetime.date(2022, 9, 4) or datetime.date(2022, 9, 5) or datetime.date(2022, 9, 6), offset = 0
    output: datetime.date(2022, 9, 6).
    
    Parameters:
    -----------------------------------------
    ser: Series of `datetime.date`; or a single `datetime.date` or string `YY-mm-dd`
    
    Returns:
    -----------------------------------------
    a series of or a single `datetime.date`
    """
    if isinstance(ser, pd.Series):  # series of datetime.date
        if len(ser) == 0:
            return pd.Series([], dtype=object)
        start_date = ser.min()
        end_date = ser.max()
    else:                           # single date
        ser = pd.Timestamp(ser).date()
        start_date = ser
        end_date = start_date
        
    start_date -= timedelta(2*abs(offset) + 5)
    end_date += timedelta(2*abs(offset) + 5)
#     return start_date, end_date
    
    # get `NYSE` calendar
    nyse = mcal.get_calendar('NYSE')
    # returns a pd.Series of datetime.date objects
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date).date    
    
    ind_arr = np.searchsorted(trading_days, ser)
    
    if isinstance(ser, pd.Series):
        return pd.Series(trading_days[ind_arr + offset], index=ser.index)
    return trading_days[ind_arr + offset]


def trading_days_between(ser_start, ser_end):
    """
    calculate the number of trading days between two series or two single dates.
    na can be accepted.
    """
    ser_start_copy = pd.to_datetime(ser_start) if isinstance(ser_start, pd.Series) else ser_start
    ser_end_copy = pd.to_datetime(ser_end) if isinstance(ser_start, pd.Series) else ser_end
    
    start_date = ser_start_copy.min() if isinstance(ser_start, pd.Series) else ser_start_copy
    end_date = ser_end_copy.max() if isinstance(ser_end, pd.Series) else ser_end_copy
    
    start_date -= timedelta(5)
    end_date += timedelta(5)
    
    # get `NYSE` calendar
    nyse_calender = mcal.get_calendar('NYSE')
    
    # returns a pd.Series of datetime.date objects
    nyse_calendar_dates = nyse_calender.valid_days(start_date=start_date, end_date=end_date).tz_localize(None)   
    
    #
    ind_arr_start = np.searchsorted(nyse_calendar_dates, ser_start_copy)
    if isinstance(ser_start, pd.Series):
        ind_arr_start = pd.Series(ind_arr_start, index=ser_start.index)
        # inputting nan into `np.searchsorted` outputs the array len
        ind_arr_start[ind_arr_start.eq(len(nyse_calendar_dates))] = np.nan
    #
    ind_arr_end = np.searchsorted(nyse_calendar_dates, ser_end_copy)
    if isinstance(ser_start, pd.Series):
        ind_arr_end = pd.Series(ind_arr_end, index=ser_start.index)
        # inputting nan into `np.searchsorted` outputs the array len
        ind_arr_end[ind_arr_end.eq(len(nyse_calendar_dates))] = np.nan
        
    return ind_arr_end - ind_arr_start


#####################
## CRSP helpers
#####################

# match permno by ticker or cusip
def get_stock_stocknames_by_ticker_or_cusip_CRSP(idd, identifier='ticker', 
                                                 start_date = None, end_date = None,
                                                db=None):
    """
    search in crsp_m_stock.stocknames database by ticker or (6-digit) cusip.
    returns permno, permco, ticker, comnam, namedt, nameenddt, cusip, ncusip of a stock.
    
    can add the filter that the company is listed in the period from start_date to end_date. 
    but be conservative of this feature.
    
    Parameters:
    -------------------------------------
    idd: string
        ticker or cusip
    identifier: string
        'ticker' or 'cusip'
    start_date:
    
    end_date:
    
    db:
    
    Returns:
    -------------------------------------
    DataFrame
    """
    
    command = "select permno, permco, ticker, comnam, namedt, nameenddt, cusip, ncusip " +\
    "from crsp_m_stock.stocknames where "
    
    if identifier == 'ticker':
        command += f"ticker = '{idd}'"
    elif identifier == 'cusip':
        command += f"(substring(cusip, 1, 6) = '{idd}' or substring(ncusip, 1, 6) = '{idd}')"
        
    if start_date is not None:
        command += f" and namedt <= '{str(start_date)}'"
    if end_date is not None:
        command += f" and nameenddt >= '{str(end_date)}'"    
        
    return db.raw_sql(command)


def get_stock_stocknames_by_permno_CRSP(permno, start_date = None, end_date = None, db=None):
    """
    returns permno, permco, ticker, comnam, namedt, nameenddt, cusip, ncusip of a stock.
    search in crsp_m_stock.stocknames database by permno.
    
    can add the filter that the company is listed in the period from start_date to end_date. 
    but be conservative of this feature.
    """
    
    command = "select permno, permco, ticker, comnam, namedt, nameenddt, cusip, ncusip " +\
    f"from crsp_m_stock.stocknames where permno = {permno}"
    
    if start_date is not None:
        command += f" and namedt <= '{str(start_date)}'"
    if end_date is not None:
        command += f" and nameenddt >= '{str(end_date)}'"    
        
    return db.raw_sql(command)



def get_stock_permno_by_ticker_CRSP(ticker,
                                    start_date = None, end_date = None,
                                   db=None,
                                   return_id=False):
    """
    returns the permno of a stock, searched by ticker or cusip.
    """
    if pd.isna(ticker):
        return (None, None, None, None) if return_id else None 
    # get stocknames file
    df_stocknames = get_stock_stocknames_by_ticker_or_cusip_CRSP(ticker, 'ticker', start_date, end_date, db)
    permno = df_stocknames.permno.unique()
    if len(permno) != 1:
        return (None, None, None, None) if return_id else None
    # matched by ticker, just use the first row of search result 
    df_stocknames = df_stocknames.iloc[0]
    return (df_stocknames.permno, df_stocknames.ticker, df_stocknames.cusip, df_stocknames.comnam) if return_id else df_stocknames.permno   



def get_stock_permno_by_ticker_and_cusip_CRSP(ticker, cusip, 
                                              start_date=None, 
                                              end_date=None, 
                                              db=None,
                                              return_id=False):
    """
    Search for the permno in `crsp_m_stock.stocknames`, by ticker and then cusip. 
    Can also specify a start date and end date during which period the stock is listed.
    
    Returns:
    ------------------------------------
    permno if return_id is False, else the tuple of (permno, ticker, cusip, comnam)
    """
    # search by ticker first
    if ~pd.isna(ticker):
        df_stocknames = get_stock_stocknames_by_ticker_or_cusip_CRSP(ticker, 'ticker', 
                                                                     start_date, end_date, db)
        permno = df_stocknames.permno.unique()
        if len(permno) == 1: # match by ticker, only use the first search result
            df_stocknames = df_stocknames.iloc[0]
            return (df_stocknames.permno, df_stocknames.ticker, df_stocknames.cusip, df_stocknames.comnam) \
                                                    if return_id else df_stocknames.permno 
        
    # either ticker is missing or no search result, search by cusip
    if ~pd.isna(cusip):
        df_stocknames = get_stock_stocknames_by_ticker_or_cusip_CRSP(cusip, 'cusip', 
                                                                     start_date, end_date, db)
        permno = df_stocknames.permno.unique()
        if len(permno) == 1: # match by cusip, only use the first search result
            df_stocknames = df_stocknames.iloc[0]
            return (df_stocknames.permno, df_stocknames.ticker, df_stocknames.cusip, df_stocknames.comnam) \
                                                    if return_id else df_stocknames.permno     
        
    # cannot match by either ticker or cusip
    return (None, None, None, None) if return_id else None   

def get_delisting_information(permno, db=None):
    """
    search for delisting information in crsp_m_stock.dsedelist database.
    returns (delist code, last trade date, delisting date, delisting amount, delisting return)
    """
    if pd.isna(permno):
        return (None, None, None, None) 
    command = "select dlstcd, dlstdt, dlpdt, dlamt, dlret from crsp_m_stock.dsedelist " + \
            f"where permno = {permno}"   
    df_delist = db.raw_sql(command)
    if len(df_delist)!= 1:
        return (None, None, None, None)
    return df_delist.values[0]


def get_stock_value_single_date_CRSP(permno, date, col, db=None):
    """
    look for one single column value, e.g. prc, ret, vol for a stock on a single date. returns a float.
    
    """
    if pd.isna(permno) or pd.isna(date):
        return None
    command = f"select {col} from crsp_m_stock.dsf where permno = {permno} and date = '{date}'"
    result = db.raw_sql(command)
    if len(result) != 1:
        return None
    value = result.values[0, 0]
    return round(abs(value), 3) if (col == 'prc' and value is not None) else value


def get_stock_value_date_range_CRSP(permno, start_date, end_date, col, db=None):
    """
    look for one single column value, e.g. prc, ret, vol for a stock on a date range. returns a Series.
    
    """
    if pd.isna(permno) or pd.isna(start_date) or pd.isna(end_date):
        return pd.Series(np.nan)
    command = f"select date, {col} from crsp_m_stock.dsf where permno = {permno} and date >= '{start_date}' and date <= '{end_date}'"
    result = db.raw_sql(command)
    result = result.set_index('date')
#     return result
    if result.shape[1] != 1:
        return None
    value = result.iloc[:, 0]
    return round(abs(value), 3) if (col == 'prc') else value



def get_stock_values_daily_CRSP(permno, start_date='1000-01-01', end_date='3000-01-01', db=None):
    """
    get the time series of price, return, volumes, share outstanding, adjustment factor from `crsp_m_stock.dsf` database.
    
    Parameters:
    -----------------------------
    
    start_date: datetime.date(), or string. default '1000-01-01'
    
    Returns:
    -----------------------------
    """
    #
    start_date = str(start_date)
    end_date = str(end_date)
    #
    command = """
    select date, permno, prc, ret, vol, shrout, cfacshr, cfacpr from crsp_m_stock.dsf where permno = 
    """ + f"{str(permno)} and date >= '{start_date}' and date <= '{end_date}'"
    #
    df_ret = db.raw_sql(command)  #.dropna()
    df_ret.ret = df_ret.ret.fillna(0)
    df_ret.prc = abs(df_ret.prc.fillna(method='ffill'))
    return df_ret.set_index('date')



def apply_func_to_ser(ser, func, *args, return_as_df = False, columns = None, **kwargs):
    """
    apply function to each element in <ser>. <ser> can be a series of a DataFrame.
    If <ser> is a DataFrame, *row is inputted to the function, i.e. the row elements are inputted as positional arguments to the function.
    
    The output can be either a DataFrame or series. If the output of <func> is a single value, the output of the function should be a Series; if the output of <func> is a tuple, this function can return a DataFrame, with <columns> as the column names.
    
    The function has the same functionality as .map(), but you can monitor progress by tqdm in this function.
    
    Parameters:
    --------------------------------
    ser: Series or DataFrame
        If DataFrame, apply function to each row.
    func:
    
    return_as_df: boolean
        return as a DataFrame if True. Column names are from the parameter <columns>.
    columns: default None
        
    args:
        for <func>
    kwargs:
        for <func>
        
    Returns:
    -------------------------------------
    Series or DataFrame
    
    """
    ret_lst = []
    index = ser.index
    if isinstance(ser, pd.DataFrame):
        for i in tqdm(index):
            ret_lst.append(func(*ser.loc[i], *args, **kwargs))
    else:
        for i in tqdm(index):
            ret_lst.append(func(ser.loc[i], *args, **kwargs))

    if return_as_df:
        ret_lst = list(map(list, ret_lst))
        return pd.DataFrame(ret_lst, index=index, columns = columns)
    return pd.Series(ret_lst, index=index)


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
    # cash only, "$2.2 cash"
    if len(term.split()) <= 2 and re.search("^\$.* cash$", term) != None:
        cash = term.removeprefix('$').removesuffix(' cash')
        try:
            return (float(cash), 0, 'Cash')
        except:
            pass
        
    # cash only, '$21'
    if (re.search('\s', term) == None) and (term[0]=='$'):
        cash = term[1:]
        try:
            return (float(cash), 0, 'Cash')
        except:
            pass
        
    # stock only, '2.1 shs com', '$10 shs com'
    if len(term.split()) <= 3 and (re.search('.* shs com$', term) != None or re.search('.* com$', term) != None or re.search('.* shs$', term) != None):
        stock = term.removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        try:
            return (0, float(stock), 'Common Stock') if stock[0]!='$' else (0, float(stock[1:]), 'Common Stock, fixed dollar')
        except:
            pass

    # combination, '$8.5 cash plus .85 shs (or shs com, or com)'
    if (re.search('^\$.* cash plus .* shs$', term) != None) or (re.search('^\$.* cash plus .* shs com$', term) != None) or (re.search('^\$.* cash plus .* com$', term) != None):
        term_new = term.removeprefix('$').removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        cash = term_new.split()[0]
        stock = term_new.split()[-1]
        try:
            return (float(cash), float(stock), 'Cash and Common Stock') if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
        except:
            pass  
        
    # combination, '0.2109 shs com plus $9 cash', '$12.546 com plus $12.054 cash' (fixed dollar)
    if (re.search('.* com plus \$.* cash$', term) != None) or (re.search('.* shs com plus \$.* cash$', term) != None) or (re.search('.* shs plus \$.* cash$', term) != None):
        term_new = term.removesuffix(" cash")
        cash = term_new.split()[-1][1:]
        stock = term_new.split()[0]
        try:
            return (float(cash), float(stock), 'Cash and Common Stock') if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
        except:
            pass   
        
    # combination, fixed dollar, '$15 cash plus com'
    if (re.search('^\$.* cash plus com$', term) != None) or (re.search('^\$.* cash plus sh com$', term) != None) or (re.search('^\$.* cash plus shs com$', term) != None):
        cash = term.split()[0][1:]
        try:
            return (float(cash), 0, 'Cash and Common Stock, fixed dollar')# if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
        except:
            pass 
        
    return (np.nan, np.nan, 'parse failed')






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
# final version
def get_ff_factors_monthly_CRSP(db, start_month='1000-01', end_month='3000-01'):
    """
    <start_month> and <end_month> can be str '2000-01' or monthly period data type.
    """
    # convert month to date
    start_month = str(start_month) + '-01'
    end_month = str(end_month) + '-31'
    #
    command = 'select date, rf, mktrf, smb, hml, umd from ff_all.factors_monthly where ' + \
    f"date >= '{start_month}' and date <= '{end_month}'"
    #
    df_ff = db.raw_sql(command).dropna()
    return to_monthly_period_index(df_ff, 'date')


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
def to_monthly_period_index(df, col_dt):
    """
    convert the column <col_dt>, a date series, into monthly Period dtype, and set it as the index of <df>.
    
    Parameters:
    ---------------------------------
    df: DataFrame

    col_dt: string
        column name of <df>. The column is a date series to be converted to montly Period.
        
    Returns:
    ---------------------------------
    a DataFrame with montly Period as index.
    """
    return (
        df.assign(month = pd.to_datetime(df[col_dt]).dt.to_period('M'))
        .drop(columns = [col_dt])
        .set_index('month')
    )




def estimate_factor_model(db, ret_ser, model='four-factor'):
    """
    estimate alpha and betas for a monthly return series <ret_ser>
    supports model = 'CAPM', 'three-factor', 'four-factor'
    """
    df_ff = get_ff_factors_monthly_CRSP(db, ret_ser.index[0], ret_ser.index[-1])
    y = ret_ser - df_ff['rf']
    if model == 'CAPM':
        X = df_ff[['mktrf']]
    elif model == 'three-factor':
        X = df_ff[['mktrf', 'smb', 'hml']]
    elif model == 'four-factor':
        X = df_ff[['mktrf', 'smb', 'hml', 'umd']]
    res = sm.OLS(y, sm.add_constant(X, has_constant='add'), missing = 'drop').fit()
    print(res.summary())
    return res









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


