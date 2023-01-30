# from os.path import expanduser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from datetime import timedelta
import pandas_market_calendars as mcal



"""
def compute_expected_return_portfolio
def compute_variance_portfolio
def compute_vol_portfolio
def plot_efficient_frontier()
def compute_efficient_frontier_two_risky()
def compute_efficient_frontier_several_risky()
def compute_MSR_portfolio(, rf)
"""
"""
from IPython.display import display, HTML
display(HTML("<style>.container { width:92% !important; }</style>"))

%load_ext autoreload
%autoreload 2
import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path = "~/Documents/G3/ORF_435"
path = expanduser(path)
sys.path.append(path)

import CRSP_helpers as crsp
import pandas as pd
import numpy as np
import datetime
# import wrds

pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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



"""
def get_delisting_information(permno, 
                              cols=['dlstcd', 'dlstdt', 'dlpdt', 'dlamt', 'dlret'],
                              db=None):

def get_stock_stocknames_CRSP(id_no, id_type='permno', date = None, 
                              cols=['permno', 'permco', 'ticker', 'comnam', 'namedt', 'nameenddt', 'cusip', 'ncusip'], 
                              db=None):
                              
def convert_stocknames_to_permno(stocknames, return_names=False):

def get_stock_permno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):


def get_stock_market_data_daily_CRSP(id_no, id_type='permno', start_date='1900-01-01', end_date='2030-01-01', 
                                cols=['permno', 'prc', 'ret', 'vol', 'shrout', 'cfacpr', 'cfacshr'], 
                                db=None):
                                

"""
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



def to_monthly_period_index(df, col_dt, format = None):
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
    # extract date column
    date_col = df[col_dt]
    # drop it 
    df_new = df.drop(columns = [col_dt])
    # change to monthly period
    df_new['month'] = pd.to_datetime(date_col, format = format).dt.to_period('M')
    # set as index
    return df_new.set_index('month')

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
    df_ff_ret = pd.concat([df_ff, ret_ser], axis=1)
    name = df_ff_ret.columns[-1]
    # minus rf rate
    df_ff_ret[name+'rf'] = df_ff_ret[name].sub(df_ff.rf)
    #
    res = sm.OLS(df_ff_ret[name+'rf'], sm.add_constant(df_ff_ret[cols_dict[model]], has_constant='add'), missing = 'drop').fit()
#     print(res.summary())
    return res


   