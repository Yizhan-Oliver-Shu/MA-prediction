import pandas as pd
import numpy as np
from os.path import expanduser

import locale
from locale import atof, setlocale

setlocale(locale.LC_ALL, 'en_US')


def print_shape(df):
    """
    Print the shape of a dataset.
    """
    print(f'The dataset is of size {df.shape}.')

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
def convert_singe_date_str_to_datetime(date):
    """
    convert a single date-like string to datetime.date object.
    
    Parameters:
    ---------------------
    date:
        one date-like strings.
        
    Returns:
    ---------------------
    datetime.date.
    """
    return pd.to_datetime(date).date()


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



##############################
## correct dataset
##############################

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
    if 1040793020 in df.index:
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

