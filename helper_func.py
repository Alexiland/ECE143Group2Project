import pandas as pd
import numpy as np


def load_dataframe(path):
    '''
    Returns a dataframe read from the given path
    @param path: path of the csv file
    @type path: str
    '''
    assert isinstance(path, str)
    df = pd.read_csv(path)
    return df


def drop_columns(df, columns):
    '''
    Returns the dataframe after removing the given columns

    @param df: Dataframe
    @param columns: list of column names to be dropped
    @type df: pd.DataFrame
    @type columns: List

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, list)

    return df.drop(columns, axis=1)


def rename_columns(df, columns):
    '''
    Returns the dataframe after renaming the given columns

    @param df: Dataframe
    @param columns: dictionary with keys as old column names and values as the new column names
    @type df: pd.DataFrame
    @type columns: Dict

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, dict)

    return df.rename(columns, axis=1)


def convert_str_float(df, col_name):
    '''
    Returns the dataframe with the input column type-casted to float
    Note: Since the data had the char ',' in the numbers, replace function is used to remove the special character

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @type df: pd.DataFrame
    @type col_name: string
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)

    df[col_name] = df[col_name].apply(lambda x: str(x).replace(',', ''))
    df[col_name] = df[col_name].astype(np.float64)
    return df[col_name]


def split_date(df, col_name, num=2):
    '''
    Returns the dataframe with the date column split into year, month and date column based on num

    Note: if num=1, then only year col is added. If num=2, year and month columns are added. If num=3, year,month and day columns are added
          Column names = 'Year', 'Month' (in 3-char format) and 'Day'

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @type df: pd.DataFrame
    @type col_name: string

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)
    assert num in [1, 2, 3]

    df[col_name] = pd.to_datetime(df[col_name], infer_datetime_format=True)
    if num == 1:
        df["Year"] = df[col_name].apply(lambda x: x.year)
    elif num == 2:
        df["Year"] = df[col_name].apply(lambda x: x.year)
        df["Month"] = df[col_name].apply(lambda x: x.strftime("%b"))
    elif num == 3:
        df["Year"] = df[col_name].apply(lambda x: x.year)
        df["Month"] = df[col_name].apply(lambda x: x.strftime("%b"))
        df["Day"] = df[col_name].apply(lambda x: x.strftime("%d"))
    df = df.drop([col_name], axis=1)
    return df


def location_split(x, delim):
    '''
    Returns a list with elements delimited by the given delim

    @param x: Input string
    @param delim: delimiter
    @type x: string
    @type delim: string

    '''

    assert isinstance(x, str)
    assert isinstance(delim, str)

    t = x.split(delim)
    if (len(t) == 3):
        t.insert(2, "")
    elif (len(t) == 2):
        t.insert(0, "")
        t.insert(2, "")
    elif (len(t) == 1):
        t.insert(0, "")
        t.insert(1, "")
        t.insert(2, "")
    t = [x.strip() for x in t]
    return t


def fill_empty_with_NaN(df, col_name, old_value):
    '''
    Returns the dataframe with the date column split into year, month and date column based on num

    Note: if num=1, then only year col is added. If num=2, year and month columns are added. If num=3, year,month and day columns are added
          Column names = 'Year', 'Month' (in 3-char format) and 'Day'

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @param old_value: value to be replaced
    @type df: pd.DataFrame
    @type col_name: string
    @type old_value: string
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)
    assert isinstance(old_value, str)

    df[col_name] = df[col_name].replace(old_value, np.nan)
    return df[col_name]