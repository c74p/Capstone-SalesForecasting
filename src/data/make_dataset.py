import os
import pandas as pd
import re
from typing import Any, Dict, List


# def import_csvs(directory: str, **kwargs: str) -> Dict[str, pd.DataFrame]:
def import_csvs(
        directory: str, **kwargs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """ Given a directory, returns a filename-indexed list of DataFrames pulled
    from csv files in the directory.  This implementation allows an optional
    'ignore_files=' kwarg for the caller to skip reading any csv files (can be
    provided as a single string or a list). Other kwargs can be provided, but
    in this implementation, they must all apply to each csv file.

    - directory: the path to the directory of interest.
    - kwargs: kwargs to pass to pd.read_csv (or ignore_files as mentioned
        above).
    - return value: a dictionary where keys are names of the csv files in the
      target directory, and values are the DataFrame version of the csvs.
    """

    dataframes: Dict[str, pd.DataFrame] = {}
    files_to_ignore: List[str] = []

    # If we need to ignore any files, put them in a list and drop the
    # 'ignore_files' flag from kwargs
    if 'ignore_files' in kwargs:
        if isinstance(kwargs['ignore_files'], str):
            files_to_ignore.append(kwargs['ignore_files'])
        else:
            for file in kwargs['ignore_files']:
                files_to_ignore.append(file)
        kwargs.pop('ignore_files')

    # Read csv files into dictionary 'dataframes' keyed by file name
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv') and file_name not in files_to_ignore:
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path, **kwargs)
            dataframes[file_name] = df

    return dataframes


def convert_to_snake_case(string: str) -> str:
    """Helper function to convert column names into snake case. Takes a string
    of any sort and makes conversions to snake case, replacing double-
    underscores with single underscores."""

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    draft = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return draft.replace('__', '_')


def replace_nans(column: pd.Series) -> None:
    """
    Replace NaNs as appropriate in given columns:
        - For columns of Floats, replace with the mean
        - For columns of Objects, replace with 'None' unless the column is
          'date' or 'week'
        - For columns of Ints, replace with the mean coerced to Int, unless the
          column is 'store'
    This is consistent with the data-wrangling; the replaced variables are
    independent variables that aren't crucial.  'store', 'date', and 'week' are
    crucial variables where a nan couldn't be imputed from the data. 'sales'
    is the target variable. Rows with nans for 'store', 'sales', 'date' and
    'week' will be thrown out when the dataframes are merged.

    This function changes the values in place; no value is returned.
    """
    # Fill NaNs for columns of floats with the mean as above
    # 'store' and 'sales' should not be floats anyway; but if there are any
    # NaNs, pandas will coerce the column dtype to float, thus we check
    if column.dtype == 'float64' and column.name not in ['store', 'sales'] and\
            len(column) > 0:
        column.fillna(column.mean(), inplace=True)

    # Fill NaNs for columns of objects with 'None' (except 'date'/'week') as
    # above
    if column.dtype == 'object' and column.name not in ['date', 'week']:
        column.fillna('None', inplace=True)

    # Fill NaNs for columns of ints with mean coerced to int (except 'store'/
    # 'sales') as above
    if column.dtype == 'int64' and column.name not in ['store', 'sales'] and\
            len(column) > 0:
                column.fillna(int(column.mean()), inplace=True)


def merge_csvs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the csvs from import_csvs into a single pd.DataFrame.

    - dfs: a dictionary of dataframes keyed by name. The reference
    implementation has dataframes generated from the following files in
    /data/raw/: 'googletrend.csv', 'state_names.csv', 'store.csv',
    'store_states.csv', 'train.csv', 'weather.csv'
    """

    # Strip '.csv' from key name if present
    # keys = list(dfs.keys())
    # for key in keys:
    # if key.endswith('.csv'):
    # dfs[key[:-4]] = dfs.pop(key)

    # Fix spelling error in weather dataframe
    if 'weather.csv' in dfs.keys() and 'Min_VisibilitykM' in \
            dfs['weather.csv'].columns:
        dfs['weather.csv'].rename(columns={'Min_VisibilitykM':
                                           'Min_VisibilityKm'}, inplace=True)

    # Replace any nans using the replace_nans function
    # Note that as currently written, 'store', 'sales', 'date', or 'week'
    # columns don't get nans replaced (see replace_nans docstring for
    # justification)
    for df in dfs.values():
        col_list = list(df.columns)
        df.columns = pd.Index(map(convert_to_snake_case, col_list))
        for column in df.columns:
            replace_nans(df[column])

    # Create column 'state' in googletrend.csv dataframe with state abbrevs
    if 'googletrend.csv' in dfs.keys() and 'file' in \
            dfs['googletrend.csv'].columns:
        dfs['googletrend.csv']['state'] = dfs['googletrend.csv'].file.str[-2:]

    new_df = {}
    for k, v in dfs.items():
        new_df[k] = v
    return new_df
