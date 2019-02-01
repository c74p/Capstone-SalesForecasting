import os
import pandas as pd
import re
from typing import Any, Dict, List

pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception


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
    """ Replace NaNs as appropriate in given columns:
        - For columns of Floats, replace with the mean
        - For columns of Objects, replace with 'None'
        - For columns of Ints, replace with the mean coerced to Int

    This function changes the values in place; no value is returned.
    """
    # Fill NaNs for columns of floats with the mean as above
    # Need to check the column is non-empty or will get an error
    if column.dtype == 'float64' and len(column) > 0:
        column.fillna(column.mean(), inplace=True)

    # Fill NaNs for columns of objects with 'None'
    if column.dtype == 'object':
        column.fillna('None', inplace=True)

    # Fill NaNs for columns of ints with mean coerced to int
    # Need to check the column is non-empty or will get an error
    if column.dtype == 'int64' and len(column) > 0:
        column.fillna(int(column.mean()), inplace=True)


# EDIT CHANGE THE TYPE DEF AND DOCSTRING IF NEEDED !!!!
def wrangle_googletrend_csv(google: pd.DataFrame) -> None:
    """Wrangle the googletrend.csv dataframe:
        - Change the 'file' column to 'state' with appropriate abbreviations.
        - Change the 'week' column to 'date', making it day-based rather than
          week-based for easy merging with the other dataframes.

    This function changes the values in place; no value is returned.
    """

    if 'file' in google.columns:
        google.dropna(axis='index', inplace=True)  # first drop rows w/any null
        if len(google) > 0:  # only includes non-null rows now
            # Create column 'state' in google dataframe with state abbrevs
            # Abbreviations are the last two characters, except for 'HB,NI'
            # import ipdb; ipdb.set_trace()
            google['state'] = google.file.str[-2:]
            google.loc[google.state == 'NI', 'state'] = 'HB,NI'
            # cond = lambda series: series.str.endswith('HB,NI') # NOQA
            # Where cond is true, hard-code 'HB,NI'
            # google['state'] = google.loc[google.file.notnull(),
            # 'file'].mask(cond, 'HB,NI', inplace=True)
            # Where cond is NOT true, take the last two (FYI, odd syntax here)
            # google['state'] = \
            #    google.loc[google.file.notnull(),
            #               'file'].where(cond, google['file'].str[-2:])

            # For each week in dataframe google, add rows for each of the
            # days in the week, so we can later merge against other day-based
            # dataframes

            # Figure out which day each week starts, along with a min and
            # max week-start-date for the dataframe
            google.loc[:, 'week_start'] = \
                pd.to_datetime(google['week'].str[:10])
            start_date = pd.to_datetime(google.week.min()[:10])
            # Note below it's -10: to get the last day of the max week
            end_date = pd.to_datetime(google.week.max()[-10:])

            # create a new dataframe, week_lookup, listing all days in the
            # period and their corresponding week
            days = pd.date_range(start_date, end_date, freq='D')
            week_lookup = pd.DataFrame({'date': days})
            week_lookup['num'] = week_lookup['date'].dt.dayofweek
            week_lookup['offset'] = (week_lookup['num'] + 1) % 7
            week_lookup['Week_Start'] = week_lookup['date'] - \
                pd.to_timedelta(week_lookup['offset'], unit='D')
            week_lookup.drop(['num', 'offset'], axis='columns', inplace=True)

            # Re-merge week_lookup back into google so we end up with the
            # appropriate 7 days for each week
            new_thing = week_lookup.merge(google, left_on='Week_Start',
                                          right_on='week_start')
            new_thing = new_thing[
                    (new_thing.date >= pd.to_datetime('2013-01-01')) &
                    (new_thing.date <= pd.to_datetime('2015-07-31'))]
            # google.drop(['file', 'Week_Start', 'week'], axis='columns',
            #            inplace=True)
            # google = new_thing.copy()

            # EDIT REMOVE THIS LATER IF NEEDED!!!!!
            return new_thing


# EDIT Update the type signature once the function has been changed to only
# return a dataframe
def merge_csvs(dfs_dict: Dict[str, pd.DataFrame]) -> (pd.DataFrame,
                                                      Dict[str, pd.DataFrame]):
    """Merge the csvs from import_csvs into a single pd.DataFrame.

    - dfs_dict: a dictionary of dataframes keyed by name.
        - The reference implementation has dataframes generated from the
          following files in /data/raw/: 'googletrend.csv', 'state_names.csv',
          'store.csv', 'store_states.csv', 'train.csv', 'weather.csv'
        - All of these files must exist and have at least one row with no
          NaN values.
    """

    csv_list = dfs_dict.keys()

    # Fix spelling errors in weather dataframe
    if 'weather.csv' in csv_list:
        if 'Min_VisibilitykM' in dfs_dict['weather.csv'].columns:
            dfs_dict['weather.csv'].rename(
                columns={'Min_VisibilitykM': 'Min_VisibilityKm'}, inplace=True)
        if 'Min_DewpointC' in dfs_dict['weather.csv'].columns:
            dfs_dict['weather.csv'].rename(
                columns={'Min_DewpointC': 'MinDew_pointC'}, inplace=True)

    # In each dataframe, update column names and replace any nans
    # Note that as currently written, 'store', 'sales', 'date', 'week', or
    # 'file' columns don't get nans replaced - these can't just be imputed
    for df in dfs_dict.values():
        col_list = list(df.columns)
        df.columns = pd.Index(map(convert_to_snake_case, col_list))
        for column in df.columns:
            if column not in ['date', 'file', 'sales', 'store', 'week']:
                replace_nans(df[column])
        # Drop any remaining rows with nans
        df.dropna(axis='index', inplace=True)

# date: google, train, weather
# state: google, state_names, store_states, weather
# store: store, store_states, train
    df = dfs_dict['store_states.csv']

    df = df.merge(dfs_dict['store.csv'], on='store')

    # Ensure that both dfs have float values for 'store' columns and merge
    dfs_dict['train.csv']['store'] = \
        dfs_dict['train.csv']['store'].astype('float')
    df['store'] = df['store'].astype('float')
    df = df.merge(dfs_dict['train.csv'], on='store')
    # print(df.date)

    if 'state_names.csv' in csv_list and len(dfs_dict['state_names.csv']) > 0:
        # Ensure that both dfs have str values for 'state' columns and merge
        df['state'] = df['state'].astype('object')
        dfs_dict['state_names.csv']['state'] = \
            dfs_dict['state_names.csv']['state'].astype('object')
        df = df.merge(dfs_dict['state_names.csv'], on='state')

    if 'weather.csv' in csv_list and len(dfs_dict['weather.csv']) > 0:
        # Ensure that both dfs have strings for merging-on columns and merge
        df['state_name'] = df['state_name'].astype('object')
        dfs_dict['weather.csv']['file'] = \
            dfs_dict['weather.csv']['file'].astype('object')
        df = df.merge(dfs_dict['weather.csv'], left_on=['state_name', 'date'],
                      right_on=['file', 'date']).drop('file', axis='columns')

    if 'googletrend.csv' in csv_list and len(dfs_dict['googletrend.csv']) > 0:
        # print('len of google csv', len(dfs_dict['googletrend.csv']))
        dfs_dict['googletrend.csv'] = \
            wrangle_googletrend_csv(dfs_dict['googletrend.csv'])

        # print(type(dfs_dict['googletrend.csv']))

        # if type(dfs_dict['googletrend.csv']) != "<class 'NoneType'>" and \
        if dfs_dict['googletrend.csv'] is not None and \
                dfs_dict['googletrend.csv'].notnull().any().any():
            # Ensure that both dfs have strings for merging columns and merge
            print('cols:', df.columns)
            df['date'] = pd.to_datetime(df['date'])
            dfs_dict['googletrend.csv']['date'] = \
                pd.to_datetime(dfs_dict['googletrend.csv']['date'])
            df = df.merge(dfs_dict['googletrend.csv'], on=['date', 'state'])

    new_dict = {}
    for k, v in dfs_dict.items():
        new_dict[k] = v
    return (df, new_dict)
