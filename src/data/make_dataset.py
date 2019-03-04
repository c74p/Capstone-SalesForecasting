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


def clean_other_dfs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframes that have no nulls.
    Returns the cleaned dataframe.
    """
    cols = map(convert_to_snake_case, df.columns)
    df.columns = cols
    return df


def clean_googletrend_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean googletrend.csv.
    Returns the cleaned dataframe.
    """
    cols = map(convert_to_snake_case, df.columns)
    df.columns = cols

    # Create a 'state' column, which is just the last 2 of the 'file' column
    # except where the 'file' column ends in 'HB,NI'
    df['state'] = df.file.str[-2:]
    df.loc[df.state == 'NI', 'state'] = 'HB,NI'

    # The googletrend.csv file contains a column 'week', formatted like
    # "2013-01-06 - 2013-01-12", where the first date is the Sunday and the
    # second date is the Saturday of the given week. Every other csv file with
    # a date-like column simply contains a date.  So we want to convert the
    # 'week' column to a 'date' column.
    # In order to do that, we'll create a whole new dataframe, 'week_lookup',
    # containing dates and the start of the week that corresponds to each date.
    # Then we'll merge them together and clean up any extra columns.

    # Get the earliest and latest dates in the dataframe
    date_min = pd.to_datetime(df.week.min()[:10])
    # Note below it's -10: to get the last day of the max week
    date_max = pd.to_datetime(df.week.max()[-10:])

    # Create a new dataframe, week_lookup, listing all days in the
    # period and their corresponding week
    days = pd.date_range(date_min, date_max, freq='D')
    week_lookup = pd.DataFrame({'date': days})

    # To get the start of the week, find the number of the day in the week
    week_lookup['num'] = week_lookup['date'].dt.dayofweek
    # Monday is day 0 in Pandas, so add 1 to get Sundays (and mod 7 to be sure)
    week_lookup['offset'] = (week_lookup['num'] + 1) % 7
    # Sunday of any given week = date - the number of the day in the week
    week_lookup['Week_Start'] = \
        week_lookup['date'] - pd.to_timedelta(week_lookup['offset'], unit='D')
    week_lookup.drop(['num', 'offset'], axis='columns', inplace=True)

    # Create column week_start in df to match up to week_lookup
    df['week_start'] = pd.to_datetime(df.week.str[:10])

    # Re-merge week_lookup back into google so we end up with the
    # appropriate 7 days for each week, and drop auxiliary columns
    df = week_lookup.merge(df, left_on='Week_Start', right_on='week_start')
    df.drop(['file', 'Week_Start', 'week'], axis='columns', inplace=True)
    # Remove extraneous dates from df (these are limited by train.csv)
    df = df[(df.date >= pd.to_datetime('2013-01-01')) &
            (df.date <= pd.to_datetime('2015-07-31'))]

    return df


def clean_store_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean store.csv.
    Returns the cleaned dataframe.
    """
    cols = map(convert_to_snake_case, df.columns)
    df.columns = cols

    for col in ['promo2_since_week', 'promo2_since_year',
                'competition_distance', 'competition_open_since_month',
                'competition_open_since_year']:
        df[col] = df[col].fillna(df[col].mean())

    df['promo_interval'] = df.promo_interval.fillna('None')
    return df


def clean_weather_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather.csv.
    Returns the cleaned dataframe.
    """
    if 'Min_VisibilitykM' in df.columns:
        df.rename(columns={'Min_VisibilitykM': 'Min_VisibilityKm'},
                  inplace=True)
    if 'Min_DewpointC' in df.columns:
        df.rename(columns={'Min_DewpointC': 'MinDew_pointC'}, inplace=True)

    cols = map(convert_to_snake_case, df.columns)
    df.columns = cols

    for col in ['max_visibility_km', 'min_visibility_km', 'mean_visibility_km',
                'max_gust_speed_km_h', 'cloud_cover']:
        df[col] = df[col].fillna(df[col].mean())

    df['events'] = df.events.fillna('No Events')
    return df


def merge_dfs(raw_dfs_dict: Dict[str, pd.DataFrame]) -> \
             (pd.DataFrame, Dict[str, pd.DataFrame]):
    """Merge the csvs from import_csvs into a single pd.DataFrame.

    - Input: a dictionary of dataframes keyed by name.
        - The reference implementation has dataframes generated from the
          following files in /data/raw/: 'googletrend.csv', 'state_names.csv',
          'store.csv', 'store_states.csv', 'train.csv', 'weather.csv'
        - All of these files must exist and have at least one row with no
          NaN values.
    - Output: a combined dataframe and dictionary of dataframes from which it
      was constructed.
    """

    # List out the dfs available, and make copies of all dfs
    raw_dfs_list = raw_dfs_dict.keys()
    dfs_dict = {}
    for df in raw_dfs_list:
        dfs_dict[df] = raw_dfs_dict[df].copy()

    # Run the custom cleaning functions on the dataframes that need them
    dfs_dict['googletrend.csv'] = \
        clean_googletrend_csv(dfs_dict['googletrend.csv'])
    dfs_dict['store.csv'] = \
        clean_store_csv(dfs_dict['store.csv'])
    dfs_dict['weather.csv'] = \
        clean_weather_csv(dfs_dict['weather.csv'])

    # Run generic 'clean_other_dfs' function on the other dataframes
    dfs_dict['state_names.csv'] = \
        clean_other_dfs(dfs_dict['state_names.csv'])
    dfs_dict['store_states.csv'] = \
        clean_other_dfs(dfs_dict['store_states.csv'])
    dfs_dict['train.csv'] = \
        clean_other_dfs(dfs_dict['train.csv'])

    # Start by merging store_states and state_names
    df = dfs_dict['store_states.csv'].merge(dfs_dict['state_names.csv'],
                                            on='state')
    # Add in weather
    df = df.merge(dfs_dict['weather.csv'],
                  left_on='state_name', right_on='file')

    # Drop file and state_name - they are colinear with 'state'
    df.drop(['file', 'state_name'], axis='columns', inplace=True)

    # Add in store
    df = df.merge(dfs_dict['store.csv'], on='store')

    # Add in train - note that since train.csv has some missing dates, where
    # the store was apparently closed, we use 'outer' to capture all the dates
    df = df.merge(dfs_dict['train.csv'], on=['date', 'store'], how='outer')

    # Add in googletrend, making sure to coerce 'date' to datetime first
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(dfs_dict['googletrend.csv'], on=['date', 'state'])

    # final cleanup
    df.loc[df.open.isnull(), 'open'] = 0
    df.loc[df.sales.isnull(), 'sales'] = 0
    df.loc[df.customers.isnull(), 'customers'] = 0
    df.loc[df.promo.isnull(), 'promo'] = 0
    df.loc[df.school_holiday.isnull(), 'school_holiday'] = 0
    df.loc[df.state_holiday.isnull(), 'state_holiday'] = '0'
    df['day_of_week'] = df.date.dt.dayofweek
    df.loc[df.customers == 0, 'open'] = 0
    df['date'] = pd.to_datetime(df['date'])
    df['week_start'] = pd.to_datetime(df['week_start'])
    df.loc[df.open == 0, 'promo'] = 0

    new_dict = {}
    for k, v in dfs_dict.items():
        new_dict[k] = v
    return (df, new_dict)
