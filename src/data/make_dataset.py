import os
import pandas as pd
from typing import Any, Dict


# def import_csvs(directory: str, **kwargs: str) -> Dict[str, pd.DataFrame]:
def import_csvs(
        directory: str, **kwargs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """ Given a directory, returns a filename-indexed list of DataFrames pulled
    from csv files in the directory. Note that kwargs can be provided, but in
    this implementation, they must all apply to each csv file. Also note that
    this implementation provides a 'ignore_files=' kwarg for the caller to skip
    reading any csv files (can be provided as a single string or a list).

    - directory: the path to the directory of interest.
    - kwargs: kwargs to pass to pd.read_csv (or ignore_files as mentioned
        above).
    - return value: a dictionary where keys are names of the csv files in the
      target directory, and values are the DataFrame version of the csvs.
    """

    dataframes: Dict[str, pd.DataFrame] = {}

    # Read csv files into dictionary 'dataframes' keyed by file name
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):

            # Check for file_name in kwarg 'ignore_files'
            import_this_csv = ('ignore_files' not in kwargs) or \
                ('ignore_files' in kwargs and
                    file_name not in kwargs['ignore_files'])

            if import_this_csv:
                file_path = os.path.join(directory, file_name)
                df = pd.read_csv(file_path, **kwargs)
                dataframes[file_name] = df

    return dataframes
