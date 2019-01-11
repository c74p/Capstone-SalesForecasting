import os
import pandas as pd
from typing import Any, Dict


# def import_csvs(directory: str, **kwargs: str) -> Dict[str, pd.DataFrame]:
def import_csvs(
        directory: str, **kwargs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """ Given a directory, returns a filename-indexed list of DataFrames pulled
    from csv files in the directory. Note that kwargs can be provided, but in
    this implementation, they must all apply to each csv file.

    - directory: the path to the directory of interest.
    - kwargs: kwargs to pass to pd.read_csv.
    - return value: a dictionary where keys are names of the csv files in the
      target directory, and values are the DataFrame version of the csvs.
    """

    dataframes: Dict[str, pd.DataFrame] = {}

    # Read csv files into dictionary 'dataframes' keyed by file name
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path, **kwargs)
            dataframes[file_name] = df

    return dataframes
