import os
import pandas as pd
from src.data import make_dataset
from typing import Dict, Any

# Config filepaths
PROJ_ROOT = os.path.abspath(os.pardir)
directory = os.path.join(PROJ_ROOT, 'data', 'raw')

# Config kwargs for test_import_csvs
kwargs: Dict[str, Any] = {'header': 0, 'low_memory': False}


def test_import_csvs_pulls_all_csvs():
    print(directory)
    dict_of_dataframes = make_dataset.import_csvs(directory, **kwargs)
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            assert filename in dict_of_dataframes

    # TURN THIS INTO A TESTCASE SO WE ONLY PULL THE CSVS ONCE 




    # pass
    # Want import_csvs() to:
    # - Find and import all the csv files in the directory
    # What could go wrong?
    # - How do we know they're imported?
    # - What if there are no csv files in there?
    # - What if there are unused csvs in there? (We don't care)
    # - What if all the right csvs are not in there?


def test_merge_all_csvs():
    pass
    # Want merge_all_csvs() to:
    # - merge all the csvs together into one, appropriately
    # What could go wrong?
    # - Not all the csvs could be there


def test_verify_csv_pull():
    pass
    # Want verify_csv_pull() to:
    # - Check the csv pull and send a message to user
    #   - Either pull was successful, or pull failed, why, and what to do next
    # What could go wrong?
    # - Not all the csvs could be there
