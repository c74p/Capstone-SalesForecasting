import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.pyplot as plt
plt.ion() # NOQA, need this line for plotting
import seaborn as sns; sns.set()
import pandas as pd
from typing import Dict, List

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right

cd.display.markdown(
    """
    # Merge dataframes
    In order to merge the dataframes, we did the following:\n
    - Cleaned each dataframe individually
        - For train.csv, store_states.csv, and state_names.csv, it was just
        making column names consistent\n
        - For googletrend.csv, it was fixing 'file' to be legitimate state
        names and changing the 'week' format into actual dates\n
        - For store.csv, it was replacing NaNs with the mean of the column\n
        - For weather.csv, there were a few mistyped column names, and some
        NaNs that had to be replaced\n\n
    We end up with a dataframe with 1,050,330 rows: there are 942 stores, and
    there are 942 days from 2013-01-01 to 2015-07-31, so we have 942 * 1115 =
    1,050,330 rows.\n
    Our table has 43 columns.
    """
    )
try:
    # If the wrangled dataframe is in the folder, use it - that's faster than
    # recreating the dataframe
    df = pd.read_csv('../../data/processed/wrangled_dataframe.csv', header=0,
                     low_memory=False)
except:
    # If not, create the dataframe
    # get file names
    # Note that I would love to not hard-code this, but Cauldron does not have
    # a cd.shared.all() or similar functionality
    files_pulled: List[str] = ['googletrend.csv', 'state_names.csv', 'store.csv',
                               'store_states.csv', 'train.csv', 'weather.csv']

    dfs_dict: Dict[str, pd.DataFrame] = cd.shared.dfs_dict

    df = make_dataset.merge_dfs(dfs_dict)

cd.display.table(df.head())
