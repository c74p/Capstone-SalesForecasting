import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.pyplot as plt
plt.ion() # NOQA, need this line for plotting
# import seaborn as sns
import pandas as pd
from typing import Dict, List

# get file names
# Note that I would love to not hard-code this, but Cauldron does not have
# a cd.shared.all() or similar functionality
files_pulled: List[str] = ['googletrend.csv', 'sample_submission.csv',
                           'state_names.csv', 'store.csv', 'store_states.csv',
                           'train.csv', 'weather.csv']


dfs: Dict[str, pd.DataFrame] = cd.shared.dfs

df: pd.DataFrame = dfs['train.csv']  # no nans
goog: pd.DataFrame = dfs['googletrend.csv']  # no nans
states: pd.DataFrame = dfs['store_states.csv']  # no nans
names: pd.DataFrame = dfs['state_names.csv']  # no nans
stores: pd.DataFrame = dfs['store.csv']  # see below for nans
weather: pd.DataFrame = dfs['weather.csv']  # see below for nans

# Nulls to deal with:
#   - Stores - CompetitionDistance fill with mean
#   - Stores - CompetitionOpenSinceMonth fill with mean
#   - Stores - CompetitionOpenSinceYear fill with mean
#   - Stores - Promo2SinceWeek fill with zero
#   - Stores - Promo2SinceYear fill with zero
#   - Stores - PromoInterval fill with 'None'
#   - Weather - Max_VisibityKm fill with mean
#   - Weather - Min_VisibitykM fill with mean
#   - Weather - Mean_VisibityKm fill with mean
#   - Weather - Max_Gust_SpeedKm_h fill with mean
#   - Weather - CloudCover fill with mean
#   - Weather - Events fill with string 'No events'


# merge appropriately

# save as new dataframe
print(df.head())
print(goog.head())
print(states.head())
print(names.head())
print(stores.head())
print(weather.head())
