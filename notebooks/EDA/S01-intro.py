import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.ion() # NOQA, need this line for plotting
import seaborn as sns
#sns.set() # NOQA, need this for styling
import pandas as pd
from typing import Dict, List

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right


# Load and show the Rossmann pic at the top
pic = plt.imread('rossmann.png')
fig, ax = plt.subplots()
ax.imshow(pic, extent=[0, 1, 0, 0.5])
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
cd.display.pyplot(fig)

cd.display.markdown(
    """
    # EDA

    Rossmann is a German retailer with 1,115 stores in 12 German states.
    Our task is to forecast sales at Rossmann stores over the course of six
    weeks.

    We'll dig in as we go, but for now, here's a quick overview of what we've
    got:\n
    - **store** and **date**, our index features\n
    - **sales**, our target variable\n
    - **customers**: Number of customers at that store on that date\n
    - **open**: Whether or not that store is open on that date\n
    - **promo**: Whether or not that store had a promo on that date\n
    - **promo2**: Whether or not that store takes part in a 'continuing and
      consecutive' promotion - maybe like a loyalty program. (We don't get much
      detail here, but we'll dig in more later.)\n
    - **trend**: again not much detail here, but this appears to be the Google
      search trend for Rossmann in a particular state for the week. It's
      an integer between 0 and 100 (actual low is 28).\n
    - The next 10 features are what they sound like, and we'll cover them
      later: **school_holiday**, **state_holiday**, **assortment**,
      **store_type**, some promo2 things, some competition features, **state**,
      **day_of_week**, and **week_start**.\n
    - The rest of the features are weather-related. For each day, in each
      state, we have 24 weather-related measurements.\n
    - Since we have 1,115 stores and 942 days (from 2013-01-01 to 2015-07-31),
      we have 1,115 * 942 = 1,050,330 observations (rows of data).
    """
)

# Import df and format datetimes
df = pd.read_csv('../../data/processed/wrangled_dataframe.csv',
                 header=0, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['week_start'] = pd.to_datetime(df['week_start'])
df.replace('promo_interval', 'promo2_interval', inplace=True)
df = df.rename(columns={'promo_interval': 'promo2_interval'})

# Rearrange columns so that it's easier to see what's going on
df = df[['store', 'date', 'sales', 'customers', 'open', 'promo', 'promo2',
         'trend', 'school_holiday', 'state_holiday', 'assortment',
         'store_type', 'promo2_since_year', 'promo2_since_week',
         'promo2_interval', 'competition_distance',
         'competition_open_since_year', 'competition_open_since_month',
         'state', 'day_of_week', 'week_start', 'max_visibility_km',
         'mean_visibility_km', 'min_visibility_km', 'mean_wind_speed_km_h',
         'max_wind_speed_km_h', 'max_gust_speed_km_h',
         'max_sea_level_pressureh_pa', 'mean_sea_level_pressureh_pa',
         'min_sea_level_pressureh_pa', 'max_temperature_c',
         'mean_temperature_c', 'min_temperature_c', 'wind_dir_degrees',
         'precipitationmm', 'cloud_cover', 'dew_point_c', 'min_dew_point_c',
         'mean_dew_point_c', 'events', 'mean_humidity', 'max_humidity',
         'min_humidity']]

# Create and display the chart
cd.display.table(df.head())

# Export df into Cauldron shared memory
cd.shared.df = df
