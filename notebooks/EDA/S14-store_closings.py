import cauldron as cd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # NOQA
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right

# Import df from Cauldron shared memory
df = cd.shared.df

cd.display.markdown(
    """
    ## Store Closings

    From July 1, 2014, to December 31, 2014, all 180 Rossmann stores in Bavaria
    closed, presumably for remodeling. Below we see the sum of sales for all
    the Bavaria stores, and then the sales for all other Rossmann stores.
    """
)

# Prep data for display
BY_stores = df.loc[df.state == 'BY', 'store'].unique()
di = df.copy()
di['date'] = pd.to_datetime(di.date)
di['week_start'] = pd.to_datetime(di.week_start)
di.set_index('date', inplace=True)

# Create and display the chart
fig, ax = plt.subplots()
di[di.store.isin(BY_stores)].groupby('date').sales.sum().plot()
ax.set_title('Total Sales in Bavaria')
ax.set_ylabel('Total Daily Sales')
ax.set_xlabel('Date')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)

# Prep data for display
non_stores = di.loc[~di.store.isin(BY_stores)]

# Create and display the chart
fig, ax = plt.subplots()
non_stores.groupby(non_stores.index).sales.sum().plot()
ax.set_title('Total Rossmann Sales Outside of Bavaria')
ax.set_ylabel('Total Daily Sales')
ax.set_xlabel('Date')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)
