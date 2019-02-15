import cauldron as cd
import matplotlib
# matplotlib.use('TkAgg') # NOQA, need this line for plotting
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
    ## Overall Sales

    Just as a baseline, here's the overall sales picture.
    """
)

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(df.groupby('date').sales.sum())
ax.set_title('Total sales by date')
ax.set_yticklabels(['${:,.0f}'.format(x/1e6) for x in ax.get_yticks()])
ax.set_ylabel('Sales, all stores combined ($Million)')
ax.set_xlabel('Date')
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Aggregated by Month

    The previous chart is a little too granular.  Are sales going up as we'd
    expect?

    Below is the same chart, aggregated by month. A couple of interesting
    things jump out:\n
    1. There was a huge year-end spike in December 2013 that we haven't seen
    replicated yet.
    2. There was a huge dip in 2014.  As it turns out, a ton of stores were
    temporarily closed in 2014 (including all 180 stores in Bavaria, for a
    period of months). We'll get back to that later.
    3. Slightly less obvious, but just as important: 2015 sales are back on
    track, with each month's sales being above 2013 sales.  (The overall 2014
    sales aren't a good point of comparison, due to item #2.)
    """
)

# Prep data for resampling
di = df.copy()
di.set_index('date', inplace=True)

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(di.sales.resample('M', label='left').sum())
ax.set_title('Total sales by month')
ax.set_yticklabels(['${:,.0f}'.format(x/1e6) for x in ax.get_yticks()])
ax.set_ylabel('Sales, all stores combined ($Million)')
ax.set_xlabel('Month')

cd.display.pyplot(fig)
