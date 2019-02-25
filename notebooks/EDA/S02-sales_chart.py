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
    2. There was an enormous dip in 2014.  As it turns out, a ton of stores
    were temporarily closed in 2014 (including all 180 stores in Bavaria, for a
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

cd.display.markdown(
    """
    ## Sales by State

    Here we see total sales by state, over the entire period of the dataset.
    Clearly North Rhine-Westfalia leads in both revenue (green bars) and store
    count (blue dots), with Bavaria (BY), Schleswig-Holstein (SH), Berlin (BE),
    and Hesse (HE) being major players.
    """
)

# Prep data for display
open = df[df.open == 1].copy()
sales_by_state = open.groupby('state').sales.sum()
stores_by_state = df.groupby('state').store.nunique()

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=sales_by_state.index, height=sales_by_state, color='green')
ax_r.plot(stores_by_state, color='blue', linestyle='none', marker='o',
          markersize=10, markeredgewidth=5)
ax_r.set_ylim([0, 650])
ax_r.set_title('Total Sales and Store Count by State')
ax_l.set_ylabel('Total Sales, $ Million (green bars)', color='green')
ax_r.set_ylabel('Stores (blue dots)', color='blue')
ax_l.set_yticklabels(['${:,.0f}'.format(x/1e6) for x in ax_l.get_yticks()])
ax_l.set_xlabel('State')
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Average Daily Sales by State

    In average daily sales per store by state, it's a slightly different story.
    Berlin (BE) is the clear leader, with Hamburg (HH), Lower Saxony/Bremen
    (HB,NI), and North Rhine-Westfalia leading in average daily sales per
    store.
    """
)

# Prep data for display
open = df[df.open == 1]
avg_daily_sales_by_state = open.groupby('state').sales.mean()
stores_by_state = df.groupby('state').store.nunique()

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=avg_daily_sales_by_state.index, height=avg_daily_sales_by_state,
         color='green')
ax_l.set_ylim([0, 10000])
ax_r.plot(stores_by_state, color='blue', linestyle='none', marker='o',
          markersize=10, markeredgewidth=5)
ax_r.set_ylim([0, 300])
ax_l.set_title('Average Daily Sales and Stores by State')
ax_l.set_ylabel('Average Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_ylabel('Stores (blue dots)', color='blue')
ax_l.set_xlabel('State')
ax_l.axhline(open.sales.mean())
cd.display.pyplot(fig)
