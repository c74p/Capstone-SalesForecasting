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
    ## Weather

    Not surprisingly, the weather has a correlation with sales. The most
    strongly-correlated variable with sales is max_visibility_km:
    """
)

# Prep data for display
open = df[df.open == 1]

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(open.groupby('max_visibility_km').sales.mean())
ax.set_title('Average Sales By Maximum Visibility (km)')
ax.set_ylabel('Average Daily Sales')
ax.set_xlabel('Maximum Visibility (km)')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Temperature

    Many other weather-related variables correlate to sales.  Here we see the
    mean temperature. Sales are elevated when the temperature is comfortable,
    and generally lower when it's too cold or too hot.
    """
)

# Prep data for display
open = df[df.open == 1]

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(open.groupby('mean_temperature_c').sales.mean())
ax.set_title('Average Sales By Mean Temperature')
ax.set_ylabel('Average Daily Sales')
ax.set_xlabel('Mean Temperature (C)')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Weather Events

    Despite the intricacy of the chart below, it basically says two things: it
    rains a lot in Germany, and weather events don't particulary correlate with
    overall sales.

    By far the most common weather events ("None" and "Rain") have average
    daily sales that are right in line with the overall average. The other
    weather events happen too infrequently to matter in the big picture.
    """
)

# Prep data for display
open = df[df.open == 1]
avg_daily_sales_by_events = open.groupby('events').sales.mean()
xs = avg_daily_sales_by_events.index
store_days_by_events = open.groupby('events').date.count()

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=xs, height=avg_daily_sales_by_events, color='green')
# ax_l.set_ylim([0, 10000])
ax_r.plot(xs, store_days_by_events, color='blue', linestyle='none', marker='o',
          markersize=10, markeredgewidth=5)
ax_l.set_title('Average Daily Sales and Store Days by Event')
ax_l.set_ylabel('Average Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_yticklabels(['{:,.0f}'.format(x) for x in ax_r.get_yticks()])
ax_r.set_ylabel('Store Days (blue dots)', color='blue')
ax_l.set_xlabel('Store Type')
fig.autofmt_xdate(rotation=45, ha='right')
ax_l.axhline(open.sales.mean())
cd.display.pyplot(fig)
