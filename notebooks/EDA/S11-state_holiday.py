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
    ## State Holiday

    Sales on state holidays are substantially stronger than sales on an average
    non-holiday.  However, only 25 days in our dataset are public holidays.
    """
)

# Prep data for display
open = df[df.open == 1].copy()
avg_daily_sales_by_state_holiday = open.groupby('state_holiday').sales.mean()
date_counts_by_state_holiday = open.groupby('state_holiday').date.nunique()
hols = ['None', 'Public', 'Easter', 'Christmas']

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=hols, height=avg_daily_sales_by_state_holiday, color='green')
ax_r.plot(hols, date_counts_by_state_holiday, color='blue', linestyle='none',
          marker='o', markersize=10, markeredgewidth=5)
ax_l.set_title('Average Daily Sales and Number of Days by State Holiday')
ax_l.set_ylabel('Average Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_yticklabels(['{:,.0f}'.format(x) for x in ax_r.get_yticks()])
ax_r.set_ylabel('Number of Days (blue dots)', color='blue')
ax_l.set_xlabel('State Holiday')
ax_l.axhline(open.sales.mean())
cd.display.pyplot(fig)
