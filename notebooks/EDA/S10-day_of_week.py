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
    ## Day of Week

    Day of week has a correlation with sales of -0.44. What's going on? In
    Germany, many stores are closed on Sunday.
    """
)

# Prep data for display

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=['M', 'T', 'W', 'R', 'F', 'Sa', 'Su'],
       height=df.groupby('day_of_week').open.mean().values)
ax.set_title('Percent of Stores Open by Day of Week')
ax.set_xlabel('Day of week')
ax.set_ylabel('Percent of Stores Open')
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Day of Week

    As a result, sales are highest on Mondays, falling steadily throughout the
    average week before a slight bump on Fridays.
    """
)

# Prep data for display

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=['M', 'T', 'W', 'R', 'F', 'Sa', 'Su'],
       height=df.groupby('day_of_week').sales.mean().values)
ax.set_title('Average Sales by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Daily Sales')
ax.set_yticklabels(['${:3,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)
