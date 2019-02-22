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
    ## School Holiday

    Sales on school holidays are slightly stronger than sales on an average
    non-holiday.  **24%** of days in our dataset are school holidays.
    """
)

# Prep data for display
open = df[df.open == 1].copy()
avg_sch_sales = open.loc[open.school_holiday == 1, 'sales'].mean()
avg_non_sch_sales = open.loc[open.school_holiday == 0, 'sales'].mean()

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=['School Holiday', 'Non-School Holiday'],
       height=[avg_sch_sales, avg_non_sch_sales], color=['blue', 'green'])
ax.set_title('Average Sales, School Holiday vs Non-Holiday')
ax.set_ylabel('Avg Daily Sales')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
ax.set_xlabel('School Holiday Status')
cd.display.pyplot(fig)
