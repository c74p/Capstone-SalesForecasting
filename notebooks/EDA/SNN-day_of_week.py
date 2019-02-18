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

    Day of week has a correlation with sales of -0.44. What's going on?
    """
)

# Prep data for display
# Note that The first promo was on 2013-01-07, so a rolling 5-day window with
# nans filled with zero is all correct
# Calculate the arrays for the chart
# heights = [0.82869679, 0.40779873, 0.25214708, 0.14337502, -0.05382353]

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(df.groupby('day_of_week').sales.sum())
ax.set_title('Total Sales by Day of Week')
ax.set_xlabel('Day of week')
ax.set_ylabel('Total sales ($Million)')
ax.set_yticklabels(['{:3.0f}%'.format(x/1e6) for x in ax.get_yticks()])
cd.display.pyplot(fig)
