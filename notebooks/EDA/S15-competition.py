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
    ## Competition

    Although we have data on the competition, the data that we have turns out
    to be incredibly uninteresting.  Below, for example, is average sales by
    competition distance up to 10km.  The competition distance seems to have no
    relationship with the sales. The same is true for the age of the
    competition.

    In fact, the competition has so little to do with the sales that among the
    competition variables, the one that has the strongest correlation with
    sales is which *month* the competition opened.
    """
)

# Prep data for display
open = df[(df.open == 1) & (df.competition_distance <= 10000)]

# Create and display the chart
fig, ax = plt.subplots()
ax.plot(open.groupby('competition_distance').sales.mean())
ax.set_title('Average Sales By Competition Distance')
ax.set_ylabel('Average Daily Sales')
ax.set_xlabel('Competition Distance (m)')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)
