import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.ion() # NOQA, need this line for plotting
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd
from typing import Dict, List

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right


df = cd.shared.df

cd.display.markdown(
    """
    ## Overall Sales

    Just as a baseline, here's the overall sales picture.
    """
)

fig, ax = plt.subplots()

ax.plot(df.groupby('date').sales.sum())
ax.set_title('Total sales by date')
ax.set_ylabel('Sales, all stores combined')
ax.set_xlabel('Date')
cd.display.pyplot(fig)
