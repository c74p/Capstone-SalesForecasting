import cauldron as cd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # NOQA
import numpy as np
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd
from scipy import stats
import statsmodels.formula.api as sm


import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right

# Import df from shared Cauldron memory
df = cd.shared.df

cd.display.markdown(
    """
    ## Sales vs Trend

    Below we see average sales by **trend**. Generally, with increasing
    **trend** we get increasing sales.
    """
)

# Prep the data for display
open = df[df.open == 1]
avg_promo2_sales = open.loc[open.promo2 == 1, 'sales'].mean()
avg_non_promo2_sales = open.loc[open.promo2 == 0, 'sales'].mean()

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
          '90-99', '100'],
       height=open.groupby(open.trend // 10).sales.mean())
ax.set_title('Average Sales by "Trend"')
ax.set_ylabel('Avg Daily Sales')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
ax.set_xlabel('Value of "Trend"')
cd.display.pyplot(fig)
