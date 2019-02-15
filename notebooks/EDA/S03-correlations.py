import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.ion() # NOQA, need this line for plotting
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right


df = cd.shared.df

cd.display.markdown(
    """
    ## Correlation - other features vs sales

    Let's get a quick handle on which variables are most likely to have a
    relationship with sales.  We'll look at correlation.

    The Pearson correlation measures whether two variables have a linear
    relationship:\n
    - A correlation of 1 means a perfect positive linear relationship.
    - A correlation of -1 means a perfect negative linear relationship.
    - A correlation of 0 means no linear relationship at all.
    - While two variables may have a non-linear relationship, the Pearson
    correlation is a common 'eyeball' statistic to help us get our bearings.

    Below are the correlation coefficients with each feature against sales, in
    descending (absolute value) order.\n
    - Somewhat obviously, **sales**, **customers**, and **open** have the
      strongest linear relationships with sales.
    - More interestingly, **promo** has a correlation with sales of 0.50.
    - **day_of_week** has a correlation with sales of 0.44 (the absolute
    value in this case is more interesting than the negative sign, as we'll
    see).\n

    No other feature has a higher linear correlation with sales than 0.15:\n
    - The Google **trend**, **promo2**, and **school_holiday** are all around
      0.10-0.13.
    - The most strongly-correlated weather variable, **max_visibility_km**, has
      a correlation with sales of 0.08. (The Google trend and weather features
      are all recorded at the state level, not at a granular store level.)\n
    """
)

calc = df.corr()['sales']
indices = abs(calc).sort_values(ascending=False).index
result = calc[[col for col in indices]]
res_df = pd.DataFrame({'Feature': result.index,
                       'Correlation With Sales': result.values})
res_df['Correlation With Sales'] = res_df['Correlation With Sales'].round(4)
cd.display.table(res_df)
