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

# Import df from shared Cauldron memory
df = cd.shared.df

cd.display.markdown(
    """
    ## Correlation - features vs sales

    Let's get a quick handle on which variables correlate most strongly with
    sales.  (See the section below the chart for a brief explanation of
    correlation if you're not familiar with it.)

    A few takeaways from the below:\n
    - **customers** has the strongest correlation with sales, at 0.9.
    **customers** can be viewed as both an input and an output: strong pricing,
    promotions, merchandising, and assortment will not only bring in more
    customers, but will also drive higher revenue per customer.
    - We can see that **promo** has a correlation with sales of 0.50, and
    **promo2** and **promo2_interval** have correlations with sales around
    -0.10.
    - **trend** has a correlation with sales of 0.13. Like **customers**, this
    variable could be viewed (though to a lesser degree) as both an input and
    an output.
    - We don't have great data on pricing, product, or merchandising that help
    to explain our sales numbers. Specifically, we have **assortment** (0.08
    correlation with sales) and **store_type** (-0.02 correlation with sales).
    - While **competition_distance** and **competition_open_since_year** seem
    like they should help us to evaluate the strength of our store locations
    in some fashion, they have correlations of around 0.01.

    The rest of the variables are essentially out of control of the business
    (weather, state holidays, etc.).
    """
)

# Create a dataframe with appropriate categoricals represented as ints, so they
# show up in a correlation matrix
no_cats = df.copy()
no_cats.replace(
    {'assortment': {'a': 0, 'b': 1, 'c': 2},
     'store_type': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
     'promo2_interval': {'None': 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2,
                        'Mar,Jun,Sept,Dec': 3},
     'state_holiday': {'0': 0, 'a': 1, 'b': 2, 'c': 3},
     'state': {'BE': 0, 'BW': 1, 'BY': 2, 'HB,NI': 3, 'HE': 4, 'HH': 5, 'NW':
               6, 'RP': 7, 'SH': 8, 'SN': 9, 'ST': 10, 'TH': 11}},
    inplace=True)

# Convert week_start to integers as well
no_cats['week_start'] = pd.to_datetime(no_cats.week_start)
no_cats['week_start'] = no_cats['week_start'] - pd.to_datetime('2012-12-30')
no_cats['week_start'] = no_cats['week_start'].dt.days

# Take correlations vs 'sales'
calc = no_cats.corr()['sales']

# Sort by absolute values
indices = abs(calc).sort_values(ascending=False).index
result = calc[[col for col in indices]]

# Put it all in a DataFrame so it looks nice
res_df = pd.DataFrame({'Feature': result.index,
                       'Correlation With Sales': result.values})
res_df['Correlation With Sales'] = res_df['Correlation With Sales'].round(4)

# Display the chart
cd.display.table(res_df)

cd.display.markdown(
    """
    The Pearson correlation measures whether two variables have a linear
    relationship:\n
    - A correlation of 1 means a perfect positive linear relationship.
    - A correlation of -1 means a perfect negative linear relationship.
    - A correlation of 0 means no linear relationship at all.
    - While two variables may have a non-linear relationship, the Pearson
    correlation is a common 'eyeball' statistic to help us get our bearings.
    """
)
