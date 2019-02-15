import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.pyplot as plt
# plt.ion() # NOQA, need this line for plotting
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right


df = cd.shared.df

cd.display.markdown(
    """
    ## Sales vs Customers

    As noted above, **customers** has a correlation of 0.90 with **sales**.
    It's pretty obvious on the chart below; the more customers, the more sales.
    Note also that as we bring in more customers, the relationship gets less
    strong, until it starts to break down around 5,000 customers in a given
    store (clearly only a few stores could even fit 5,000 customers in a day).

    We don't know the specific definition of 'customer' in this case, or how
    they're counted.  Is it someone who bought, or just someone who came into
    the store? In either case, we'll want to work with the marketing team to
    bring more people through the doors.

    For now, since the correlation with sales is so strong, and because the
    only levers we have to pull -- the only things in the data that we can
    control -- are **promo** and **promo2**, let's continue to focus on
    **sales** and keep **customers** as a secondary focus.
    """
)

avg_sales_by_customers = df.groupby('customers').sales.mean()

fig, ax = plt.subplots()
ax.plot(avg_sales_by_customers)
ax.set_title('Average Sales by Number of Customers')
ax.set_xlabel('Number of Customers')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)
