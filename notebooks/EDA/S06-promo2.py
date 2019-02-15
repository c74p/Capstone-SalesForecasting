import cauldron as cd
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right


df = cd.shared.df

cd.display.markdown(
    """
    ## Customers vs Promo

    So let's look at **promo** vs **sales**; there's a correlation of 0.50
    between them, and **promo** is one of the few things that are under the
    business's control in this dataset.

    Overall, sales at a store with a promotion are about $2,000 (33%) higher
    than sales at a store without a promotion.
    """
)

open = df[df.open == 1]

avg_promo_sales = open.loc[open.promo == 1, 'sales'].mean()
avg_non_promo_sales = open.loc[open.promo == 0, 'sales'].mean()

fig, ax = plt.subplots()

ax.bar(x=['Non-promo', 'Promo'],
       height=[avg_non_promo_sales, avg_promo_sales], color=['blue', 'green'])
ax.set_title('Average Promo vs Non-Promo Sales')
ax.set_ylabel('Avg Daily Sales')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
ax.set_xlabel('Promotion Status')

cd.display.pyplot(fig)


cd.display.markdown(
    """
    ## Customers vs Promo by State
    This incremental $2,000 for a promo appears roughly to hold
    up by state. Below we're comparing non-promoted sales (in green) to
    promoted sales (in blue); the colored horizontal lines indicate overall
    averages. As we can see, promotions are very effective in each state, with
    some slight differences.

    In particular, the three states at the right of the chart below (SN, ST,
    and TH) have below-average daily non-promoted sales (in green); their
    promoted sales (in blue) are **more than $2,000 above** their average
    non-promoted sales.

    On the other hand, two of our top-performing non-promoted states ('HB,NI'
    and HH) have average promoted sales that are **not quite** $2,000 more than
    their non-promoted sales.

    Our best-performing state (BE for Berlin, the capital of Germany) does best
    in both non-promoted and promoted sales. This state does so well that it
    even **out-performs the $2,000 average uplift** for a promotion.
    """
)

avg_daily_promo_sales_by_state = \
    open[open.promo == 1].groupby('state').sales.mean()
avg_daily_non_promo_sales_by_state = \
    open[open.promo == 0].groupby('state').sales.mean()

fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()

ax_l.bar(x=avg_daily_non_promo_sales_by_state.index,
         height=avg_daily_non_promo_sales_by_state, color='green')
ax_l.set_ylim([0, 10000])
ax_r.bar(x=avg_daily_promo_sales_by_state.index,
         height=avg_daily_promo_sales_by_state, color='blue',
         alpha=0.2)
ax_r.set_ylim([0, 10000])

ax_l.set_title('Promo vs Non-Promo Sales by State')
ax_l.set_ylabel('Avg Daily Non-Promo Sales (green)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_yticklabels(['${:,.0f}'.format(x) for x in ax_r.get_yticks()])
ax_r.set_ylabel('Avg Daily Promo Sales (blue)', color='blue', alpha=0.4)
ax_l.set_xlabel('State')
ax_r.axhline(avg_daily_non_promo_sales_by_state.mean(),
             color='green', linestyle='-')
ax_l.axhline(avg_daily_promo_sales_by_state.mean(),
             color='blue', linestyle='-')
cd.display.pyplot(fig)
