import cauldron as cd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # NOQA
import numpy as np
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd
from scipy import stats

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right

# Import df from shared Cauldron memory
df = cd.shared.df

cd.display.markdown(
    """
    ## Sales vs Promo2

    So what about **promo2**?  Below we see **promo2** sales vs. non-**promo2**
    sales. Looks like **promo2** sales are *lower* than non-**promo2** sales.
    """
)

# Prep the data for display
open = df[df.open == 1]
avg_promo2_sales = open.loc[open.promo2 == 1, 'sales'].mean()
avg_non_promo2_sales = open.loc[open.promo2 == 0, 'sales'].mean()

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=['Non-promo2', 'Promo2'],
       height=[avg_non_promo2_sales, avg_promo2_sales],
       color=['blue', 'green'])
ax.set_title('Average Promo2 vs Non-Promo2 Sales')
ax.set_ylabel('Avg Daily Sales')
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
ax.set_xlabel('Promotion Status')
cd.display.pyplot(fig)


cd.display.markdown(
    """
    ## Sales vs promo2_interval
    Let's look by promo2_interval to get a little more detail. Again,
    **promo2** is a "continuing and consecutive" promotion for some stores.
    **promo2_interval** indicates which months the **promo2** is refreshed in
    each quarter.

    Again, below we see that non-**promo2** daily sales are higher than
    **promo2** stores, regardless of when the **promo2** is refreshed. We also
    see (the blue dots) that about 550 of the 1,115 stores do not run
    **promo2**.
    """
)

# Prep data for display
open = df[df.open == 1]
avg_daily_sales_by_promo2_int = open.groupby('promo2_interval').sales.mean()
xs = avg_daily_sales_by_promo2_int.index
store_counts_by_promo2_int = df.groupby('promo2_interval').store.count()/942

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()

ax_l.bar(x=xs, height=avg_daily_sales_by_promo2_int, color='green')
ax_l.set_ylim([0, 10000])
ax_r.plot(store_counts_by_promo2_int, color='blue', linestyle='none',
          marker='o', markersize=10, markeredgewidth=5)
ax_r.set_ylim([0, 600])
ax_l.set_title('Average Daily Sales and Stores by promo2_interval')
ax_l.set_ylabel('Average Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_yticklabels(['{:,.0f}'.format(x) for x in ax_r.get_yticks()])
ax_r.set_ylabel('Stores (blue dots)', color='blue')
ax_l.set_xlabel('promo2_interval')
fig.autofmt_xdate(rotation=45, ha='right')
ax_l.axhline(df.sales.mean())
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## promo2 before and after
    Why would we run **promo2** if daily average sales are *lower* where we run
    it?  We have some examples of stores that started running **promo2** after
    January 2013 (the beginning of our data), so we can compare those stores'
    sales before and after they started **promo2**, to the stores that did not
    run **promo2** at all.

    First, there were 32 stores that started **promo2** the week of August 11,
    2013. Below we see that the **promo2** stores have better performance after
    **promo2** started, whereas the non-**promo2** stores did not, over a 15-
    week period. It's a little hard to see visually, so beneath the chart are
    the slopes of the regression line over the time period.
    """
)

# Prep the data for display
promo2 = df[(df.date >= '2013-04-28') & (df.date <= '2013-11-24') &
            (df.promo2 == 1)]
non_promo2 = df[(df.date >= '2013-04-28') & (df.date <= '2013-11-24') &
                (df.promo2 == 0)]
avg_sales_by_promo2 = promo2.groupby('date').sales.mean()
avg_sales_by_non_promo2 = non_promo2.groupby('date').sales.mean()

# Create regressions for the two avg_sales_by_ lines
xs = np.linspace(0, 211, 211)
a0, b0 = np.polyfit(xs, avg_sales_by_promo2, deg=1)
a1, b1 = np.polyfit(xs, avg_sales_by_non_promo2, deg=1)

# Create and display the chart
fig, (ax_l, ax_r) = plt.subplots(1, 2, sharex='col', sharey='row')
ax_l.plot(avg_sales_by_promo2)
ax_l.plot(avg_sales_by_promo2.index, a0*xs + b0)
ax_r.plot(avg_sales_by_non_promo2)
ax_r.plot(avg_sales_by_non_promo2.index, a1*xs + b1)
fig.autofmt_xdate(rotation=45, ha='right')
ax_l.set_title('Average Daily Sales, promo2 Stores\nWhere promo2 Started '
               '2013-08-11')
ax_r.set_title('Average Daily Sales, non-promo2 Stores')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
cd.display.pyplot(fig)

display_string = f'Slope of sales growth for promo2 stores: \t{round(a0, 4)}\n'
display_string += f'Slope of sales growth for non-promo2 stores: \t'
display_string += f'{round(a1, 4)}'

# Prep data for t-test, collecting printable results in display_string
display_string = 'Two-sided T-test for means:\n'
display_string += f'\t{round(a0, 4)}\n'
display_string += f'\t{round(a1, 4)}\n'

# Run t-test
t_val, t_p_val = stats.ttest_ind(avg_sales_by_promo2, avg_sales_by_non_promo2)

# Finalize and display
display_string += f'T-val: {t_val}, p-val: {t_p_val}\n'
cd.display.code_block(display_string)

cd.display.markdown(
    """
    Similarly, there were 29 stores that started **promo2** the week of March
    9, 2014. Below we see that while all these stores saw seasonal sales
    declines over the time period, the **promo2** stores saw a smaller dip in
    sales than the non-**promo2** stores.
    """
)

# Prep the data for display
promo2 = df[(df.date >= '2013-11-24') & (df.date <= '2014-06-22') &
            (df.promo2 == 1)]
non_promo2 = df[(df.date >= '2013-11-24') & (df.date <= '2014-06-22') &
                (df.promo2 == 0)]
avg_sales_by_promo2 = promo2.groupby('date').sales.mean()
avg_sales_by_non_promo2 = non_promo2.groupby('date').sales.mean()

# Create regressions for the two avg_sales_by_ lines
xs = np.linspace(0, 211, 211)
a0, b0 = np.polyfit(xs, avg_sales_by_promo2, deg=1)
a1, b1 = np.polyfit(xs, avg_sales_by_non_promo2, deg=1)

# Create and display the chart
fig, (ax_l, ax_r) = plt.subplots(1, 2, sharex='col', sharey='row')
ax_l.plot(avg_sales_by_promo2)
ax_l.plot(avg_sales_by_promo2.index, a0*xs + b0)
ax_r.plot(avg_sales_by_non_promo2)
ax_r.plot(avg_sales_by_non_promo2.index, a1*xs + b1)
fig.autofmt_xdate(rotation=45, ha='right')
ax_l.set_title('Average Daily Sales, promo2 Stores\nWhere promo2 Started'
               '2014-03-09')
ax_r.set_title('Average Daily Sales, non-promo2 Stores')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
cd.display.pyplot(fig)

display_string = f'Slope of sales growth for promo2 stores: \t{round(a0, 4)}\n'
display_string += f'Slope of sales growth for non-promo2 stores: \t'
display_string += f'{round(a1, 4)}\n\n'

# Prep data for t-test, collecting printable results in display_string
display_string = 'Two-sided T-test for means:\n'
display_string += f'\t{round(a0, 4)}\n'
display_string += f'\t{round(a1, 4)}\n'

# Run t-test
t_val, t_p_val = stats.ttest_ind(avg_sales_by_promo2, avg_sales_by_non_promo2)

# Finalize and display
display_string += f'T-val: {t_val}, p-val: {t_p_val}\n'
cd.display.code_block(display_string)

cd.display.markdown(
    """Our p-values are less than 0.05 in both cases. This supports the
    position that the differences in sales growth between **promo2** and non-
    **promo2** stores is not due to random chance.
    """
)
