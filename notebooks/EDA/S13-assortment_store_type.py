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
    ## Assortment

    Assortment has a correlation with sales of 0.08. The 'Extra' assortment has
    outstanding sales figures, but there are very few of them. The 'Extended'
    assortment has much better sales than the 'Basic' format, and there are
    over 500 'Extended' stores in the Rossmann Germany business.
    """
)

# Prep data for display
open = df[df.open == 1].copy()
avg_daily_sales_by_assortment = open.groupby('assortment').sales.mean()
store_count_by_assortment = open.groupby('assortment').store.nunique()
assorts = ['Basic', 'Extra', 'Extended']

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=assorts, height=avg_daily_sales_by_assortment, color='green')
ax_l.set_ylim([0, 10000])
ax_r.plot(assorts, store_count_by_assortment.values, color='blue',
          linestyle='none', marker='o', markersize=10, markeredgewidth=5)
ax_r.set_ylim([0, 650])
ax_l.set_title('Average Daily Sales and Stores by Assortment')
ax_l.set_ylabel('Average Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_ylabel('Stores (blue dots)', color='blue')
ax_l.set_xlabel('Assortment')
ax_l.axhline(df.sales.mean())
cd.display.pyplot(fig)

cd.display.markdown(
    """
    ## Store Type

    Store type is not strongly correlated with sales. Store type 'b' has daily
    average sales of about $10,000; but there are a handful of them. The other
    store types have daily average sales figures that are nearly
    indistinguishable.
    """
)

# Prep data for display
open = df[df.open == 1].copy()
avg_daily_sales_by_store_type = open.groupby('store_type').sales.mean()
store_count_by_store_type = open.groupby('store_type').store.nunique()

# Create and display the chart
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.bar(x=avg_daily_sales_by_store_type.index,
         height=avg_daily_sales_by_store_type, color='green')
ax_l.set_ylim([0, 10100])
ax_r.plot(store_count_by_store_type, color='blue', linestyle='none',
          marker='o', markersize=10, markeredgewidth=5)
ax_r.set_ylim([0, 650])
ax_r.set_title('Average Daily Sales and Store Count by Store Type')
ax_l.set_ylabel('Avg Daily Sales (green bars)', color='green')
ax_l.set_yticklabels(['${:,.0f}'.format(x) for x in ax_l.get_yticks()])
ax_r.set_ylabel('Stores (blue dots)', color='blue')
ax_l.set_xlabel('Store Type');
cd.display.pyplot(fig)
