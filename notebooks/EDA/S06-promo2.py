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
    ## Impact of Multiple Promos

    So if we can reliably get an **extra $2,000 in revenue** per store by
    running a promotion, can we just run more promotions? It depends.

    Below we see the uplift of a promo over the last 5 days' sales, broken down
    by the number of promotions in the previous 5 days.

    If there haven't been any promotions in the last 5 days, we can expect on
    average an 80% uplift; but **each extra promotion cuts our uplift in
    half**.

    At the extreme right, we see that a promo on a 5th consecutive day has
    a negative uplift (against the previous promoted days' sales).

    So, we see diminishing returns as we promote more intensely.
    """
)

# Prep data for display
# Note that The first promo was on 2013-01-07, so a rolling 5-day window with
# nans filled with zero is all correct
roll5 = df.copy()
roll5.set_index(['store', 'date'], inplace=True)
roll5.sort_index(inplace=True)

# Collect the average # of promos in the last 5 days inclusive
roll5['promo_last_5'] = roll5.promo.rolling(5).mean()

# Calculate the uplift over the last 5 days inclusive
roll5['uplift'] = roll5.sales / roll5.sales.rolling(5).mean() - 1
roll5.fillna(0, inplace=True)

# Calculate the arrays for the chart
xs = [0, 1, 2, 3, 4]
heights = \
    roll5[roll5.promo == 1].groupby('promo_last_5').uplift.mean().values
# heights = [0.82869679, 0.40779873, 0.25214708, 0.14337502, -0.05382353]

# Create and display the chart
fig, ax = plt.subplots()
ax.bar(x=xs, height=heights)
ax.set_title('Consecutive Promotions are Less Effective')
ax.set_xlabel('Number of Promotions Previous 5 Days')
ax.set_ylabel('Average Sales Uplift on Promotion')
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
cd.display.pyplot(fig)

cd.display.markdown(
    """ ## Opportunities to Promote More
    But are there opportunities to fit promotions in appropriately, without
    killing our current sales and profitability?

    Maybe.  Below we see that over 30% of store-days have no promotions in the
    previous 5 days.

    So it's possible that we might be able to slot in some more promotions,
    with some finer analysis of the financials of promotions. This would
    require additional detail and additional cost/benefit analysis (data
    that we don't currently have).
    """
)

avg_promo_last_5 = roll5[roll5.open == 1].groupby('promo_last_5').state.count()
avg_promo_last_5 = avg_promo_last_5/len(roll5[roll5.open == 1])

fig, ax = plt.subplots()
ax.bar(x=avg_promo_last_5.index, height=avg_promo_last_5.values, width=0.15)
ax.set_title('Total Promos in a 5-Day Span')
ax.set_xlabel('Stores with # Promos, Rolling 5 Days')
ax.set_ylabel('% of Total Open Days')
ax.set_yticklabels(['{:3.0f}%'.format(y*100) for y in ax.get_yticks()])
ax.set_xticklabels(['{:1.0f}'.format(x*5) for x in ax.get_xticks()])
cd.display.pyplot(fig)

# Export roll5 dataframe into Cauldron shared memory
cd.shared.roll5 = roll5
