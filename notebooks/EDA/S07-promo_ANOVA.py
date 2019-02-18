import cauldron as cd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # NOQA
import seaborn as sns
sns.set() # NOQA, need this for styling
import pandas as pd
from scipy import stats

import os, sys # NOQA
sys.path.append('../../src/data')
import make_dataset # NOQA, need the lines above to get directories right

# Import df and roll5 from Cauldron shared memory
df = cd.shared.df
roll5 = cd.shared.roll5

cd.display.markdown(
    """
    ## Statistical testing of promo uplifts (technical)

    Just to be sure, let's check that the difference in effectiveness of promos
    we see above is real.  We'll use a one-way ANOVA test with p=0.05; in other
    words, if p < 0.05, we can conclude with 95% confidence that the difference
    in uplifts is not a result of random chance.

    """
)

# Prep data for display, collecting printable results in display_string
arrays = []
display_string = ''
display_string += 'One-way ANOVA test for difference of means:\n'

# Create a list of arrays to check using ANOVA
for i in range(1, 6):
    vals = \
        roll5[(roll5.promo == 1) & (roll5.promo_last_5 == i/5)].uplift.values
    display_string += f'\t{round(vals.mean(), 4)}\n'
    arrays.append(vals)

# Run ANOVA test
f_val, p_val = stats.f_oneway(*arrays)

# Finalize and display
display_string += f'F-val: {f_val}, p-val: {p_val}\n'
cd.display.code_block(display_string)

cd.display.markdown(
    """
    Okay, so we can confidently claim that some of the means are different. But
    we don't know which ones, so let's run a t-test to check whether the two
    closest are *really* different, to 95% confidence.  The two closest are for
    2 and 3 days above, with mean uplifts of 25.21% and 14.34% respectively.

    Again we'll use a p-value of p=0.05, meaning that if p < 0.05, we can
    conclude with 95% confidence that the difference is real.
    """
)

# Prep data for display, collecting printable results in display_string
display_string = 'Two-sided T-test for means:\n'
display_string += f'\t{round(arrays[2].mean(), 4)}\n'
display_string += f'\t{round(arrays[3].mean(), 4)}\n'

# Run t-test
t_val, t_p_val = stats.ttest_ind(arrays[2], arrays[3])

# Finalize and display
display_string += f'T-val: {t_val}, p-val: {t_p_val}\n'
cd.display.code_block(display_string)

cd.display.markdown(
    """
    Pretty convincing.

    While we're here, let's just double-check that the difference in overall
    promoted vs non-promoted sales -- the one that got us going down this path
    in the first place -- is real as well.
    """
)

# Prep data for display, collecting printable results in display_string
promo_sales = df[(df.open == 1) & (df.promo == 1)].sales
non_promo_sales = df[(df.open == 1) & (df.promo == 0)].sales

display_string = 'Two-sided T-test for means:\n'
display_string += '\t${:,.2f}\n'.format(promo_sales.mean())
display_string += '\t${:,.2f}\n'.format(non_promo_sales.mean())

# Run t-test
t_val, t_p_val = stats.ttest_ind(promo_sales, non_promo_sales)

# Finalize and display
display_string += f'T-val: {t_val}, p-val: {t_p_val}\n'
cd.display.code_block(display_string)

cd.display.markdown(
    """
    Yup, it's real.
    """
)
