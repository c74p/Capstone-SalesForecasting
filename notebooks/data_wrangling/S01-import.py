import cauldron as cd
import os
import sys

sys.path.append('../../src/data')
import make_dataset # NOQA, need the line above to get directories right

# Config filepaths
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
cd.shared.PROJ_ROOT = PROJ_ROOT
cd.shared.raw_directory = os.path.join(PROJ_ROOT, 'data', 'raw')

# Pull in all csvs, ignoring test.csv and submission.csv because no
# target-variable data in those
dfs_dict = make_dataset.import_csvs(cd.shared.raw_directory,
                                    ignore_files=['test.csv',
                                                  'sample_submission.csv'],
                                    header=0,
                                    low_memory=False)

print(dfs_dict.keys())

cd.display.markdown(
    """
    #Import dataframes

    Here we'll import the dataframes in the directory and show the
    head of each. \n
    - Note that the directory /data/raw may contain a couple of other files
    like 'test.csv' or 'sample_submission.csv', but they don't add any new
    information for our current purposes.\n

    ## train.csv
    This is the most important file - information by store by date. \n
    - Sales is our target variable. \n
    - Customers is the number of customers on that date.\n
    - Open and Promo are as they sound, and we don't have more information
    than that.\n
    - StateHoliday and SchoolHoliday are as they sound. Note that StateHoliday
    can be a (public holiday), b (Easter), c (Christmas), or 0 (none).\n

    """
    )

cd.display.table(dfs_dict['train.csv'].head())


cd.display.markdown(
    """
    ## store.csv
    This file has information about each particular store.\n
    - Store type: one of 'a', 'b', 'c', 'd'\n
    - Assortment: 'a' = basic, 'b' = extra, 'c' = extended\n
    - CompetitionDistance is in meters. The closest competitor to any given
    store is 20 meters, while the furthest 'closest competitor' is nearly 50
    miles from a Rossmann store.\n
    - CompetitionOpenSinceMonth and Year are as they sound. Most of the
    competitors have been opened relatively recently.\n
    - Promo2: according to data/raw/description.txt, Promo2 is a continuing
    and consecutive promotion for some stores. 0 = not participating, 1 = 
    participating.\n
    - Promo2SinceWeek and Year are as they sound. Note that if Promo2 = 0, a
    NaN value is meaningful here.\n
    - PromoInterval: relative to Promo2.  Options are 'Jan,Apr,Jul,Oct',
    'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec' - note that 'Sept' here has 4
    characters.\n
    """
    )

cd.display.table(dfs_dict['store.csv'].head())

cd.display.markdown(
    """
    ## weather.csv
    This file has weather by state and date.\n
    - file: This is the name of the state rather than its abbreviation, which
    necessitates the use of the store_states csv later.\n
    - Other than Date, the rest of the file is various weather measurements.\n
    """
    )

cd.display.table(dfs_dict['weather.csv'].head())

cd.display.markdown(
    """
    ## googletrend.csv
    This file has google search trends by state and date.\n
    - file: This is the state abbreviation, along with some other characters
    that we'll strip out.\n
    - week: This is the week of the measurement.\n
    - trend: this is the trend, which we'll concatenate to our dataframe.\n
    """
    )

cd.display.table(dfs_dict['googletrend.csv'].head())

cd.display.markdown(
    """
    ## store_states.csv
    This file lists the state that each store is in, so we can merge the
    dataframes together.\n
    """
    )

cd.display.table(dfs_dict['store_states.csv'].head())


cd.display.markdown(
    """
    ## state_names.csv
    This file lists state names and abbreviations, so we can merge the
    dataframes together.\n
    """
    )

cd.display.table(dfs_dict['state_names.csv'].head())

# Make sure to share dfs_dict in memory for next step
cd.shared.put('dfs_dict', dfs_dict)
