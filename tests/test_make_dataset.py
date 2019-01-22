import datetime
from hypothesis import assume, given
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import datetimes, just, one_of, sampled_from
import numpy as np
import pandas as pd
from src.data import make_dataset
from unittest import TestCase, mock

# This is the test file for the src/data/make_dataset.py file.
# Note that it uses the object-oriented unittest style for some parts, and
# the stripped-down function approach (after definition of appropriate
# strategies) for property-based testing in Hypothesis. It may be a little
# jarring, but I found the function approach easier to use in Hypothesis.


class test_Import_Csvs(TestCase):

    def setUp(self):

        # Config file directory and read_csv return values for mock
        # These may be used in the import_csvs_* series of tests below
        self.fake_files = ['a.csv', 'b.csv', 'c.csv']
        self.fake_read = ['', '', '']

        # Config empty directory and read_csv return values for mock
        # These may be used in the import_csvs_* series of tests below
        self.fake_empty_files = []
        self.fake_empty_read = ['', '', '']

    def tearDown(self):
        pass

    def test_import_csvs_pulls_all_csvs(self):
        """All available csvs in the directory should be pulled"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv', side_effect=self.fake_read):
                read = make_dataset.import_csvs('bogus_dir')
                assert read == {k: v for k, v in
                                zip(self.fake_files, self.fake_read)}

    def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
        """Nothing should be returned from an empty directory"""
        with mock.patch('os.listdir', return_value=self.fake_empty_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_empty_read):
                read = make_dataset.import_csvs('bogus_dir')
                assert read == {}

    def test_import_csvs_can_ignore_files(self):
        """A single 'ignore_files=' file should be ignored"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv', side_effect=self.fake_read):
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files='b.csv')
                assert read == {'a.csv': '', 'c.csv': ''}

    def test_import_csvs_can_ignore_files_as_list(self):
        """A list of 'ignore_files=' files should be ignored"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_read) as mock_pandas:
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files=['b.csv'])
                assert read == {'a.csv': '', 'c.csv': ''}
                # how do I assert 'ignore_files' not in kwargs when the
                # function is called?
                mock_pandas.assert_called_with('bogus_dir/c.csv')


# Config dataframe strategies for hypothesis testing
# These may be used in the merge_csvs_* series of tests below


# Configuration and strategy for googletrend.csv file
google_file_vals = ["Rossmann_DE", "Rossmann_DE_BE", "Rossmann_DE_BW",
                    "Rossmann_DE_BY", "Rossmann_DE_HE", "Rossmann_DE_HH",
                    "Rossmann_DE_NI", "Rossmann_DE_NW", "Rossmann_DE_RP",
                    "Rossmann_DE_SH", "Rossmann_DE_SL", "Rossmann_DE_SN",
                    "Rossmann_DE_ST", "Rossmann_DE_TH"]

google_strat = data_frames([
    column('file', elements=sampled_from(google_file_vals)),
    column('week', elements=datetimes(
        min_value=datetime.datetime(2000, 1, 1, 0, 0, 0),
        max_value=datetime.datetime(2018, 12, 31, 11, 59, 59))),
    column('trend', elements=sampled_from(range(0, 101)))])

# Configuration for state_names.csv file
# Since this file is crucial to structuring the merged pdf, it's hard-coded
state_names = ["BadenWuerttemberg", "Bayern", "Berlin", "Brandenburg",
               "Bremen", "Hamburg", "Hessen", "MecklenburgVorpommern",
               "Niedersachsen", "NordrheinWestfalen", "RheinlandPfalz",
               "Saarland", "Sachsen", "SachsenAnhalt", "SchleswigHolstein",
               "Thueringen"]

state_abbreviations = ["BB", "BE", "BW", "BY", "HB", "HB,NI", "HE", "HH", "MV",
                       "NW", "RP", "SH", "SL", "SN", "ST", "TH"]

state_names_df = pd.DataFrame({'StateName': state_names,
                              'State': state_abbreviations})

# Configuration and strategy for store.csv file
stores_strat = data_frames([
    column('Store', elements=sampled_from(range(1, 1116))),
    column('StoreType', elements=sampled_from(['a', 'b', 'c', 'd'])),
    column('Assortment', elements=sampled_from(['a', 'b', 'c'])),
    column('CompetitionDistance',
           elements=one_of(just(np.nan),
                           [float(i) for i in range(0, 100000)])),
    column('CompetitionOpenSinceMonth',
           elements=one_of(just(np.nan),
                           [float(i) for i in range(1, 13)])),
    column('CompetitionOpenSinceYear',
           elements=one_of(just(np.nan),
                           [float(i) for i in range(1900, 2016)])),
    column('Promo2', elements=sampled_from([0, 1])),
    column('Promo2SinceWeek',
           elements=one_of(just(np.nan),
                           [float(i) for i in range(0, 51)])),
    column('Promo2SinceYear',
           elements=one_of(just(np.nan),
                           [float(i) for i in range(2009, 2016)])),
    column('PromoInterval', elements=sampled_from(['Feb,May,Aug,Nov',
                                                   'Jan,Apr,Jul,Oct',
                                                   'Mar,Jun,Sept,Dec',
                                                   np.nan
                                                   ]))
    ])

# Configuration and strategy for store_states.csv file
store_states_strat = data_frames([
    column('Store', elements=sampled_from(range(1, 1116))),
    column('State', elements=sampled_from(["BE", "BW", "BY", "HB,NI", "HE",
                                           "HH", "NW", "RP", "SH", "SN", "ST",
                                           "TH"]))
    ])

# Configuration and strategy for train.csv file
train_strat = data_frames([
    column('Store', elements=sampled_from(range(1, 1116))),
    column('DayOfWeek', elements=sampled_from(range(1, 8))),
    column('Date', elements=one_of([d.strftime('%Y-%m-%d') for d in
                                    range('2013-01-01', '2015-07-31')])),
    column('Sales', elements=sampled_from(range(0, 50000))),
    column('Customers', elements=sampled_from(range(0, 10000))),
    column('Open', elements=one_of([0, 1])),
    column('Promo', elements=one_of([0, 1])),
    column('StateHoliday', elements=sampled_from('0', 'a', 'b', 'c')),
    column('SchoolHoliday', elements=one_of([0, 1]))
    ])

# Configuration and strategy for weather.csv file
weather_strat = data_frames([
    column('file', elements=sampled_from(state_names)),
    column('date', elements=range('2013-01-01 00:00:00',
                                  '2015-09-17 00:00:00')),
    column('Max_TemperatureC', elements=sampled_from(range(-20, 45))),
    column('Min_TemperatureC', elements=sampled_from(range(-20, 45))),
    column('Dew_PointC', elements=sampled_from(range(-20, 45))),
    column('MeanDew_PointC', elements=sampled_from(range(-20, 45))),
    column('MinDew_PointC', elements=sampled_from(range(-80, 45))),
    column('Max_Humidity', elements=sampled_from(range(20, 100))),
    column('Mean_Humidity', elements=sampled_from(range(20, 100))),
    column('Min_Humidity', elements=sampled_from(range(0, 100))),
    column('Max_Sea_Level_PressurehPa', elements=sampled_from(range(900,
                                                                    1100))),
    column('Mean_Sea_Level_PressurehPa', elements=sampled_from(range(900,
                                                                     1100))),
    column('Min_Sea_Level_PressurehPa', elements=sampled_from(range(900,
                                                                    1100))),
    column('Max_VisibilityKm',
           elements=sampled_from([float(i) for i in range(0, 40)])),
    column('Mean_VisibilityKm',
           elements=sampled_from([float(i) for i in range(0, 40)])),
    column('Min_VisibilitykM',
           elements=sampled_from([float(i) for i in range(0, 40)])),
    column('Max_Wind_SpeedKm_h', elements=sampled_from(range(0, 110))),
    column('Mean_Wind_SpeedKm_h', elements=sampled_from(range(0, 110))),
    column('Max_Gust_SpeedKm_h',
           elements=sampled_from([float(i) for i in range(10, 120)]))  # ,
    # Check Max_Gust_SpeedKm_h; add Precipitationmm, CloudCover, Events, and
    # WindDirDegrees
    ])


@given(google_strat)
def test_test_of_hypothesis(df):
    assume(len(df) > 0)
    assert df['file'].dtype == object
    assert df['week'].dtype == '<M8[ns]'
    df['week'] = df['week'].dt.strftime('%Y-%m-%d')
    assert df['week'].dtype == object


def test_merge_csvs():
    pass
# Want merge_all_csvs() to:
# - merge all the csvs together into one, appropriately
# What could go wrong?
# - Not all the csvs could be there
# - bunch of NaNs
# - files could incorrectly come together


def test_verify_csv_pull():
    pass
# Want verify_csv_pull() to:
# - Check the csv pull and send a message to user
#   - Either pull was successful, or pull failed, why, and what to do next
# What could go wrong?
# - Not all the csvs could be there


def test_XXX_Test_Name():
    pass
    # raise NotImplementedError('Insert test code here.')
    #  Examples:
    # self.assertEqual(fp.readline(), 'This is a test')
    # self.assertFalse(os.path.exists('a'))
    # self.assertTrue(os.path.exists('a'))
    # self.assertTrue('already a backup server' in c.stderr)
    # self.assertIn('fun', 'disfunctional')
    # self.assertNotIn('crazy', 'disfunctional')
    # with self.assertRaises(Exception):
    #   raise Exception('test')
    #
    # Unconditionally fail, e.g. in a try block that should raise
    # self.fail('Exception was not raised')
