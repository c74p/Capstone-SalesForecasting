from hypothesis import assume, HealthCheck, given, settings
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import sampled_from
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
    column('week', dtype='datetime64[ns]'),
    column('trend', dtype='int64')])

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
    column('CompetitionDistance', dtype='float64'),
    column('CompetitionOpenSinceMonth', dtype='float64'),
    column('CompetitionOpenSinceYear', dtype='float64'),
    column('Promo2', elements=sampled_from([0, 1])),
    column('Promo2SinceWeek', dtype='float64'),
    column('Promo2SinceYear', dtype='float64'),
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
    column('Store', dtype='int64'),
    column('DayOfWeek', dtype='int64'),
    column('Date', dtype='datetime64[ns]'),
    column('Sales', dtype='int64'),
    column('Customers', dtype='int64'),
    column('Open', elements=sampled_from([0, 1])),
    column('Promo', elements=sampled_from([0, 1])),
    column('StateHoliday', elements=sampled_from(['0', 'a', 'b', 'c'])),
    column('SchoolHoliday', elements=sampled_from([0, 1]))
    ])

# Configuration and strategy for weather.csv file
weather_strat = data_frames([
    column('file', elements=sampled_from(state_names)),
    column('date', dtype='datetime64[ns]'),
    column('Max_TemperatureC', dtype='int64'),
    column('Min_TemperatureC', dtype='int64'),
    column('Dew_PointC', dtype='int64'),
    column('MeanDew_PointC', dtype='int64'),
    column('MinDew_PointC', dtype='int64'),
    column('Max_Humidity', dtype='int64'),
    column('Mean_Humidity', dtype='int64'),
    column('Min_Humidity', dtype='int64'),
    column('Max_Sea_Level_PressurehPa', dtype='int64'),
    column('Mean_Sea_Level_PressurehPa', dtype='int64'),
    column('Min_Sea_Level_PressurehPa', dtype='int64'),
    column('Max_VisibilityKm', dtype='float64'),
    column('Mean_VisibilityKm', dtype='float64'),
    column('Min_VisibilitykM', dtype='float64'),
    column('Max_Wind_SpeedKm_h', dtype='int64'),
    column('Mean_Wind_SpeedKm_h', dtype='int64'),
    column('Max_Gust_SpeedKm_h', dtype='float64'),
    column('Precipitationmm', dtype='float64'),
    column('CloudCover', elements=sampled_from(['NA'] +
           [str(i) for i in range(0, 9)])),
    column('Events', elements=sampled_from([np.nan] +
           ['Rain', 'Fog-Rain-Snow', 'Snow', 'Rain-Snow', 'Fog-Snow',
            'Rain-Thunderstorm', 'Rain-Snow-Hail', 'Fog-Rain', 'Fog',
            'Fog-Snow-Hail', 'Thunderstorm', 'Fog-Rain-Thunderstorm',
            'Rain-Snow-Hail-Thunderstorm', 'Fog-Rain-Hail', 'Rain-Hail',
            'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow-Hail', 'Fog-Thunderstorm',
            'Rain-Snow-Thunderstorm', 'Fog-Rain-Hail-Thunderstorm',
            'Snow-Hail'])),
    column('WindDirDegrees', dtype='int64'),
    ])


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(google_strat, stores_strat, store_states_strat, train_strat,
       weather_strat)
def test_merge_csvs_properties(google_df, stores_df, store_states_df, train_df,
                               weather_df):
    assume(all([len(google_df) > 0, len(stores_df) > 0,
                len(store_states_df) > 0, len(train_df) > 0,
                len(weather_df) > 0]))
    assert google_df['file'].dtype == object
    assert google_df['week'].dtype == '<M8[ns]'
    google_df['week'] = google_df['week'].dt.strftime('%Y-%m-%d')
    assert google_df['week'].dtype == object


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
