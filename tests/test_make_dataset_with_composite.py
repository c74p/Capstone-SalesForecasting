import datetime
from hypothesis import example, given, HealthCheck, settings
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import composite, datetimes, integers, just
from hypothesis.strategies import sampled_from, text
import numpy as np
import pandas as pd
import pytest
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


# Configuration to create dataframe strategies for hypothesis testing
# These may be used in the merge_csvs_* series of tests below

state_abbreviations = ["BB", "BE", "BW", "BY", "HB", "HB,NI", "HE", "HH", "MV",
                       "NW", "RP", "SH", "SL", "SN", "ST", "TH"]

state_names = ["BadenWuerttemberg", "Bayern", "Berlin", "Brandenburg",
               "Bremen", "Hamburg", "Hessen", "MecklenburgVorpommern",
               "Niedersachsen", "NordrheinWestfalen", "RheinlandPfalz",
               "Saarland", "Sachsen", "SachsenAnhalt", "SchleswigHolstein",
               "Thueringen"]

google_file_vals = ["Rossmann_DE", "Rossmann_DE_BE", "Rossmann_DE_BW",
                    "Rossmann_DE_BY", "Rossmann_DE_HE", "Rossmann_DE_HH",
                    "Rossmann_DE_NI", "Rossmann_DE_NW", "Rossmann_DE_RP",
                    "Rossmann_DE_SH", "Rossmann_DE_SL", "Rossmann_DE_SN",
                    "Rossmann_DE_ST", "Rossmann_DE_TH"]


@composite
def create_dataframes(draw):
    """Generate dataframes for property-based testing."""

    # The next 25 or so lines are strategies to be used in creating dataframes
    stores = integers(min_value=0, max_value=2000)
    states = sampled_from(state_abbreviations)
    dates = datetimes(min_value=datetime.datetime(2013, 1, 1),
                      max_value=datetime.datetime(2015, 12, 12))

    # Take the 'states' strategy and prepend 'Rossmann_DE' to what it gives you
    google_files = states.flatmap(lambda state: just('Rossmann_DE_' + state))

    # Below we create the strategy for spelling out a google_week entry.
    # The monstrous thing below is the monadic version of the function in
    # comments here:
    # def create_google_weeks():
    #   today = draw(dates)
    #   idx = (today.weekday() + 1) % 7
    #   last_sun = today - datetime.timedelta(idx)
    #   next_sat = last_sun + datetime.timedelta(6)
    #   return last_sun.strftime('%Y-%m-%d') + ' - ' +\
    #       next_sat.strftime('%Y-%m-%d')
    google_weeks = dates.flatmap(lambda today: just((today.weekday() + 1) % 7)
        .flatmap(lambda idx: just(today - datetime.timedelta(idx))
        .flatmap(lambda last_sun: just(last_sun + datetime.timedelta(6))
        .flatmap(lambda next_sat: just(last_sun.strftime('%Y-%m-%d') + ' - ' +
            next_sat.strftime('%Y-%m-%d')))))) # NOQA

    # Create dataframes from the strategies above
    google_df = draw(data_frames([
        column('file', elements=google_files),
        column('week', elements=google_weeks),
        column('trend', elements=integers(min_value=0, max_value=100))]))

    # Since this file is crucial to structuring the merged pdf, it's hard-coded
    state_names_df = pd.DataFrame({'StateName': state_names,
                                  'State': state_abbreviations})

    stores_df = draw(data_frames([
        column('Store', elements=stores, dtype='int64', unique=True),
        column('StoreType', elements=sampled_from(['a', 'b', 'c', 'd'])),
        column('Assortment', elements=sampled_from(['a', 'b', 'c'])),
        column('CompetitionDistance', dtype='float64'),
        column('CompetitionOpenSinceMonth', dtype='float64'),
        column('CompetitionOpenSinceYear', dtype='float64'),
        column('Promo2', elements=sampled_from([0, 1])),
        column('Promo2SinceWeek', dtype='float64'),
        column('Promo2SinceYear', dtype='float64'),
        column('PromoInterval',
               elements=sampled_from(['Feb,May,Aug,Nov',
                                      'Jan,Apr,Jul,Oct',
                                      'Mar,Jun,Sept,Dec',
                                      np.nan]))
        ]))

    store_states_df = draw(data_frames([
        column('Store', elements=stores, unique=True),
        column('State', elements=states)
        ]))

    train_df = draw(data_frames([
        column('Store', elements=stores),
        column('DayOfWeek', dtype='int64'),
        column('Date', elements=dates),
        column('Sales', dtype='int64'),
        column('Customers', dtype='int64'),
        column('Open', elements=sampled_from([0, 1])),
        column('Promo', elements=sampled_from([0, 1])),
        column('StateHoliday', elements=sampled_from(['0', 'a', 'b', 'c'])),
        column('SchoolHoliday', elements=sampled_from([0, 1]))
        ]))

    weather_df = draw(data_frames([
        column('file', elements=sampled_from(state_names)),
        column('date', elements=dates),
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
                'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow-Hail',
                'Fog-Thunderstorm', 'Rain-Snow-Thunderstorm',
                'Fog-Rain-Hail-Thunderstorm', 'Snow-Hail'])),
        column('WindDirDegrees', dtype='int64'),
        ]))

    return {'googletrend.csv': google_df, 'state_names.csv': state_names_df,
            'store_states.csv': store_states_df, 'store.csv': stores_df,
            'train.csv': train_df, 'weather.csv': weather_df}


@given(text())
@example('Precipitation_Mm')
def test_convert_to_snake_case(t):
    new = make_dataset.convert_to_snake_case(t)
    assert new.lower() == new
    assert new.replace('__', 'XX') == new


@pytest.mark.props
@given(create_dataframes())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_merge_csvs_properties(dfs):
    # google = dfs['googletrend']
    # state_names = dfs['state_names']
    # states = dfs['store_states']
    # stores = dfs['store']
    # train = dfs['train']
    # weather = dfs['weather']
    # assert len(google) == 0 or google['file'].dtype == 'object'
    # assert len(google) == 0 or google['week'].dtype == 'object'
    # assert len(state_names) == 0 or state_names['State'].dtype == 'object'
    # assert len(states) == 0 or states['State'].dtype == 'object'
    # assert len(stores) == 0 or stores['Store'].dtype == 'int64'
    # assert len(train) == 0 or train['Store'].dtype == 'int64'
    # assert len(weather) == 0 or weather['file'].dtype == 'object'

    new_df = make_dataset.merge_csvs(dfs)

    # Check on csv and dataframe naming formatting
    assert '.csv' not in ''.join(list(new_df.keys()))
    assert 'googletrend' not in list(new_df.keys())
    # Check on column naming formatting
    assert 'min_visibilityk_m' not in new_df['weather'].columns
    assert 'min_visibility_km' in new_df['weather'].columns
    assert ''.join(list(new_df.keys())).lower() == ''.join(list(new_df.keys()))
    # Check on nan-filling

    # EDIT ADD ONE HERE FOR THE WHOLE DATAFRAME WHEN IT'S DONE
    for df in new_df.values():
        for col in df:  # col = column, name-shadowing made me choose 'col'
            assert len(new_df[df]) == 0 # or\
                # (new_df[df][col].isnull()).all() or\
                # new_df[df][col].isnull().sum() == 0

    assert len(new_df['store']) == 0 or\
        (new_df['store'].promo2_since_week.isnull()).all() or\
        new_df['store'].promo2_since_week.isnull().sum() == 0
    assert len(new_df['store']) == 0 or\
        (new_df['store'].promo2_since_year.isnull()).all() or\
        new_df['store'].promo2_since_year.isnull().sum() == 0
    assert len(new_df['store']) == 0 or\
        (new_df['store'].promo_interval.isnull()).all() or\
        new_df['store'].promo_interval.isnull().sum() == 0
    assert len(new_df['store']) == 0 or\
        (new_df['store'].competition_distance.isnull()).all() or\
        new_df['store'].competition_distance.isnull().sum() == 0
    assert len(new_df['store']) == 0 or\
        (new_df['store'].competition_open_since_month.isnull()).all() or\
        new_df['store'].competition_open_since_month.isnull().sum() == 0
    assert len(new_df['store']) == 0 or\
        (new_df['store'].competition_open_since_year.isnull()).all() or\
        new_df['store'].competition_open_since_year.isnull().sum() == 0


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
