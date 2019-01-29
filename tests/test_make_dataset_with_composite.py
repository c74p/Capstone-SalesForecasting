import datetime
from hypothesis import example, given, HealthCheck, settings
from hypothesis import Verbosity # NOQA
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import composite, datetimes, floats, integers, just
from hypothesis.strategies import one_of, sampled_from, SearchStrategy, text
import numpy as np
import pandas as pd
import pytest
from typing import Dict
from unittest import TestCase, mock

import sys, os # NOQA
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + '/../')
from src.data import make_dataset # NOQA

# This is the test file for the src/data/make_dataset.py file.

pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception


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


@composite
def create_dataframes(draw) -> Dict[str, pd.DataFrame]:
    """Generate dataframes for property-based testing."""

    # create strategies to be used in creating dataframes

    # define a 'plus_nan' strategy wrapper to explicitly include np.NaN
    @composite
    def plus_nan(draw, strat: SearchStrategy) -> SearchStrategy:
        return draw(one_of(just(np.NaN), strat))

    stores = integers(min_value=0, max_value=2000)
    stores_plus_nan = plus_nan(stores)

    states = sampled_from(state_abbreviations)
    states_plus_nan = plus_nan(states)

    dates = datetimes(min_value=datetime.datetime(2013, 1, 1),
                      max_value=datetime.datetime(2015, 12, 12))
    dates_plus_nan = plus_nan(dates)

    integers_plus_nan = plus_nan(integers())

    # Take the 'states' strategy and prepend 'Rossmann_DE' to what it gives you
    # Then add in NaN as a possibility for good measure
    google_files = states.flatmap(lambda state: just('Rossmann_DE_' + state))
    google_files_plus_nan = plus_nan(google_files)

    # create the strategy for spelling out a google_week entry (and add nan)
    @composite
    def create_google_weeks(draw, strat: SearchStrategy) -> SearchStrategy:
        day = draw(dates)
        idx = (day.weekday() + 1) % 7
        last_sun = day - datetime.timedelta(idx)
        next_sat = last_sun + datetime.timedelta(6)
        return last_sun.strftime('%Y-%m-%d') + ' - ' +\
            next_sat.strftime('%Y-%m-%d')
    google_weeks_plus_nan = plus_nan(create_google_weeks(dates))

    # Create dataframes from the strategies above
    # Note that each column has one of three strategies that include possible
    # nan values:
    # 1) It explicitly includes the 'plus_nan' wrapper
    # 2) It's sampled_from a list that explicitly includes nan
    # 3) It uses the 'floats' strategy, with allow_nan=True.  (The 'floats'
    #    strategy implicitly allows nans but PEP 20 dude)
    google_df = draw(data_frames([
        column('file', elements=google_files_plus_nan),
        column('week', elements=google_weeks_plus_nan),
        column('trend',
               elements=plus_nan(integers(min_value=0, max_value=100)))]))

    # Since this file is crucial to structuring the merged pdf, it's hard-coded
    state_names_df = pd.DataFrame({'StateName': state_names,
                                  'State': state_abbreviations})

    stores_df = draw(data_frames([
        column('Store', elements=stores_plus_nan, unique=True),
        column('StoreType',
               elements=sampled_from(['a', 'b', 'c', 'd', np.NaN])),
        column('Assortment', elements=sampled_from(['a', 'b', 'c', np.NaN])),
        column('CompetitionDistance',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('CompetitionOpenSinceMonth',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('CompetitionOpenSinceYear',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Promo2', elements=sampled_from([0, 1, np.NaN])),
        column('Promo2SinceWeek',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Promo2SinceYear',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('PromoInterval',
               elements=sampled_from(['Feb,May,Aug,Nov', 'Jan,Apr,Jul,Oct',
                                      'Mar,Jun,Sept,Dec', np.NaN]))
        ]))

    store_states_df = draw(data_frames([
        column('Store', elements=stores_plus_nan, unique=True),
        column('State', elements=states_plus_nan)
        ]))

    train_df = draw(data_frames([
        column('Store', elements=stores_plus_nan),
        column('DayOfWeek', elements=integers_plus_nan),
        column('Date', elements=dates_plus_nan),
        column('Sales', elements=integers_plus_nan),
        column('Customers', elements=integers_plus_nan),
        column('Open', elements=sampled_from([0, 1, np.NaN])),
        column('Promo', elements=sampled_from([0, 1, np.NaN])),
        column('StateHoliday',
               elements=sampled_from(['0', 'a', 'b', 'c', np.NaN])),
        column('SchoolHoliday', elements=sampled_from([0, 1, np.NaN]))
        ]))

    # Note that there are a lot of integer-valued columns in here; that's what
    # came out of the original dataframe. May need to revisit whether it's
    # better to code these as floats from the beginning.
    weather_df = draw(data_frames([
        column('file', elements=sampled_from([np.NaN] + state_names)),
        column('date', elements=dates_plus_nan),
        column('Max_TemperatureC', elements=integers_plus_nan),
        column('Min_TemperatureC', elements=integers_plus_nan),
        column('Dew_PointC', elements=integers_plus_nan),
        column('MeanDew_PointC', elements=integers_plus_nan),
        column('MinDew_PointC', elements=integers_plus_nan),
        column('Max_Humidity', elements=integers_plus_nan),
        column('Mean_Humidity', elements=integers_plus_nan),
        column('Min_Humidity', elements=integers_plus_nan),
        column('Max_Sea_Level_PressurehPa', elements=integers_plus_nan),
        column('Mean_Sea_Level_PressurehPa', elements=integers_plus_nan),
        column('Min_Sea_Level_PressurehPa', elements=integers_plus_nan),
        column('Max_VisibilityKm',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Mean_VisibilityKm',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Min_VisibilitykM',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Max_Wind_SpeedKm_h', elements=integers_plus_nan),
        column('Mean_Wind_SpeedKm_h', elements=integers_plus_nan),
        column('Max_Gust_SpeedKm_h',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('Precipitationmm',
               elements=floats(allow_infinity=False, allow_nan=True)),
        column('CloudCover', elements=sampled_from(['NA', np.NaN] +
               [str(i) for i in range(0, 9)])),
        column('Events', elements=sampled_from([np.NaN] +
               ['Rain', 'Fog-Rain-Snow', 'Snow', 'Rain-Snow', 'Fog-Snow',
                'Rain-Thunderstorm', 'Rain-Snow-Hail', 'Fog-Rain', 'Fog',
                'Fog-Snow-Hail', 'Thunderstorm', 'Fog-Rain-Thunderstorm',
                'Rain-Snow-Hail-Thunderstorm', 'Fog-Rain-Hail', 'Rain-Hail',
                'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow-Hail',
                'Fog-Thunderstorm', 'Rain-Snow-Thunderstorm',
                'Fog-Rain-Hail-Thunderstorm', 'Snow-Hail'])),
        column('WindDirDegrees', elements=integers_plus_nan),
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


def check_googletrend_csv(df_dict: Dict[str, pd.DataFrame]) -> None:
    """Check the transformations done with the file googletrend.csv

    No return value.
    """

    # Checks on googletrend.csv transformations
    if 'googletrend.csv' in df_dict.keys():
        google = df_dict['googletrend.csv']

        # Check that googletrend column 'file' values get translated to a
        # 'state' column correctly - everything in 'state' column should be 2
        # chars except for 'HB,NI'
        if 'file' in google.columns and len(google[google.file.notnull()]) > 0:
            assert all(google.loc[google['state'].str.len() > 2] == 'HB,NI')
            assert all(google.loc[google.state != 'HB,NI', 'state'].str.len()
                       == 2)

        # Check that dates get added correctly - for each state where 'week'
        # appears in the input df, the output df should have daily values for
        # each day from start to end
        if 'date' in google.columns and len(google[google.date.notnull()]) > 0:
            for st in google.state.unique():
                if len(google.loc[(google.state == st) &
                       (google.date.notnull())]) > 0:
                    min = google.loc[google.state == st, 'date'].min()
                    max = google.loc[google.state == st, 'date'].max()
                    assert len(google.loc[google.state == st]) == \
                        (max - min)/pd.Timedelta(days=1)


@pytest.mark.props
@given(create_dataframes())
@example({'googletrend.csv': pd.DataFrame({'file': ['HB,NI']})})
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
# Add in the setting below to @settings above when needed
#          verbosity=Verbosity.verbose)
def test_merge_csvs_properties(dfs):

    df_dict = make_dataset.merge_csvs(dfs)

    # EDIT remove this later if still considered a bad idea
    # Check on csv and dataframe naming formatting
    # assert '.csv' not in ''.join(list(df_dict.keys()))

    # Check on column naming
    if 'weather.csv' in df_dict.keys():
        assert 'min_visibilityk_m' not in df_dict['weather.csv'].columns
        assert 'min_visibility_km' in df_dict['weather.csv'].columns
    # Make sure all column names are lower-case
    assert ''.join(list(df_dict.keys())).lower() == \
           ''.join(list(df_dict.keys()))

    # EDIT UPDATE THIS FOR THE WHOLE DATAFRAME WHEN IT'S DONE
    # Check that NaNs are removed appropriately.
    # For 'store', 'sales', 'date', 'week': NaNs fundamentally
    # change the meaning of the data, so those remain NaNs and will
    # be removed in the merge later
    for name, df in df_dict.items():
        if len(df) > 0 and df.isnull().any().any():
            for col in df.columns:
                if col not in ['store', 'sales', 'date', 'week']:
                    assert df[col].isnull().sum() == 0 or \
                        (df[col].isnull()).all()

    check_googletrend_csv(df_dict)


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
