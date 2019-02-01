import datetime
from hypothesis import assume, example, given, HealthCheck, note, settings
from hypothesis import unlimited
from hypothesis import Verbosity # NOQA
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import composite, floats, integers, just
from hypothesis.strategies import sampled_from, SearchStrategy, text
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

    # Min and max numbers of examples to generate unless we have good reason
    # to do otherwise
    ceiling = 10
    floor = 1

    # Create a strategy to generate store numbers
    # Note that in order to create well-formed dataframes, we'll limit the
    # number of choices to much lower than the real number
    stores = integers(min_value=0, max_value=ceiling)

    # Note that in order to create well-formed dataframes, here we'll assign
    # each store to a state - meaning if we want to generate a state, we'll
    # first choose a store number and then use that to assign a state
    number_of_states = len(state_abbreviations)

    @composite
    def state_strat(draw) -> SearchStrategy:
        store = draw(stores)
        state_num = store % number_of_states
        return state_abbreviations[state_num]

    states = state_strat()

    # Note that in order to create well-formed dataframes, here we'll assign
    # each store to a date - meaning if we want to generate a date, we'll
    # first choose a store number and then use that to assign a date
    date_range = pd.date_range(start='2013-01-01', end='2015-12-12', freq='D')
    number_of_dates = len(date_range)

    @composite
    def date_strat(draw) -> SearchStrategy:
        store = draw(stores)
        date_num = store % number_of_dates
        return date_range[date_num]

    dates = date_strat()

    # Take the 'states' strategy and prepend 'Rossmann_DE' to what it gives you
    # Then add in NaN as a possibility for good measure
    google_files = states.flatmap(lambda state: just('Rossmann_DE_' + state))

    # create the strategy for spelling out a google_week entry (and add nan)
    @composite
    def create_google_weeks(draw) -> SearchStrategy:
        day = draw(dates)
        idx = (day.weekday() + 1) % 7
        last_sun = day - datetime.timedelta(idx)
        next_sat = last_sun + datetime.timedelta(6)
        return last_sun.strftime('%Y-%m-%d') + ' - ' +\
            next_sat.strftime('%Y-%m-%d')
    google_weeks = create_google_weeks()

    # Create dataframes from the strategies above
    # We'll create dataframes with all non-NaN values, then add NaNs to rows
    # after the fact
    # Note assuming unique weeks here - an engineering choice (i.e. hack)
    google_df = draw(data_frames([
        column('file', elements=google_files),
        column('week', elements=google_weeks, unique=True),
        column('trend',
               elements=(integers(min_value=0, max_value=100)))],
        index=range_indexes(min_size=floor, max_size=ceiling)))

    # Add the nans and one row that connects with others
    rows = len(google_df)
    google_df.loc[rows] = [np.NaN, np.NaN, np.NaN]
    google_df.loc[rows+1] = [np.NaN, '2014-01-05 - 2014-01-11', 42]
    google_df.loc[rows+2] = ['Rossmann_DE_BE', np.NaN, 42]
    google_df.loc[rows+3] = \
        ['Rossmann_DE_BE', '2014-01-05 - 2014-01-11', np.NaN]
    google_df.loc[rows+4] = \
        ['Rossmann_DE_BE', '2014-01-05 - 2014-01-11', 42]

    # Hard coding here since it's crucial to folding the df together correctly
    # Also it's easy to see this csv is correct by inspection (16 rows)
    state_names_df = pd.DataFrame({'StateName': state_names,
                                  'State': state_abbreviations})

    # We'll create a stores dataframe allowing non-unique values for store,
    # then remove any overlaps - in the hope that it's easier to generate
    # Note assuming unique stores here - an engineering choice (i.e. hack)
    stores_df = draw(data_frames(columns=[
        column('Store', elements=stores, unique=True),
        column('StoreType',
               elements=sampled_from(['a', 'b', 'c', 'd'])),
        column('Assortment', elements=sampled_from(['a', 'b', 'c'])),
        column('CompetitionDistance',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('CompetitionOpenSinceMonth',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('CompetitionOpenSinceYear',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Promo2', elements=sampled_from([0, 1])),
        column('Promo2SinceWeek',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Promo2SinceYear',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('PromoInterval',
               elements=sampled_from(['Feb,May,Aug,Nov', 'Jan,Apr,Jul,Oct',
                                      'Mar,Jun,Sept,Dec']))],
        index=range_indexes(min_size=floor, max_size=ceiling)))

    # Add the nans and one row that connects
    rows = len(stores_df)
    len_cols = len(stores_df.columns)
    ref_row = [42, 'b', 'b', 42, 2, 2013, 1, 42, 2013, 'Feb,May,Aug,Nov']
    for col_num in range(len_cols):
        new_row = ref_row[:col_num] + [np.NaN] + ref_row[col_num + 1:]
        stores_df.loc[rows+col_num] = new_row
    stores_df.loc[rows+len_cols] = [np.NaN for i in range(len_cols)]
    stores_df.loc[rows+len_cols+1] = ref_row

    # We'll create a store_states dataframe allowing non-unique values for
    # store, then remove any overlaps - in the hope that it's easier to
    # create.
    # Note assuming unique stores here - an engineering choice (i.e. hack)
    store_states_df = draw(data_frames(columns=[
        column('Store', elements=stores, unique=True),
        column('State', elements=states)],
        index=range_indexes(min_size=floor, max_size=ceiling)))

    # Add the nans and one row that connects
    rows = len(store_states_df)
    len_cols = len(store_states_df.columns)
    ref_row = [42, 'BE']
    for col_num in range(len_cols):
        new_row = ref_row[:col_num] + [np.NaN] + ref_row[col_num + 1:]
        store_states_df.loc[rows+col_num] = new_row
    store_states_df.loc[rows+len_cols] = [np.NaN for i in range(len_cols)]
    store_states_df.loc[rows+len_cols+1] = ref_row

    # We'll create a train dataframe allowing non-unique values for
    # store, then remove any overlaps - in the hope that it's easier to
    # create.
    # Note assuming unique stores here - an engineering choice (i.e. hack)
    train_df = draw(data_frames(columns=[
        column('Store', elements=stores, unique=True),
        column('DayOfWeek', elements=integers()),
        column('Date', elements=dates),
        column('Sales', elements=integers()),
        column('Customers', elements=integers()),
        column('Open', elements=sampled_from([0, 1])),
        column('Promo', elements=sampled_from([0, 1])),
        column('StateHoliday',
               elements=sampled_from(['0', 'a', 'b', 'c'])),
        column('SchoolHoliday', elements=sampled_from([0, 1]))],
        index=range_indexes(min_size=floor, max_size=ceiling)
        ))

    # Add the nans and one row that connects
    rows = len(train_df)
    len_cols = len(train_df.columns)
    ref_row = [42, 1, '2014-01-06', 42, 42, 1, 1, 0, 0]
    for col_num in range(len_cols):
        new_row = ref_row[:col_num] + [np.NaN] + ref_row[col_num + 1:]
        train_df.loc[rows+col_num] = new_row
    train_df.loc[rows+len_cols] = [np.NaN for i in range(len_cols)]
    train_df.loc[rows+len_cols+1] = ref_row

    # Note that there are a lot of integer-valued columns in here; that's what
    # came out of the original dataframe. May need to revisit whether it's
    # better to code these as floats from the beginning.
    # Note assuming unique dates here - an engineering choice (i.e. hack)
    weather_df = draw(data_frames([
        column('file', elements=sampled_from(state_names)),
        column('date', elements=dates, unique=True),
        column('Max_TemperatureC', elements=integers()),
        column('Mean_TemperatureC', elements=integers()),
        column('Min_TemperatureC', elements=integers()),
        column('Dew_PointC', elements=integers()),
        column('MeanDew_PointC', elements=integers()),
        column('Min_DewpointC', elements=integers()),
        column('Max_Humidity', elements=integers()),
        column('Mean_Humidity', elements=integers()),
        column('Min_Humidity', elements=integers()),
        column('Max_Sea_Level_PressurehPa', elements=integers()),
        column('Mean_Sea_Level_PressurehPa', elements=integers()),
        column('Min_Sea_Level_PressurehPa', elements=integers()),
        column('Max_VisibilityKm',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Mean_VisibilityKm',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Min_VisibilitykM',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Max_Wind_SpeedKm_h', elements=integers()),
        column('Mean_Wind_SpeedKm_h', elements=integers()),
        column('Max_Gust_SpeedKm_h',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('Precipitationmm',
               elements=floats(allow_infinity=False, allow_nan=False)),
        column('CloudCover', elements=sampled_from(['NA'] +
               [str(i) for i in range(0, 9)])),
        column('Events', elements=sampled_from(
               ['Rain', 'Fog-Rain-Snow', 'Snow', 'Rain-Snow', 'Fog-Snow',
                'Rain-Thunderstorm', 'Rain-Snow-Hail', 'Fog-Rain', 'Fog',
                'Fog-Snow-Hail', 'Thunderstorm', 'Fog-Rain-Thunderstorm',
                'Rain-Snow-Hail-Thunderstorm', 'Fog-Rain-Hail', 'Rain-Hail',
                'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow-Hail',
                'Fog-Thunderstorm', 'Rain-Snow-Thunderstorm',
                'Fog-Rain-Hail-Thunderstorm', 'Snow-Hail'])),
        column('WindDirDegrees', elements=integers())],
        index=range_indexes(min_size=floor, max_size=ceiling)))

    # Add the nans and one row that connects
    rows = len(weather_df)
    len_cols = len(weather_df.columns)
    ref_row = ['BE', '2014-01-06'] + [1 for i in range(20)] + ['Rain'] + [1]
    for col_num in range(len_cols):
        new_row = ref_row[:col_num] + [np.NaN] + ref_row[col_num + 1:]
        weather_df.loc[rows+col_num] = new_row
    weather_df.loc[rows+len_cols] = [np.NaN for i in range(len_cols)]
    weather_df.loc[rows+len_cols+1] = ref_row

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
    if 'googletrend.csv' in df_dict.keys() and \
            df_dict['googletrend.csv'] is not None:
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
                if len(google.loc[(google.state == st)]) > 0:
                    weeks = google.loc[google.state == st,
                                       'week_start'].unique()
                    days = len(google.loc[google.state == st])
                    # min = google.loc[google.state == st, 'date'].min()
                    # max = google.loc[google.state == st, 'date'].max()
                    assert days % len(weeks) == 0


@pytest.mark.props
@given(create_dataframes())
@example({'googletrend.csv': pd.DataFrame({'file': ['HB,NI']})})
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow,
          HealthCheck.filter_too_much,
          HealthCheck.data_too_large], timeout=unlimited,
          max_examples=10)
# Add in the setting below to @settings above when needed
#          verbosity=Verbosity.verbose)
def test_merge_csvs_properties(input_df_dict: Dict[str, pd.DataFrame]) -> None:
    """Test make_dataset.merge_csvs.  No return value"""

    input_dataframe, df_dict = make_dataset.merge_csvs(input_df_dict)

    assume(df_dict['train.csv']['date'].notnull().any())

    # EDIT Consider changing these to check on the final dataframe once it's
    # available
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
        if df is not None and len(df) > 0 and df.isnull().any().any():
            for col in df.columns:
                if col not in ['store', 'sales', 'date', 'week', 'file']:
                    assert df[col].isnull().sum() == 0 or \
                        (df[col].isnull()).all()

    check_googletrend_csv(df_dict)

    # If state_names.csv is included, appropriate columns should be there
    if 'state_names.csv' in df_dict.keys() and \
            df_dict['state_names.csv'] is not None and\
            len(df_dict['state_names.csv']) > 0:
        # This is a separate condition to avoid a Keyerror
        if any(df_dict['state_names.csv']['state'].notnull()):
            # assert 'state_name' in input_dataframe.columns
            # assert 'store' in input_dataframe.columns
            # assert 'state' in input_dataframe.columns
            pass

    # If weather.csv is included, appropriate columns should be there
    if 'weather.csv' in df_dict.keys() and \
            df_dict['weather.csv'] is not None and \
            len(df_dict['weather.csv']) > 0:
        # This is a separate condition to avoid a Keyerror
        if any(df_dict['weather.csv']['date'].notnull()):
            for col in ['cloud_cover', 'date', 'dew_point_c', 'events',
                        'max_gust_speed_km_h', 'max_humidity',
                        'max_sea_level_pressureh_pa', 'max_temperature_c',
                        'max_visibility_km', 'max_wind_speed_km_h',
                        'mean_dew_point_c', 'mean_humidity',
                        'mean_sea_level_pressureh_pa', 'mean_temperature_c',
                        'mean_visibility_km', 'mean_wind_speed_km_h',
                        'min_dew_point_c', 'min_humidity',
                        'min_sea_level_pressureh_pa', 'min_temperature_c',
                        'min_visibility_km', 'precipitationmm', 'state',
                        'wind_dir_degrees']:
                # assert col in input_dataframe.columns
                pass

    # If googletrend.csv is included, appropriate columns should be there
    if 'googletrend.csv' in df_dict.keys() and \
            df_dict['googletrend.csv'] is not None and \
            any(df_dict['googletrend.csv'].notnull().sum() > 0):
        # This is a separate condition to avoid a Keyerror
        if any(df_dict['googletrend.csv']['date'].notnull()):
            # assert 'trend' in input_dataframe.columns
            pass

    # Appropriate columns from store.csv should be there
    for col in ['assortment', 'competition_distance',
                'competition_open_since_month', 'competition_open_since_year',
                'promo2', 'promo2_since_week', 'promo2_since_year',
                'promo_interval', 'store', 'store_type']:
        # assert col in input_dataframe.columns
        pass

    # Appropriate columns from train.csv should be there
    for col in ['customers', 'date', 'day_of_week', 'open', 'promo', 'sales',
                'school_holiday', 'state_holiday', 'store']:
        # assert col in input_dataframe.columns
        pass

    note('train date printout:' + str(df_dict['train.csv'].date))

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
