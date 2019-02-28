from hypothesis import given, example
from hypothesis.strategies import text
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
from src.data import make_dataset # NOQA

from tdda.constraints.pd.constraints import verify_df # NOQA
from tdda.referencetest import ReferenceTestCase # NOQA

# This is the test file for the src/data/make_dataset.py file.

pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception

PROJ_ROOT = Path('..')


class Test_Preprocessing(TestCase):
    """Test the preprocessing steps needed to either train a model, or
    predict using a row of data."""

    def setUp(self):

        # Config file directory and read_csv return values for mock
        # These may be used in the import_csvs_* series of tests below
        self.fake_files = ['a.csv', 'b.csv', 'c.csv']
        self.fake_read = ['', '', '']

        self.df =\
            pd.DataFrame(columns=['store', 'state', 'date',
                                  'max_temperature_c', 'mean_temperature_c',
                                  'min_temperature_c', 'dew_point_c',
                                  'mean_dew_point_c', 'min_dew_point_c',
                                  'max_humidity', 'mean_humidity',
                                  'min_humidity', 'max_sea_level_pressureh_pa',
                                  'mean_sea_level_pressureh_pa',
                                  'min_sea_level_pressureh_pa',
                                  'max_visibility_km', 'mean_visibility_km',
                                  'min_visibility_km', 'max_wind_speed_km_h',
                                  'mean_wind_speed_km_h',
                                  'max_gust_speed_km_h', 'precipitationmm',
                                  'cloud_cover', 'events', 'wind_dir_degrees',
                                  'store_type', 'assortment',
                                  'competition_distance',
                                  'competition_open_since_month',
                                  'competition_open_since_year', 'promo2',
                                  'promo2_since_week', 'promo2_since_year',
                                  'promo_interval', 'day_of_week', 'sales',
                                  'customers', 'open', 'promo',
                                  'state_holiday', 'school_holiday',
                                  'trend', 'week_start'],
                         data=[[1, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88, 64,
                                37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21, 13,
                                40.0, 0.0, 6.0, Rain, 290, c, a, 1270.0, 9.0,
                                2008.0, 0, 23.595446584938703,
                                2011.7635726795095, None, 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14'],
                               [56, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, c, 6620.0,
                                3.0, 2012.0, 1, 10.0, 2014.0,
                                "Mar,Jun,Sept,Dec", 5, 9351.0, 667.0, 1.0, 0.0,
                                0, 0.0, 85, '2015-06-14'],
                               [69, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, c, c, 1130.0,
                                7.224704336399474, 2008.6688567674114, 1, 40.0,
                                2011.0, "Jan,Apr,Jul,Oct", 5, 6895.0, 941.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14'],
                               [77, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, c, 1090.0,
                                8.0, 2009.0, 1, 10.0, 2014.0,
                                "Jan,Apr,Jul,Oct", 5, 7656.0, 687.0, 1.0, 0.0,
                                0, 0.0, 85, '2015-06-14'],
                               [111, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, c, 7890.0,
                                7.224704336399474, 2008.6688567674114, 1, 37.0,
                                2009.0, "Jan,Apr,Jul,Oct", 5, 6039.0, 600.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14'],
                               [120, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, a, 2290.0,
                                12.0, 2014.0, 1, 37.0, 2009.0,
                                "Jan,Apr,Jul,Oct", 5, 5135.0, 491.0, 1.0, 0.0,
                                0, 0.0, 85, '2015-06-14'],
                               [128, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, c, 2000.0,
                                7.224704336399474, 2008.6688567674114, 1, 1.0,
                                2013.0, "Jan,Apr,Jul,Oct", 5, 7604.0, 648.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14'],
                               [130, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, c, a, 900.0,
                                7.224704336399474, 2008.6688567674114, 1, 13.0,
                                2010.0, "Jan,Apr,Jul,Oct", 5, 4318.0, 482.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14'],
                               [135, HE, '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, Rain, 290, d, a, 5190.0,
                                7.224704336399474, 2008.6688567674114, 1, 1.0,
                                2013.0, "Jan,Apr,Jul,Oct", 5, 5823.0, 595.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14']])

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
        """A list of 'ignore_files=' files should be ignored.
        This is the version with a singleton list"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_read) as mock_pandas:
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files=['b.csv'])
                assert read == {'a.csv': '', 'c.csv': ''}
                # This remembers the last call so we only specify 'c.csv'
                mock_pandas.assert_called_with('bogus_dir/c.csv')

    def test_import_csvs_can_ignore_files_as_list_2(self):
        """A list of 'ignore_files=' files should be ignored.
        This is the version with a non-singleton list"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_read) as mock_pandas:
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files=['b.csv',
                                                              'c.csv'])
                assert read == {'a.csv': ''}
                mock_pandas.assert_called_with('bogus_dir/a.csv')


# @pytest.mark.skip(reason='takes too long right now')
@given(text())
@example('Precipitation_Mm')
def test_convert_to_snake_case(t):
    "Check that convert_to_snake_case lower-cases without leaving '__'."""
    new = make_dataset.convert_to_snake_case(t)
    assert new.lower() == new
    assert '__' not in new


# Constraint testing on the initial files and on the generated 'wrangled' file

class test_Merge_Csvs(TestCase):

    def setUp(self):
        """Set paths for constraint files, as well as raw csvs and processed
        merged csv. Also pull in the raw csvs and run merge_dfs on it, since
        we're already opening the files in order to do constraint testing.
        Since we're opening the files themselves anyway and merge_dfs is a
        custom function for this specific set of data files, we'll just test
        against the actual raw files."""

        # Basic configuration
        CONSTRAINTS_PATH = Path('../data/interim/constraints_initial_csvs')
        RAW_CSV_PATH = Path('../data/raw')
        PROCESSED_PATH = Path('../data/processed')
        self.constraint_paths = {}
        self.raw_csv_paths = {}
        self.raw_dfs_dict = {}
        self.dfs_dict = {}
        self.filenames = ['googletrend.csv', 'state_names.csv',
                          'store_states.csv', 'store.csv', 'train.csv',
                          'weather.csv']

        # loop through and create paths to constraints files and raw files,
        # and open the raw files in order to check constraints and merge csvs
        for name in self.filenames:
            self.constraint_paths[name] = \
                CONSTRAINTS_PATH / ''.join([name[:-4], '.tdda'])
            self.raw_csv_paths[name] = RAW_CSV_PATH / name
            self.raw_dfs_dict[name] = \
                pd.read_csv(self.raw_csv_paths[name], header=0,
                            low_memory=False)
            # Make sure to .copy() to ensure raw_dfs_dict is not changed
            self.dfs_dict[name] = self.raw_dfs_dict[name].copy()

        # Create the final merged df and paths to csv and constraint files
        self.merged_df, self.dfs_dict = make_dataset.merge_dfs(self.dfs_dict)
        self.constraint_paths['wrangled_csv'] = \
            CONSTRAINTS_PATH / 'wrangled.tdda'
        self.raw_csv_paths['wrangled'] = \
            PROCESSED_PATH / 'wrangled_dataframe.csv'

    def tearDown(self):
        pass

    def test_input_csvs_meet_constraints(self):
        """Check that each csv in the /data/raw directory meets the constraints
        required.  This should be a layup - the files and the constraints
        should not have changed."""
        self.failures = {}
        for name in self.filenames:
            df = self.raw_dfs_dict[name]
            v = verify_df(df, self.constraint_paths[name])
            assert v.failures == 0

    def test_clean_googletrend(self):
        """Check that state_names gets cleaned without obvious errors."""
        df = make_dataset.clean_googletrend_csv(
            self.raw_dfs_dict['googletrend.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 13188
        assert df.date.dtype == '<M8[ns]'
        assert df.week_start.dtype == '<M8[ns]'
        assert df.state.dtype == 'O'
        assert df.trend.dtype == 'int64'
        assert df.notnull().all().all()

    def test_clean_state_names(self):
        """Check that state_names gets cleaned without obvious errors."""
        df = make_dataset.clean_other_dfs(self.raw_dfs_dict['state_names.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 16
        assert df.state_name.dtype == 'O'
        assert df.state.dtype == 'O'
        assert df.notnull().all().all()

    def test_clean_store_states(self):
        """Check that store_states gets cleaned without obvious errors."""
        df = make_dataset.clean_other_dfs(
            self.raw_dfs_dict['store_states.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 1115
        assert df.state.dtype == 'O'
        assert df.store.dtype == 'int64'
        assert df.notnull().all().all()

    def test_clean_store_csv(self):
        """Check that store.csv gets cleaned without obvious errors."""
        df = make_dataset.clean_store_csv(self.raw_dfs_dict['store.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 1115
        assert df.store.dtype == 'int64'
        assert df.store_type.dtype == 'O'
        assert df.assortment.dtype == 'O'
        assert df.competition_distance.dtype == 'float64'
        assert df.competition_open_since_month.dtype == 'float64'
        assert df.competition_open_since_year.dtype == 'float64'
        assert df.promo2.dtype == 'int64'
        assert df.promo2_since_week.dtype == 'float64'
        assert df.promo2_since_year.dtype == 'float64'
        assert df.promo_interval.dtype == 'O'
        assert df.notnull().all().all()

    def test_clean_train_csv(self):
        """Check that train.csv gets cleaned without obvious errors."""
        df = make_dataset.clean_other_dfs(self.raw_dfs_dict['train.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 1017209
        assert df.store.dtype == 'int64'
        assert df.day_of_week.dtype == 'int64'
        assert df.date.dtype == 'O'
        assert df.sales.dtype == 'int64'
        assert df.customers.dtype == 'int64'
        assert df.open.dtype == 'int64'
        assert df.promo.dtype == 'int64'
        assert df.state_holiday.dtype == 'O'
        assert df.school_holiday.dtype == 'int64'
        assert df.notnull().all().all()

    def test_clean_weather_csv(self):
        """Check that weather.csv gets cleaned without obvious errors."""
        df = make_dataset.clean_weather_csv(self.raw_dfs_dict['weather.csv'])
        assert list(df.columns) == \
            list(map(lambda x: x.lower(), list(df.columns)))
        assert len(df) == 15840
        assert df.file.dtype == 'O'
        assert df.date.dtype == 'O'
        assert df.max_temperature_c.dtype == 'int64'
        assert df.mean_temperature_c.dtype == 'int64'
        assert df.min_temperature_c.dtype == 'int64'
        assert df.dew_point_c.dtype == 'int64'
        assert df.mean_dew_point_c.dtype == 'int64'
        assert df.min_dew_point_c.dtype == 'int64'
        assert df.max_humidity.dtype == 'int64'
        assert df.mean_humidity.dtype == 'int64'
        assert df.min_humidity.dtype == 'int64'
        assert df.max_sea_level_pressureh_pa.dtype == 'int64'
        assert df.mean_sea_level_pressureh_pa.dtype == 'int64'
        assert df.min_sea_level_pressureh_pa.dtype == 'int64'
        # EDIT Remove this comment at the end if everything passes
        # Note this is goofy
        assert df.max_visibility_km.dtype == 'float64'
        assert df.mean_visibility_km.dtype == 'float64'
        assert df.min_visibility_km.dtype == 'float64'
        assert df.max_wind_speed_km_h.dtype == 'int64'
        assert df.mean_wind_speed_km_h.dtype == 'int64'
        assert df.max_gust_speed_km_h.dtype == 'float64'
        assert df.precipitationmm.dtype == 'float64'
        assert df.cloud_cover.dtype == 'float64'
        assert df.events.dtype == 'O'
        assert df.wind_dir_degrees.dtype == 'int64'
        assert df.notnull().all().all()

    @pytest.mark.thisone
    def test_merged_csv_has_google_states_names_store_train_and_weather(self):
        """Intermediate test to make sure dfs get merged smoothly"""
        assert 'trend' in self.merged_df.columns  # from googletrend.csv
        assert 'state' in self.merged_df.columns  # from state_names.csv
        assert 'store' in self.merged_df.columns  # from store_states.csv
        assert 'assortment' in self.merged_df.columns  # from store.csv
        assert 'sales' in self.merged_df.columns  # from train.csv
        assert 'precipitationmm' in self.merged_df.columns  # from weather.csv
        # There are 1,115 stores, and 942 days in the timeframe
        assert len(self.merged_df) == 942*1115  # = 1,050,330
        assert not self.merged_df.open.isnull().any()
        assert not self.merged_df.sales.isnull().any()
        assert not self.merged_df.customers.isnull().any()
        assert not self.merged_df.promo.isnull().any()
        assert not self.merged_df.school_holiday.isnull().any()
        assert not self.merged_df.state_holiday.isnull().any()
        assert (self.merged_df.day_of_week ==
                self.merged_df.date.dt.dayofweek).all()
        assert len(self.merged_df[(self.merged_df.customers == 0) &
                                  (self.merged_df.open == 1)]) == 0
        # Check that each ('store', 'date') tuple is unique
        assert (self.merged_df.groupby(['store', 'date']).size() == 1).all()
        # Check that each ('store', 'state') tuple has 942 instances -- again,
        # 942 days for each store in the dataset -- in other words, each
        # ('store', 'state') tuple is unique up to differing dates
        assert (self.merged_df.groupby(['store', 'state']).size() == 942).all()
        # If you're closed, you're not on promo
        assert len(self.merged_df[(self.merged_df.promo == 1) &
                                  (self.merged_df.open == 0)]) == 0

    # @pytest.mark.skip(reason='takes too long right now')
    def test_wrangled_csv_meets_constraints(self):
        """Check that the wrangled csv meets the constraints required."""
        wrangled_df = pd.read_csv(self.raw_csv_paths['wrangled'],
                                  low_memory=False)
        v = verify_df(wrangled_df, self.constraint_paths['wrangled_csv'])
        assert v.failures == 0


class Test_Wrangled_Csv(ReferenceTestCase):

    def setUp(self):
        self.RAW_CSV_PATH = PROJ_ROOT / 'data' / 'raw'
        self.REF_CSV_PATH = PROJ_ROOT / 'data' / 'processed' / \
            'wrangled_dataframe.csv'

    def tearDown(self):
        pass

    # @pytest.mark.skip(reason='not ready yet')
    def test_wrangled_csv_correct(self):
        """Check that the final constructed csv is an exact duplicate of the
        reference csv."""

        df, _ = make_dataset.merge_dfs(
            make_dataset.import_csvs(self.RAW_CSV_PATH,
                                     ignore_files=['test.csv',
                                                   'sample_submission.csv'],
                                     header=0,
                                     low_memory=False))

        ref_df = pd.read_csv(self.REF_CSV_PATH, header=0, low_memory=False)

        # Note that the file saves two columns as dtype '<M8[ns]', which is
        # a subset of datetime64. Below we're casting the reference data to
        # the more general datetime64 - apparently the reference test doesn't
        # consider these sufficiently close otherwise.
        # I'd rather convert the in-memory dataset to match the reference data
        # set, but for some reason that test failed.
        # See https://stackoverflow.com/questions/
        # 29206612/difference-between-data-type-datetime64ns-and-m8ns for more
        # detail.
        ref_df['date'] = pd.to_datetime(ref_df['date'])
        ref_df['week_start'] = pd.to_datetime(ref_df['week_start'])

        self.assertDataFramesEqual(df, ref_df)
# Example assertions if needed
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
