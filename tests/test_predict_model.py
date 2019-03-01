from hypothesis import given, example
from hypothesis.strategies import text
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
from src.models import predict_model # NOQA

# This is the test file for the src/models/predict_model.py file.

#pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception

PROJ_ROOT = Path('..')


class Test_predict_model(TestCase):
    """Test the steps needed to make a prediction from the existing model."""

    def setUp(self):

        # Config dataframe example
        # This can be used to simulate a run with the '-test_value'
        # command-line option, or with the '-new_value' option
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
                         data=[[1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88, 64,
                                37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21, 13,
                                40.0, 0.0, 6.0, 'Rain', 290, 'c', 'a', 1270.0, 9.0,
                                2008.0, 0, 23.595446584938703,
                                2011.7635726795095, None, 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14']])

        # Error message for an incorrect call to predict()
        self.ERR_MSG = \
        """USAGE: \n Option 1: -test_value=<INT> where 0 <= INT <="""
        """41608\n An optional flag of '-context' will also"""
        """provide the actual value for comparison.\n Option 2: """
        """-new_value=<FILENAME> where <FILENAME> is a .csv file"""
        """in data/interim/ with a header and a single row of"""
        """data."""

        # Data path for the test data
        self.TEST_DATA_PATH = Path('../data/interim/test_data.csv')

        # Path for the models
        self.MODEL_PATH = Path('../models')

    def tearDown(self):
        pass

    def test_no_parameters_gets_error_message(self):
        """The 'predict' option should either be called with the 'test_value=
        <INT>' flag, or with the 'new_value=<FILENAME>' flag"""
        res = predict_model.predict()
        assert res == self.ERR_MSG

    def test_both_parameters_gets_error_message(self):
        """The 'predict' option should either be called with the 'test_value=
        <INT>' flag, or with the 'new_value=<FILENAME>' flag - but not both"""
        res = predict_model.predict(test_value=42, new_value='fake_file.csv')
        assert res == self.ERR_MSG

    def test_test_value_oob_gets_error_message(self):
        """The 'predict' option with the 'test_value=<INT>' flag requires that
        <INT> be between 0 and 41608."""
        res = predict_model.predict(test_value=-1)
        assert res == self.ERR_MSG
        res = predict_model.predict(test_value=41609)
        assert res == self.ERR_MSG

    def test_correct_test_value_call_works(self):
        """Calling predict with test_value=0 should call
        preprocess.preprocess() and load_learner() with the correct path, and
        should result in an answer of 4097."""
        res = predict_model.predict(test_value=0)
        assert res == "4097"

#def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
        #"""Nothing should be returned from an empty directory"""
        #with mock.patch('os.listdir', return_value=self.fake_empty_files):
            #with mock.patch('pandas.read_csv',
                            #side_effect=self.fake_empty_read):
                #read = make_dataset.import_csvs('bogus_dir')
                #assert read == {}
