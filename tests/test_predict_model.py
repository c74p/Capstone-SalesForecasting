from hypothesis import given, example
from hypothesis.strategies import text
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
sys.path.insert(1, THIS_PATH + '/../src/models/')
from src.models import predict_model, preprocess  # NOQA

# This is the test file for the src/models/predict_model.py file.

# pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt=Exception

PROJ_ROOT = Path('..')


class TestPredictModel(TestCase):
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
                         data=[[1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, 'Rain', 290, 'c', 'a',
                                1270.0, 9.0, 2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 0.0, 85, '2015-06-14']])

        # Error message for an incorrect call to predict()
        #self.err_msg = \
        #    """USAGE: \n Option 1: -test_value=<INT> where 0 <= INT <= 40282
        #    \n An optional flag of '-context' will also
        #    provide the actual value for comparison.\n Option 2:
        #    new_value=<FILENAME> where <FILENAME> is a .csv file
        #    in data/interim/ with a header and a single row of
        #    data."""

        self.err_msg = \
            ("\nUSAGE: \n\n OPTION 1: python3 predict_model.py "
             "--test_value=<INT>\n\t\twhere 0 <= INT <= 40281"
             "\n\n If the optional flag of '--context=True' is included, "
             "the actual sales value will be provided for comparison.\n\n "
             "OPTION 2: python3 predict_model.py --new_value=<FILENAME>\n\t\t "
             "where <FILENAME> is a .csv file in data/interim/ with a header "
             "and a single row of data.\n")
        # Data path for the test data
        self.data_path = Path('../data/interim/')
        self.test_data_path = Path('../data/interim/test_data.csv')

        # Path for the models
        self.models_path = Path('../models')

    def tearDown(self):
        pass

    def test_no_parameters_gets_error_message(self):
        """The 'predict' option should either be called with the 'test_value=
        <INT>' flag, or with the 'new_value=<FILENAME>' flag"""
        res = predict_model.predict()
        assert res == self.err_msg

    def test_both_parameters_gets_error_message(self):
        """The 'predict' option should either be called with the 'test_value=
        <INT>' flag, or with the 'new_value=<FILENAME>' flag - but not both"""
        res = predict_model.predict(test_value=42, new_value='fake_file.csv')
        assert res == self.err_msg

    def test_test_value_oob_gets_error_message(self):
        """The 'predict' option with the 'test_value=<INT>' flag requires that
        <INT> be between 0 and 41608."""
        res = predict_model.predict(test_value=-1)
        assert res == self.err_msg
        res = predict_model.predict(test_value=40283)
        assert res == self.err_msg

    def test_correct_test_value_call_works(self):  # NOQA
        """Dumb reference test: calling predict with test_value=0 should
        result in an answer of exp(8.3232). TODO: consider mocking the
        calls to preprocess.preprocess() and load_learner(); might be
        faster."""
        res = predict_model.predict(data_path=self.data_path,
                                    models_path=self.models_path,
                                    test_value=0)
        assert abs(float(res) - 4198.543914975132) < 0.01

    def test_correct_test_value_call_with_context_works(self):
        """Dumb reference test: calling predict with test_value=0 and
        context=True should result in an answer of exp(8.3232), with
        appropriate context. TODO: consider mocking out the calls to
        preprocess.preprocess() and load_learner(); might be faster."""
        res = predict_model.predict(data_path=self.data_path,
                                    models_path=self.models_path,
                                    test_value=0, context=True)
        assert res == ('The predicted value is 4198.543914975132 and the '
                       'actual value is 4097.0.')

    def test_correct_new_value_call_works(self):
        """Dumb reference test: calling predict with new_value=<example>
        should result in an answer of exp(8.3232). TODO: consider mocking
        out the calls to preprocess.preprocess() and load_learner(); might be
        faster."""
        # Fake a test_value from the existing pre-made dataframe
        # Use .iloc[0] to make sure we're using a Series per expectations
        res = predict_model.predict(data_path=self.data_path,
                                    models_path=self.models_path,
                                    new_value='example_data_row.csv')
        assert abs(float(res) - 4198.543914975132) < 0.01
