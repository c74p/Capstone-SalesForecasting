#from fastai.tabular import *
from hypothesis import given, example
from hypothesis.strategies import text
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock
from unittest.mock import patch

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
from src.models import preprocess, train_model # NOQA

# This is the test file for the src/models/train_model.py file.

#pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception

PROJ_ROOT = Path('..')


class Test_train_model(TestCase):
    """Test the steps needed to train the model, compare against the current
    best model, and update the 'current best' if appropriate.
    """

    def setUp(self):

        # Basic dataframe for testing
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
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14'],
                               [1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88, 64,
                                37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21, 13,
                                40.0, 0.0, 6.0, 'Rain', 290, 'c', 'b', 1270.0, 9.0,
                                2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14'],
                               [1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88, 64,
                                37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21, 13,
                                40.0, 0.0, 6.0, 'Rain', 290, 'c', 'c', 1270.0, 9.0,
                                2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14']])

        # Error message for an incorrect call to predict()
        self.ERR_MSG = \
        """USAGE: \n update-model=[<train>, <valid>] where <train> and <valid>
        are path names to train and validation sets."""

        # Data path for the test data
        # I recognize these file names and PATH_NAMES are kind of contradictory
        self.TRAIN_DATA_PATH = Path('../data/interim/train_valid_data.csv')
        self.VALID_DATA_PATH = Path('../data/interim/test_data.csv')
        self.valid_df = pd.read_csv(self.VALID_DATA_PATH, low_memory=False)

        # Path for the models
        self.MODEL_PATH = Path('../models')

    def tearDown(self):
        pass

    def test_rmspe(self):
        """Here are some hard-coded values, rather than generated, for speed"""
        assert train_model.rmspe(np.array([0]), np.array([1])) == 1
        assert train_model.rmspe(np.array([0.25]), np.array([1])) == 0.75
        assert train_model.rmspe(np.array([0.5]), np.array([1])) == 0.5
        assert train_model.rmspe(np.array([0.75]), np.array([1])) == 0.25
        assert train_model.rmspe(np.array([1]), np.array([0])) == np.inf

    def test_get_pred_new_data_old_model(self):
        """Dumb reference test checking versuse the actual. This will need to
        be updated later when the model gets updated, or it will likely fail.
        """
        # First test the actual results (will have to throw this out or update
        # later when the test gets updated)
        res = train_model.get_pred_new_data_old_model(self.valid_df,
                                                      self.MODEL_PATH)
        assert abs(res - 0.048635791657389196) < 0.001


    #@patch('fastai.tabular.load_learner')
    @patch('src.models.train_model.Learner.get_preds', return_value=(np.array([1,1,1]),1))
    @patch('src.models.train_model.rmspe', return_value=1, side_effect=print('rmspe'))
    def test_get_pred_new_data_old_model2(self, mock_rmspe, mock_gp):
        """The old model should be pulled in, and the accuracy of the old
        model gauged by the new data vs actuals.
        """
        res = train_model.get_pred_new_data_old_model(self.valid_df,
                                                      self.MODEL_PATH)
        #assert mock_load_learner.called
        assert mock_gp.called
        assert mock_rmspe.called

#def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
        #"""Nothing should be returned from an empty directory"""
        #with mock.patch('os.listdir', return_value=self.fake_empty_files):
            #with mock.patch('pandas.read_csv',
                            #side_effect=self.fake_empty_read):
                #read = make_dataset.import_csvs('bogus_dir')
                #assert read == {}
