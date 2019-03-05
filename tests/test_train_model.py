import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock
from unittest.mock import patch

import sys, os  # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
from src.models import preprocess, train_model # NOQA

# This is the test file for the src/models/train_model.py file.

# pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt=Exception

PROJ_ROOT = Path('..')


class TestTrainModel(TestCase):
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
                         data=[[1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, 'Rain', 290, 'c', 'a',
                                1270.0, 9.0, 2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14'],
                               [1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, 'Rain', 290, 'c', 'b',
                                1270.0, 9.0, 2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14'],
                               [1, 'HE', '2015-06-20', 17, 14, 11, 9, 7, 5, 88,
                                64, 37, 1021, 1020, 1018, 31.0, 11.0, 10.0, 21,
                                13, 40.0, 0.0, 6.0, 'Rain', 290, 'c', 'c',
                                1270.0, 9.0, 2008.0, 0, 23.595446584938703,
                                2011.7635726795095, 'None', 5, 4097.0, 494.0,
                                1.0, 0.0, 0, 1, 85, '2015-06-14']])

        # Error message for an incorrect call to predict()
        self.err_msg = \
            """USAGE: \n update-model=[<train>, <valid>] where <train> and
            <valid> are path names to train and validation sets."""

        # Data path for the test data
        # I recognize these file names and PATH_NAMES are kind of contradictory
        self.train_data_path = Path('../data/interim/train_valid_data.csv')
        self.valid_data_path = Path('../data/interim/test_data.csv')
        self.valid_df = pd.read_csv(self.valid_data_path, low_memory=False)

        # Path for the models
        self.model_path = Path('../models')

    def tearDown(self):
        pass

    def test_rmspe(self):  # NOQA
        """Here are some hard-coded values, rather than generated, for speed"""
        assert train_model.rmspe(np.array([0]), np.array([1])) == 1
        assert train_model.rmspe(np.array([0.25]), np.array([1])) == 0.75
        assert train_model.rmspe(np.array([0.5]), np.array([1])) == 0.5
        assert train_model.rmspe(np.array([0.75]), np.array([1])) == 0.25
        assert train_model.rmspe(np.array([1]), np.array([0])) == np.inf

    def test_get_pred_new_data_old_model(self):
        """Dumb reference test checking versus the actual. This will need to
        be updated later when the model gets updated, or it will likely fail.
        """
        # First test the actual results (will have to throw this out or update
        # later when the test gets updated)
        _, res = train_model.get_pred_new_data_old_model(self.valid_df,
                                                         self.model_path)
        assert abs(res - 0.048635791657389196) < 0.001

    @patch('src.models.preprocess.preprocess')
    @patch('src.models.train_model.load_learner')
    def test_get_pred_new_data_old_model_calls_pt1(self, mock_load_learner,
                                                   mock_preprocess):
        """Calling this in parts to avoid having to make a complicated mock.
        The old model should be pulled in, and the accuracy of the old
        model gauged by the new data vs actuals.
        """
        with self.assertRaises(ValueError):
            # It raises because we don't pass enough info to 'learn' to call
            # .get_preds() - that's why this test is split into two parts
            train_model.get_pred_new_data_old_model(self.df, self.model_path)
        assert mock_preprocess.called
        assert mock_load_learner.called

    @patch('src.models.train_model.Learner.get_preds',
           return_value=(np.array([1, 1, 1]), 1))
    @patch('src.models.train_model.rmspe')
    def test_get_pred_new_data_old_model_calls_pt2(self, mock_rmspe,
                                                   mock_get_preds):
        """Calling this in parts to avoid having to make a complicated mock.
        The old model should be pulled in, and the accuracy of the old
        model gauged by the new data vs actuals.
        """
        train_model.get_pred_new_data_old_model(self.df, self.model_path)
        assert mock_get_preds.called
        assert mock_rmspe.called

    @patch('src.models.preprocess.preprocess')
    @patch('src.models.preprocess.gather_args')
    @patch('src.models.train_model.TabularList')
    @patch('src.models.train_model.tabular_learner')
    def test_get_pred_new_model_calls_pt1(self, mock_tabular_learner,
                                          mock_tabular_list,
                                          mock_gather_args, mock_preprocess):
        """The data should be processed, the model run, and the new accuracy
        calculated.
        """
        with self.assertRaises(ValueError):
            # It raises because we don't pass enough info to 'learn' to call
            # .get_preds()
            train_model.get_new_model_and_pred(train_df=self.df[:2],
                                               valid_df=self.df[2:],
                                               path=self.model_path)
        assert mock_preprocess.called
        assert mock_gather_args.called
        assert mock_tabular_list.from_df.called
        assert mock_tabular_learner.called

    @pytest.mark.this
    def test_compare_rmspes(self):
        """compare_rmspes should compare the rmspes of the two models and return
        them in order.
        """
        winner = ('winner', 0.10)
        loser = ('loser', 0.50)

        # Note that in this test we're abusing the fact that Python has no real
        # type checker. In the real code, we're passing models, not strings.
        assert train_model.compare_rmspes(winner[0], winner[1], loser[0],
                                          loser[1]) == ['winner', 'loser']

# def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
# """Nothing should be returned from an empty directory"""
# with mock.patch('os.listdir', return_value=self.fake_empty_files):
# with mock.patch('pandas.read_csv',
# side_effect=self.fake_empty_read):
# read = make_dataset.import_csvs('bogus_dir')
# assert read == {}
