from datetime import datetime
from fastai import callbacks
from fastai.metrics import exp_rmspe
from fastai.tabular import DatasetType, TabularList
from functools import partial
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
        #assert mock_preprocess.called
        mock_preprocess.assert_called_with(self.df)
        #assert mock_load_learner.called
        mock_load_learner.assert_called
        #mock_load_learner.assert_called_with(self.model_path, test=TabularList.
        #                                    from_df(mock_preprocess, path=
        #                                            self.model_path))

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
        mock_get_preds.assert_called_with(ds_type=DatasetType.Test)
        mock_rmspe.assert_called

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
        mock_preprocess.assert_called()
        mock_gather_args.assert_called()
        mock_tabular_list.from_df.assert_called_with(mock_preprocess(),
            path=self.model_path,
            procs=mock_gather_args()['procs'],
            cat_names=mock_gather_args()['cat_names'],
            cont_names=mock_gather_args()['cont_names'])
        mock_tabular_learner.assert_called()
        #TODO either get this to work or remove it
        #mock_tabular_learner.assert_called_with(
        #    mock_tabular_list.from_df().split_by_idx().label_from_df().
        #    databunch(),
        #    layers=[100, 100],
        #    ps=[0.001, 0.01],
        #    emb_drop=0.01,
        #    metrics=exp_rmspe,
        #    y_range=None,
        #    callback_fns=[partial(callbacks.tracker.TrackerCallback,
        #                          monitor='exp_rmspe'),
        #                  partial(callbacks.tracker.EarlyStoppingCallback,
        #                          mode='min', monitor='exp_rmspe',
        #                          min_delta=0.01, patience=1),
        #                  partial(callbacks.tracker.SaveModelCallback,
        #                          monitor='exp_rmspe', mode='min',
        #                          every='improvement',
        #                          name=
        #                          datetime.now().strftime("%Y-%m-%d-%X"))])

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

    @pytest.mark.this
    def test_save_models(self):
        """save_models should save the second-best model to an appropriate
        file, and the best model to the current best model file.
        """
        assert False
# def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
# """Nothing should be returned from an empty directory"""
# with mock.patch('os.listdir', return_value=self.fake_empty_files):
# with mock.patch('pandas.read_csv',
# side_effect=self.fake_empty_read):
# read = make_dataset.import_csvs('bogus_dir')
# assert read == {}
