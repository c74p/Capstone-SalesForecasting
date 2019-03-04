from fastai.tabular import *
from hypothesis import given, example
from hypothesis.strategies import text
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

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
        """The old model should be pulled in, the new data preprocessed, and
        the accuracy of the old model gauged by the new data vs actuals.
        """
        # First test the actual results (will have to throw this out or update
        # later when the test gets updated)
        res = train_model.get_pred_new_data_old_model(self.valid_df,
                                                      self.MODEL_PATH)
        assert abs(res - 0.048635791657389196) < 0.001

#def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
        #"""Nothing should be returned from an empty directory"""
        #with mock.patch('os.listdir', return_value=self.fake_empty_files):
            #with mock.patch('pandas.read_csv',
                            #side_effect=self.fake_empty_read):
                #read = make_dataset.import_csvs('bogus_dir')
                #assert read == {}
