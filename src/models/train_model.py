from fastai.basic_train import Learner, load_learner
from fastai.tabular import DatasetType, TabularList
import math
import numpy as np
import pandas as pd
from pathlib import Path
from src.models import preprocess

DATA_PATH = Path('../data/interim')
MODELS_PATH = Path('../models/')

ERR_MSG = \
"""USAGE: \n Option 1: -test_value=<INT> where 0 <= INT <="""
"""41608\n An optional flag of '-context' will also"""
"""provide the actual value for comparison.\n Option 2: """
"""-new_value=<FILENAME> where <FILENAME> is a .csv file"""
"""in data/interim/ with a header and a single row of"""
"""data."""


MAX_TEST_VALUE=41608
MIN_TEST_VALUE=0

def rmspe(predicted: np.array, actual: np.array) -> float:
    """Root mean squared percentage error"""
    return np.sqrt((((actual - predicted)/actual)**2).sum()/len(actual))

def get_pred_new_data_old_model(valid_df: pd.DataFrame, path: Path) -> float:
    """Get a RSMPE score for predictions from the existing best model, with
    new data.

    Input: a pd.DataFrame for the validation data and the path for the model.
    Output: the root mean squared percentage error for the predicted sales.
    """
    valid_df = preprocess.preprocess(valid_df)
    learn = load_learner(MODELS_PATH,
                         test=TabularList.from_df(valid_df, path=MODELS_PATH))

    # get log predictions and compare to actual values
    log_preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    valid_preds = np.exp(np.array(log_preds.flatten()))
    valid_reals = valid_df.loc[valid_df.sales != 0, 'sales'].values
    return rmspe(valid_preds, valid_reals)
