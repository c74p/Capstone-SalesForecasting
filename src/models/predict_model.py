from fastai.tabular import *
import math
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

def predict(**kwargs):
    """Get a prediction for a single value.

    Input: see ERR_MSG above for required inputs.
    Output: if new_value is requested, the output (if not ERR_MSG) is a string
    containing simply the forecasted value. If test_value is requested by
    itself, the output is as for new_value. If test_value and context are
    requested, the output is a sentence explaining the test prediction and the
    actual value from the test dataset.
    """
    # Exactly one of ('test_value', 'new_value') must be in kwargs
    if ('test_value' not in kwargs and 'new_value' not in kwargs) or \
        ('test_value' in kwargs and 'new_value' in kwargs):
        return ERR_MSG

    # If test_value, check boundaries and get test value
    if (kwargs['test_value'] < MIN_TEST_VALUE) or \
        (kwargs['test_value'] > MAX_TEST_VALUE):
        return ERR_MSG

    #try:
    # Get the test dataframe and process it
    test_df = pd.read_csv(DATA_PATH/'test_data.csv', low_memory=False)
    test_df = preprocess.preprocess(test_df)

    # Load the model and get the prediction
    learn = load_learner(MODELS_PATH)
    example = test_df.iloc[kwargs['test_value']]
    # TODO
    log_pred_tens, = learn.predict(example)

    # Remember the model gives log predictions, so we need to exp it
    prediction = math.exp(log_pred)

    return prediction

    #except:
        #return ERR_MSG
