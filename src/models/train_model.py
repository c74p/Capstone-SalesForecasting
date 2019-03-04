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

def get_pred_single_val(data: pd.Series, path: Path) -> float:
    """Get a prediction for a single row of data.

    Input: a pd.Series for the data and the path for the model.
    Output: the predicted sales for that row of data.
