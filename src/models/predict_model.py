from fastai.basic_train import Learner, load_learner
from fastai.tabular import *
import math
import pandas as pd
from pathlib import Path
from src.models import preprocess

DATA_PATH = Path('../data/interim')
MODELS_PATH = Path('../models/')

ERR_MSG = \
            """USAGE: \n Option 1: -test_value=<INT> where 0 <= INT <= 40282
            \n An optional flag of '-context' will also
            provide the actual value for comparison.\n Option 2:
            new_value=<FILENAME> where <FILENAME> is a .csv file
            in data/interim/ with a header and a single row of
            data."""

MAX_TEST_VALUE = 40282
MIN_TEST_VALUE = 0


def get_pred_single_val(data: pd.Series, path: Path) -> float:
    """Get a prediction for a single row of data.

    Input: a pd.Series for the data and the path for the model.
    Output: the predicted sales for that row of data.
    """
    # Load the model and get the prediction
    learn = load_learner(path)

    log_pred_tens, _, _ = learn.predict(data)

    # The model returns a tensor (Float [x]) so we need to get x
    log_pred = log_pred_tens.data[0]

    # Also it gives log predictions, so we need to exp it
    prediction = math.exp(log_pred)

    return prediction


def errors_in_kwargs(**kwargs) -> bool:
    """Check for errors in the parameters passed to predict().

    Input: the **kwargs to predict()
    Output: True if errors, False if no errors
    """
    # Exactly one of ('test_value', 'new_value') must be in kwargs
    if ('test_value' not in kwargs and 'new_value' not in kwargs) or \
            ('test_value' in kwargs and 'new_value' in kwargs):
        return True

    if 'test_value' in kwargs:
        # Check boundaries
        if (kwargs['test_value'] < MIN_TEST_VALUE) or \
                (kwargs['test_value'] > MAX_TEST_VALUE):
            return True

    # If no errors, return False
    return False


def predict(**kwargs) -> str:
    """Get a prediction for a single value and return it to the screen.

    Input: see ERR_MSG above for required inputs and options.
    Output: if new_value is requested, the output (if not ERR_MSG) is a string
    containing simply the forecasted value. If test_value is requested by
    itself, the output is as for new_value. If test_value and context are
    requested, the output is a sentence explaining the test prediction and the
    actual value from the test dataset.
    """

    if errors_in_kwargs(**kwargs):
        return ERR_MSG

    if 'test_value' in kwargs:

        try:
            # Get the test dataframe and process it
            test_df = pd.read_csv(DATA_PATH/'test_data.csv', low_memory=False)
            test_df = preprocess.preprocess(test_df)

            # Get our example row and get the prediction from it
            example = test_df.iloc[kwargs['test_value']]
            prediction = get_pred_single_val(example, MODELS_PATH)

            if 'context' in kwargs and kwargs['context']:
                return ('The predicted value is ' + str(prediction) + ' and '
                        'the actual value is ' + str(example.sales) + '.')
            return str(prediction)

        except:
            return ERR_MSG

    if 'new_value' in kwargs:

        try:
            series = kwargs['new_value']

            # Convert our series to a dataframe so we can process it
            df = series.to_frame().T
            df = preprocess.preprocess(df)
            prediction = get_pred_single_val(df.iloc[0], MODELS_PATH)

            return str(prediction)

        except:
            return ERR_MSG
