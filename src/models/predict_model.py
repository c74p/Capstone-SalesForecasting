from fastai.basic_train import Learner, load_learner
from fastai.tabular import *
import math
import os
import pandas as pd
from pathlib import Path
import preprocess
import sys
from typing import Dict

DATA_PATH = Path('../../data/interim')
MODELS_PATH = Path('../../models/')

ERR_MSG = \
    ("\nUSAGE: \n\n OPTION 1: python3 predict_model.py --test_value=<INT>\n"
     "\t\twhere 0 <= INT <= 40281"
     "\n\n If the optional flag of '--context=True' is included, "
     "the actual sales value will be provided for comparison.\n\n OPTION 2: "
     "python3 predict_model.py --new_value=<FILENAME>\n\t\t where <FILENAME> "
     "is a .csv file in data/interim/ with a header and a single row of "
     "data.\n")

MAX_TEST_VALUE = 40281
MIN_TEST_VALUE = 0


def get_pred_single_val(data: pd.Series, path: Path) -> float:
    """Get a prediction for a single row of data.

    Input: a pd.Series for the data and the path for the model.
    Output: the predicted sales for that row of data.
    """
    # Get the right model to load                                               
    models = [file for file in os.listdir(path) if
              file.startswith('current_best')]
    best_model = sorted(models, reverse=True)[0]

    # Load the model and get the prediction
    #learn = load_learner(path/best_model)
    learn = load_learner(path=path, fname=best_model)

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


def predict(data_path=DATA_PATH, models_path=MODELS_PATH, **kwargs) -> str:
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

        # Get the test dataframe and process it
        test_df = pd.read_csv(data_path/'test_data.csv', low_memory=False)
        test_df = preprocess.preprocess(test_df)

        # Get our example row and get the prediction from it
        example = test_df.iloc[kwargs['test_value']]
        prediction = get_pred_single_val(example, models_path)

        if 'context' in kwargs and kwargs['context']:
            return ('The predicted value is ' + str(prediction) + ' and '
                    'the actual value is ' + str(example.sales) + '.')
        return str(prediction)

    if 'new_value' in kwargs:

        # Convert our series to a dataframe so we can process it
        df = pd.read_csv(data_path/kwargs['new_value'])
        df = preprocess.preprocess(df)

        prediction = get_pred_single_val(df.iloc[0], models_path)

        return str(prediction)


if __name__ == '__main__':
    args: Dict = {}
    try:
        for arg in sys.argv[1:]:
            arg, val = arg.strip('-').split('=')
            if arg == 'test_value':
                args[arg] = int(val)
            else:
                args[arg] = val

        result = predict(**args)
        print(result)

    except:
        print(ERR_MSG)
