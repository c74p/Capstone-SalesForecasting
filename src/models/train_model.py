from datetime import datetime
from fastai import callbacks
from fastai.basic_train import Learner, load_learner
from fastai.metrics import exp_rmspe
from fastai.tabular import DatasetType, FloatList, tabular_learner, TabularList
from fastai.train import fit_one_cycle
from functools import partial
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import preprocess
import sys
from typing import Any, Dict, List, Tuple

DATA_PATH = Path('../../data/interim')
MODELS_PATH = Path('../../models/')

ERR_MSG = \
    ("USAGE: \n\tpython3 train_model.py --train_data=<train_data> "
     "--valid_data=<valid_data>\n\n"
     "\twhere <train_data> and <valid_data> are valid .csv files in\n"
     "\tthe directory 'data/interim/'\n")

MAX_TEST_VALUE = 41608
MIN_TEST_VALUE = 0


def rmspe(predicted: np.array, actual: np.array) -> float:
    """Root mean squared percentage error"""
    return np.sqrt((((actual - predicted)/actual)**2).sum()/len(actual))


def get_pred_new_data_old_model(valid_df: pd.DataFrame,
                                path: Path = MODELS_PATH) -> Tuple[Learner,
                                                                   float]:
    """Get a RSMPE score for predictions from the existing best model, with
    new data.

    Input: a pd.DataFrame for the validation data and the path for the model.
    Output: the model ready to save, and the root mean squared percentage error
    for the predicted sales. (If this model is second-best, we'll still want
    to save it to a different file for record-keeping purposes.)
    """
    valid_df = preprocess.preprocess(valid_df)

    # Get the right model to load
    models = [file for file in os.listdir(path) if
              file.startswith('current_best')]
    best_model = sorted(models, reverse=True)[0]
    learn = load_learner(path=path, fname=best_model,
                         test=TabularList.from_df(valid_df, path=path))

    # get log predictions and compare to actual values
    log_preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    valid_preds = np.exp(np.array(log_preds.flatten()))
    valid_reals = valid_df.loc[valid_df.sales != 0, 'sales'].values
    new_rmspe = rmspe(valid_preds, valid_reals)
    return (learn, new_rmspe)


def get_new_model_and_pred(train: pd.DataFrame, valid: pd.DataFrame,
                           path: Path = MODELS_PATH) -> Tuple[Learner, float]:
    """Take new train and validation dataframes, re-run the model, and return
    the model and its root mean squared percentage error.

    Input: the train dataframe, the validation dataframe, and the path for the
    models to be saved.
    Output: the model (ready to save if better than the old one) and its rmspe.
    """

    # Sort the train/valid sets and stick em together
    train.sort_index(inplace=True)
    valid.sort_index(inplace=True)
    df = train.append(valid).copy()

    # We'll need to know how many items in our validation set later
    n_valid = len(valid[valid.sales != 0])

    # Preprocessing
    df = preprocess.preprocess(df)
    inner_args = preprocess.gather_args(df)

    # Create a databunch by starting with a TabularList and applying the usual
    # transformations
    data = (TabularList.from_df(df, path=path,
                                cat_names=inner_args['cat_names'],
                                cont_names=inner_args['cont_names'],
                                procs=inner_args['procs']))

    n_items = len(data.items)

    # Since we sorted by index and appended, our validation set is just the
    # n_valid highest items in our list
    data = data.split_by_valid_func(lambda i: i >= n_items - n_valid)
    data = data.label_from_df(cols=inner_args['dep_var'], label_cls=FloatList,
                              log=True)
    data = data.databunch()

    # Create a learner
    # Let's construct the learner from scratch here, in case we want to change
    # the architecture later (we can and should - this is very basic)
    learn = tabular_learner(data, layers=[100, 100], ps=[0.001, 0.01],
                            emb_drop=0.01, metrics=exp_rmspe, y_range=None,
                            callback_fns=[partial(callbacks.tracker.
                                                  TrackerCallback,
                                                  monitor='exp_rmspe'),
                                          partial(callbacks.tracker.
                                                  EarlyStoppingCallback,
                                                  mode='min',
                                                  monitor='exp_rmspe',
                                                  min_delta=0.01, patience=1),
                                          partial(callbacks.tracker.
                                                  SaveModelCallback,
                                                  monitor='exp_rmspe',
                                                  mode='min',
                                                  every='improvement',
                                                  name=datetime.now().
                                                  strftime("%Y-%m-%d-%X"))])

    # Since repeated model runs showed us that 1e-3 was a good maximum learning
    # rate for this model and since we're doing a no-human-intervention run,
    # we'll use 1e-3 for this model. While this model is in place, we can run
    # some offline tests as needed to see whether the maximum learning rate
    # should be changed, but in most cases the 1e-3 is probably good, even if
    # the model changes (again, we can test offline and update if needed).

    # Also, since we have the early-stopping callback with the save-model
    # callback set to 'every=improvement', we'll run 10 cycles even though we
    # probably won't need nearly that many
    learn.fit_one_cycle(cyc_len=10, max_lr=1e-3)

    # Get our predictions from the model and calculate rmspe
    log_preds, log_reals = learn.get_preds(ds_type=DatasetType.Valid)
    preds = np.exp(log_preds).flatten()
    reals = np.exp(log_reals)
    new_rmspe = rmspe(preds, reals)
    return (learn, new_rmspe)


def compare_rmspes(model0, rmspe0, model1, rmspe1):
    """Compare the rmspes of the two models and return them in order.

    Input: the two models and their rmspes.
    Output: A list of the the models to be saved, in order best to worst.
    """
    if rmspe0 <= rmspe1:
        return (model0, model1)
    return (model1, model0)


def save_models(winner: Learner, loser: Learner) -> None:
    """Saves ('exports') the models with descriptions for later use.

    Input: the winning and losing models (not the names, the models themselves
    -- probably called something like 'learner').
    Output: no return value.  Exports the models to the path.
    """
    winner_save_string = \
        'current_best-' + datetime.now().strftime("%Y-%m-%d-%X")
    loser_save_string = \
        'second_best-' + datetime.now().strftime("%Y-%m-%d-%X")

    winner.export(fname=winner_save_string)
    loser.export(fname=loser_save_string)


if __name__ == '__main__':
    args: Dict = {}

    try:
        for arg in sys.argv[1:]:
            arg, val = arg.strip('-').split('=')
            args[arg] = val

        train_df = pd.read_csv(DATA_PATH/args['train_data'], low_memory=False)
        valid_df = pd.read_csv(DATA_PATH/args['valid_data'], low_memory=False)
        model0, rmspe0 = get_pred_new_data_old_model(valid_df=valid_df)
        model1, rmspe1 = get_new_model_and_pred(train=train_df,
                                                valid=valid_df)
        models_to_save = compare_rmspes(model0, rmspe0, model1, rmspe1)
        save_models(models_to_save[0], models_to_save[1])
        if models_to_save[0] == model0:
            print('Retaining existing model, as it fit the validation data '
                  'better. You can find it in your model path with the name '
                  '"current-best-" and the most recent date.')
        else:
            print('Trained a new model that replaced the existing model. '
                  'You can find it in your model path with the name '
                  '"current-best-" and the most recent date.')

    except:
        print(ERR_MSG)
