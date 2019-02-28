from fastai import tabular
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe for modeling. The data, along with the data
    from the gather_args() function will get passed to either the training or
    prediction method.

    Inputs: TODO
    Output: a dataframe to pass to .train() or .get_preds()
    """
    # Drop week_start since add_datepart() will do that
    df.drop('week_start', axis='columns', inplace=True)

    # Drop any sales == 0 since they'll mess up rmspe (div by zero)
    df = df[df.sales != 0]

    tabular.add_datepart(df, 'date', drop=True, time=False)

    return df


def gather_args(df: pd.DataFrame) -> Dict[str, Any]:
    """Gather the additional arguments needed to pass to .train() or
    .get_preds() in the fastai library.  The sole purpose of this function
    is to ensure a consistent set of args between training and prediction.

    Inputs: the dataframe of interest
    Output: a dictionary of arguments
    """
    args = {}
    args['path'] = Path('../../models')
    args['procs'] = [tabular.FillMissing, tabular.Categorify,
                     tabular.Normalize]
    args['cat_names'] = \
        ['assortment', 'events', 'promo_interval', 'state',
         'state_holiday', 'store_type', 'Day', 'Dayofweek', 'Is_month_end',
         'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
         'Is_year_end', 'Is_year_start', 'Month', 'Week', 'Year']
    args['cont_names'] = list(set(df.columns) - set(args['cat_names']))
    args['dep_var'] = 'sales'

    return args
