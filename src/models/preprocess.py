from fastai import tabular
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

def preprocess(inp_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe for modeling. The data, along with the data
    from the gather_args() function will get passed to either the training or
    prediction method.

    Inputs: a raw dataframe
    Output: a processed dataframe to pass to .train() or .get_preds()
    """

    df = inp_df.copy()

    # Sort by date since we have a timeseries
    df.sort_values(by=['date', 'store'], inplace=True)

    # Drop week_start and day_of_week since add_datepart() will do that
    df.drop('week_start', axis='columns', inplace=True)
    df.drop('day_of_week', axis='columns', inplace=True)

    # If our whole df has sales == 0, it must be a single-row df used for a
    # single prediction, so just take the first row
    if (df.sales == 0).all():
        df = df.iloc[0]
    else:
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
