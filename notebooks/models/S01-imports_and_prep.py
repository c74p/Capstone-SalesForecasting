import cauldron as cd
import contextlib
from datetime import datetime
from fastai import *
from fastai.tabular import callbacks, DatasetType, exp_rmspe, load_learner
from fastai.tabular import FloatList, TabularList, tabular_learner
from functools import partial
import io
import matplotlib
matplotlib.use('TkAgg') # NOQA, need this line for plotting            
import matplotlib.image as mpimg                                       
import matplotlib.pyplot as plt                                        
plt.ion() # NOQA, need this line for plotting                          
import numpy as np
import pandas as pd
from pathlib import Path
import random
import seaborn as sns 

import os, sys # NOQA                                                  
sys.path.append('../../src/models')
import preprocess # NOQA, need the lines above to get directories right

# Prepare to show the training of the model
DATA_PATH = Path('../../data/interim')

# Note that this is all one dataframe, to be split into training
# and validation sets
df = pd.read_csv(DATA_PATH/'train_valid_data.csv', low_memory=False)

# Preprocess the dataframe and collect the args to pass to fastai library
df = preprocess.preprocess(df)
args = preprocess.get_args(df)

# Validation set is just 80% of the whole thing
valid_idx = range(int(0.8 * len(df)), len(df))

cd.display.markdown(
    """# Training a Neural Network in Fast.ai

    Here we do some light preprocessing (mostly consisting of using Fast.ai's
    tabular.add_datepart() function, that adds in things like is it the
    end of the quarter or not.

    Then we ask fast.ai to use it to create a 'databunch', which is a set of
    training/validation(/possibly test) data all in one place.

    Below we see the result of the preprocessing - note that categories remain
    categories in this representation, but they are 'embedded' into vectors
    that the neural network learns like any other. Also, the continuous
    variables have been normalized.

    We're using a fairly simple network, with two layers of 100 nodes each. In
    order to avoid overfitting, we'll randomly 'drop out' some data; in the
    first layer, our 'dropout rate' is 0.001, and in the second layer, our
    dropout rate is 0.01. We don't want to drop too many in the earlierlayer.
    We also have a dropout rate in our embedding matrix (which translates
    categories of data into vectors) of 0.01.
    """
)

# Create and show the databunch
data = (TabularList.from_df(df, path=args['path'], cat_names=['cat_names'],
                            cont_names=args['cont_names'], procs=args['procs'])
        .split_by_idx(valid_idx)
        .label_from_df(cols=args['dep_var'], label_cls=FloatList, log=True)
        .databunch())

cd.display.table(data.show_batch())


learn = tabular_learner(data, layers=[100, 100], ps=[0.001, 0.01],
    emb_drop=0.01, metrics=exp_rmspe, y_range=None,
    callback_fns=[partial(callbacks.tracker.TrackerCallback,
                          monitor='exp_rmspe'),
                  partial(callbacks.tracker.EarlyStoppingCallback, mode='min',
                          monitor='exp_rmspe', min_delta=0.01, patience=1),
                  partial(callbacks.tracker.SaveModelCallback,
                          monitor='exp_rmspe', mode='min',
                          every='improvement',
                          name=datetime.now().strftime("%Y-%m-%d-%X"))])

cd.display.markdown(
    """# Finding the learning rate
    Fast.ai comes with a utility called the 'learning rate finder'. Under the
    hood, fast.ai uses Leslie Smith's 'one-cycle policy' to modulate learning
    rates and momentum rates throughout a training epoch, but the library still
    needs to know where to set the maximum learning rate. So the idea is, look
    at the chart below, find where the loss is still improving, and use that.

    In our specific example, the chart appears to suggest that we use a
    learning rate of 1e00 (i.e. 1) - but in my tests, even a learning rate of
    1e-2 was too aggressive, and usually resulted in the model diverging
    completely.

    Thus, we'll use 1e-3, which worked well in most of my tests.
    """
)

learn.lr_find()
cd.display.pyplot(learn.recorder.plot())

cd.display.markdown(
    """From there, it's a matter of letting the model learn the best weights. I
    set an early-stopping callback on the model, so if it stops getting better,
    it will quit; otherwise, it will run for up to 10 runs through the data. (I
    haven't had it run for more than 5 runs through in my testing.
    """)

cd.display.pyplot(learn.fit_one_cycle(cyc_len=10, max_lr=1e-3))

cd.display.markdown(
    """With the training complete, let's take a look at the results. First, for
    some perspective, the results on the training set are below. We see some
    forecasts that we've missed, but overall our root-mean-squared-percentage
    error (rmspe) is good.

    In particular, the winners of this contest in 2015 had an rmspe around
    0.10 (lower is better). Our training results don't compare directly to the
    test results, but it's worth keeping an eye on to see how we're doing in
    our training vs. test runs.
    """
)


def draw_preds_vs_actuals(df: pd.DataFrame, set_name: str) -> None:
    """Draws a chart of the predicted vs actual sales.

    Inputs: a dataframe with columns entitled 'Predicted' and 'Actual', and
    the name of the set ('Training', 'Validation', or 'Test') for use in the
    chart title.
    Outputs: draws a chart. No return value.
    """

    fig, ax = plt.subplots()
    ax.scatter(df.Predicted, df.Actual)
    ax.plot(np.linspace(0, 40000, 5000), np.linspace(0, 40000, 5000),
            color='black')
    ax.set_title('Forecasts on the ' + set_name + ' set')
    ax.set_xlabel('Predicted sales')
    ax.set_ylabel('Actual sales')
    ax.set_xticklabels(['${:,.0f}'.format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
    cd.display.pyplot(fig)


def rmspe(predicted, actual):
    """Root mean squared percentage error"""
    return np.sqrt((((actual - predicted)/actual)**2).sum()/len(actual))

# Our predictions and actuals are in log form, so get them and exp them
train_log_preds, train_log_reals = learn.get_preds(ds_type=DatasetType.Train)
train_preds = np.exp(train_log_preds).flatten()
train_reals = np.exp(train_log_reals)

# Put in dataframe for easy access
train_sub = pd.DataFrame({'Predicted': train_preds, 'Actual': train_reals})

# Prepare and show chart
draw_preds_vs_actuals(train_sub, 'Training')

disp_string = 'RMSPE on the Training Set: '
disp_string += str(round(rmspe(train_sub.Predicted.values,
                         train_sub.Actual.values), 4))
cd.display.print(disp_string)

cd.display.markdown(
    """Now for the validation set, a similar story:
    """
)

# Our predictions and actuals are in log form, so get them and exp them
log_preds, log_reals = learn.get_preds(ds_type=DatasetType.Valid)
preds = np.exp(log_preds).flatten()
reals = np.exp(log_reals)

# Put in dataframe for easy access
sub = pd.DataFrame({'Predicted': preds, 'Actual': reals})

# Prepare and show chart
draw_preds_vs_actuals(sub, 'Validation')

disp_string = 'RMSPE on the Validation Set: '
disp_string += str(round(rmspe(sub.Predicted.values,
                         sub.Actual.values), 4))
cd.display.print(disp_string)

# Save the model and export the state of the processors for later use
# with the test data
model_name = datetime.now().strftime("%Y-%m-%d-%X")
learn.save(model_name)
learn.export(model_name)


# Prep the test data set and add it to the databunch
test_df = pd.read_csv(DATA_PATH/'test_data.csv', low_memory=False)

# Preprocess the dataframe and collect the args to pass to fastai library
test_df = preprocess.preprocess(test_df)
test_args = preprocess.get_args(test_df)

learn = load_learner(path=test_args['path'], fname=model_name,
                     test=TabularList.from_df(test_df, path=test_args['path']))

cd.display.markdown(
    """And finally what we really care about, the test set:
    """
)

# Our predictions and actuals are in log form, so get them and exp them
log_test_preds, log_test_reals = learn.get_preds(ds_type=DatasetType.Test)
test_preds = np.exp(log_test_preds).flatten()
test_reals = test_df.loc[test_df.sales != 0, 'sales'].values

# Put in dataframe for easy access
test_sub = pd.DataFrame({'Predicted': test_preds, 'Actual': test_reals})

# Prepare and show chart
draw_preds_vs_actuals(sub, 'Test')

disp_string = 'RMSPE on the Test Set: '
disp_string += str(round(rmspe(test_sub.Predicted.values,
                         test_sub.Actual.values), 4))
cd.display.print(disp_string)

cd.display.markdown(
    """That's pretty good. Just to get a handle on how good it is, here's a
    look at all of the forecastst that are off by more than 10%, and how many
    there are as a percentage of the total test set.
    """
)

wrongs = test_sub[
    np.abs((test_sub.Predicted - test_sub.Actual) / test_sub.Actual) > 0.1]

# Create and display the chart
fig, ax = plt.subplots()
ax.scatter(wrongs.Predicted, wrongs.Actual)
ax.plot(np.linspace(0, 40000, 5000), np.linspace(0, 40000, 5000),
        color='black')
ax.set_title('Forecasts on the Test Set that are Wrong by > 10%')
ax.set_xlabel('Predicted sales')
ax.set_ylabel('Actual sales')
ax.set_xticklabels(['${:,.0f}'.format(x) for x in ax.get_xticks()])
ax.set_yticklabels(['${:,.0f}'.format(x) for x in ax.get_yticks()])
cd.display.pyplot(fig)

print(f'Total number of forecasts off by over 10%: {len(wrongs)}')
print(f'Number of forecasts off by over 10%, as a percent of the test set: '
      '{len(wrongs)/len(test_sub)}')
