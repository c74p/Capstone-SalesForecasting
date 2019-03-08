Capstone-SalesForecasting
==============================

Sales Forecasting and Data Science project on the Rossmann (German retailer)
data set

Items in this README:
- How to run this repo
- A note on Cauldron vs Jupyter notebooks
- Project organization map

How to run this repo
------------
You'll need a CUDA-enabled environment to run the neural network model. You can
get directions for Google Cloud here: https://course.fast.ai/start_gcp.html
Note that there are lots of other environments mentioned at the upper left of
that link, but I found the Google Cloud environment to 'just work' with the
Fast.ai library in a way that others (e.g. AWS) did not.

With that out of the way:

**1)** Clone the repo (you'll need https://git-lfs.github.com/ if you don't
already have it).

**2)** In your conda/venv environment of choice:
pip install -r requirements.txt

**3)** Check the tests! 
cd tests; py.test It takes about two minutes on my
Google Cloud instance to run the tests.

**4)** Lots of options for use!

**a) Run the model** If you just want to run the model and see it in action:
go to notebooks/ models and run the Jupyter notebook

**b) Get a prediction** If you want to get a prediction: go to src/models and
run python predict_model.py --test_value=INT where 0 < INT < 40282

You can add '--context=True' if you want the actual sales value as well as the
predicted value.

Also, if you'd rather pass in a single-row .csv file to get a prediction, you
can do that:

python predict_model.py --new_value='../../data/interim/example_data_row.csv'

Of course, you can create and pass in other prediction data if you like.

**c) Update the model with new data** If you want to pass new data to the model
and update the model as appropriate, go to src/models and run
    python train_model.py --train_data=<train_data> --valid_data=<valid_data>
where <train_data> and <valid_data> are valid .csv files.

I don't assume that you have these sitting around; I'd recommend that you pass
'../../data/interim/train_valid_data.csv' as the training data and
'../../data/interim/fake_all_sales_as_forecast.csv' as the validation data.
There's also a 'fake_all_sales_double.csv' intended for use as an example
validation data set in the same directory.

A note on Cauldron vs Jupyter notebooks
------------
Although Jupyter notebooks are widely used in data science, they suffer from a
few deficiencies: they're not *just* real code, they maintain constant global
state, they are difficult for Github versioning, and their internal editor
leaves much to be desired.

As a result, for this project I tested working in Cauldron notebooks
(http://unnotebook.com). They address some of the shortcomings of Jupyter
notebooks and are more conducive to a production environment.

What does this mean for you?
- The notebook directories notebooks/data_wrangling and notebooks/EDA have a
  directory called 'html' - within that directory is an html file where you
  can see all the visuals. (The notebooks/models directory only contains a
  Jupyter notebook.)
- The code that creates the notebook is already in the directory, and is real
  Python code.
- If you prefer looking at the notebook itself, which includes both code and
  output in one place, Cauldron has a simple reader that requires a quick and
  simple install from http://unnotebook.com/reader-install/. Each /notebooks
  sub-directory has a .cauldron file that you can open.
- Each notebook directory in /notebooks/ has a .pdf file showing the output of
  the notebook, like a Jupyter notebook without the code. (The visuals in the
  pdf aren't quite as good as the other two options.)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Placeholder only.
    ├── README.md          <- The top-level README for using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Placeholder for a Sphinx project.
    │
    ├── models             <- Trained and serialized models, model predictions,
    │                         or model summaries.
    │
    ├── notebooks          <- Notebooks; see 'A Note on Cauldron vs Jupyter
    │                         notebooks' in this document for more details.
    │
    ├── references         <- Placeholder for explanatory materials.
    │
    ├── reports            <- Placeholder for reports.
    │
    ├── requirements.txt   <- The requirements file for reproducing the
    │                         environment (`pip freeze > requirements.txt`).
    │
    ├── setup.py           <- makes project pip installable (pip install -e .).
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn data into features for modeling.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained
    │   │   │                 models to make predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Placeholder.
    │
    ├── tests              <- Tests for /src. Note that due to Cauldron
    │                         requirements, tests for
    │                         Cauldron notebooks are in /notebooks/ 
    │                         (see http://unnotebook.com/docs/content/testing/
    │                         					      /intro/)
    │
    └── tox.ini            <- tox file with settings for running tox;
			      see tox.testrun.org.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
