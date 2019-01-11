Capstone-SalesForecasting
==============================

Sales Forecasting and Data Science project on the Rossmann (German retailer) data set

Items in this README:
- A Note on Cauldron vs Jupyter Notebooks
- Project Organization map

A Note on Cauldron vs Jupyter Notebooks
------------
Although Jupyter notebooks are widely used in data science, they suffer from a
few deficiencies: they're not *just* real code, they maintain constant global
state, they are difficult for Github versioning, and their internal editor
leaves much to be desired.

As a result, for this project I'm working in Cauldron notebooks
(http://unnotebook.com). They address some of the shortcomings of Jupyter
notebooks and are more conducive to a production environment.

What does this mean for you?
- Each notebook directory in /notebooks/ has a .pdf file showing the output of
  the notebook, like a Jupyter notebook without the code. (It's actually easier
  to follow the narrative in this format anyway).
- The code that creates the notebook is already in the directory, and is real
  Python code with real testing.
- If you prefer looking at the notebook itself, which includes both code and
  output in one place, Cauldron has a simple reader that requires a quick and
  simple install from http://unnotebook.com/reader-install/. Each /notebooks
  sub-directory has a .cauldron file that you can open.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests              <- Tests for /src. Note that due to Cauldron requirements, tests for 
    │                         Cauldron notebooks are in /notebooks/ 
    │                         (see http://unnotebook.com/docs/content/testing/intro/)
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
