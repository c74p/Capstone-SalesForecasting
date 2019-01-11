import os
from src.data import make_dataset

# User story: user runs src.make_dataset() on the current directory and gets a
# fully functional dataset, including:
#   - number of rows is correct
#   - training data is in there
#   - trend data is in there
#   - weather data is in there
#   - state (of the store location within Germany) data is in there
#   - WHAT ELSE? EDIT
# User gets a note that all files expected to be found, were found

# Config filepaths
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
directory = os.path.join(PROJ_ROOT, 'data', 'raw')

df = make_dataset.import_csvs(directory)

assert len(df) == 50000
# add assertions here


# User story: user runs src.make_dataset() on a directory that's missing a file
# and gets an error message, specifying:
#   - Files expected to be found in there that were not found
#   - Files expected to be found in there that *were* found
#   - Note on what to do next - options to consider
#   - WHAT ELSE? EDIT
