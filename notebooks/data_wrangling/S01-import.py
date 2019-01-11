import cauldron as cd
import os
import pandas as pd

cd.display.markdown(
    """
    # Import dataframes

    Here we'll import the dataframes in the directory /data/raw and show the
    head of each.
    """
    )

# Read files in the directory into shared cauldron memory (remove them later!)
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
directory = os.path.join(PROJ_ROOT, 'data', 'raw')
for file_name in os.listdir(directory):
    # Ignoring file test.csv since we don't have target values for it
    if file_name.endswith('.csv') and file_name != 'test.csv':
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path, header=0, low_memory=False)
        print(file_name + ':')
        cd.display.table(df.head())
        cd.shared.prefix = df
