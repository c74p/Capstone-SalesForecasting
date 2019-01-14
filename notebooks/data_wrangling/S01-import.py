import cauldron as cd
import os
import sys
# sys.path.append('/run/media/chachi/USB30FD/Capstone-SalesForecasting/src/data')
sys.path.append('../../src/data')
# import src.make_dataset
import make_dataset

cd.display.markdown(
    """
    # Import dataframes

    Here we'll import the dataframes in the directory /data/raw and show the
    head of each.
    """
    )

# Config filepaths
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
directory = os.path.join(PROJ_ROOT, 'data', 'raw')

# Read csv files in the directory into shared cauldron memory and display them
# for file_name in os.listdir(directory):
    # Ignore file test.csv since we don't have target values for it
    # if file_name.endswith('.csv') and file_name != 'test.csv':
        # file_path = os.path.join(directory, file_name)
        # df = pd.read_csv(file_path, header=0, low_memory=False)

dfs = make_dataset.import_csvs(directory, ignore_files=['test.csv'], header=0,
                               low_memory=False)

for name, df in dfs:
    print(name + ':')
    cd.display.table(df.head())
    cd.shared.name = df
