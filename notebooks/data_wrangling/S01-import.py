import cauldron as cd
import os
import sys

sys.path.append('../../src/data')
import make_dataset # NOQA, need the line above to get directories right

cd.display.markdown(
    """
    # Import dataframes

    Here we'll import the dataframes in the directory /data/raw and show the
    head of each. (Note that we don't use test.csv because we don't have any
    sales results for that file, so it's useless for our purposes.)
    """
    )

# Config filepaths
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
cd.shared.directory = os.path.join(PROJ_ROOT, 'data', 'raw')

# Pull in all csvs, ignoring test.csv and submission.csv because no
# target-variable data in those
dfs_dict = make_dataset.import_csvs(cd.shared.directory,
                                    ignore_files=['test.csv',
                                                  'sample_submission.csv'],
                                    header=0,
                                    low_memory=False)

for name in dfs_dict.keys():
    print(name + ':')
    cd.display.table(dfs_dict[name].head())

# Make sure to share dfs_dict in memory for next step
cd.shared.put('dfs_dict', dfs_dict)
