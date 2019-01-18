import os
import sys
import pandas as pd

from tdda.constraints.pd.constraints import discover_df

inpath = './train.csv'
outpath = '../interim/constraints_initial_csvs/train_constraints.tdda'

df = pd.read_csv(inpath, low_memory=False)
constraints = discover_df(df)

with open(outpath, 'w') as f:
    f.write(constraints.to_json())

if os.path.exists(outpath):
    print('Written %s successfully.' % outpath)
    sys.exit(0)
else:
    print('Failed to write %s.' % outpath, file=sys.stderr)
    sys.exit(1)
