import os
import pandas as pd
import sys

from tdda.constraints.pd.constraints import discover_df

inpath = '../data/processed/wrangled_dataframe.csv'
outpath = ''.join(['../data/interim/constraints_initial_csvs/',
                   'wrangled_dataframe_constraints.tdda'])

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
