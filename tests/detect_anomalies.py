import os
import pandas as pd
import sys

from tdda.constraints.pd.constraints import detect_df # NOQA

inpath = '../data/processed/wrangled_dataframe.csv'
constraint_path = ''.join(['../data/interim/constraints_initial_csvs/',
                           'wrangled_dataframe_constraints.tdda'])
outpath = '../data/interim/constraints_initial_csvs/wrangled_anomalies.tdda'

df = pd.read_csv(inpath, low_memory=False)
v = detect_df(df, constraint_path)
detection_df = v.detected()
if detection_df:
    print(detection_df.to_string())
    with open(outpath, 'w') as f:
        detection_df.to_csv(f)
else:
    print(f'No anomalies detected between {inpath} and {constraint_path}.')
    with open(outpath, 'w') as f:
        f.write(
            f'No anomalies detected between {inpath} and {constraint_path}.')

if os.path.exists(outpath):
    print('Written %s successfully.' % outpath)
    sys.exit(0)
else:
    print('Failed to write %s.' % outpath, file=sys.stderr)
    sys.exit(1)
