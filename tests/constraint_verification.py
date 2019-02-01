import os
import pandas as pd
import sys

from tdda.constraints.pd.constraints import verify_df

inpath = '../data/processed/wrangled_dataframe.csv'
constraint_path = ''.join(['../data/interim/constraints_initial_csvs/',
                           'wrangled_dataframe_constraints.tdda'])
outpath = '../data/interim/constraints_initial_csvs/wrangled_verification.tdda'

df = pd.read_csv(inpath, low_memory=False)
v = verify_df(df, constraint_path)

print('Constraints passing: %d\n' % v.passes)
print('Constraints failing: %d\n' % v.failures)
if v.failures > 0:
    print('\n', str(v))
    print('\n', v.to_frame())

if v.failures == 0:
    with open(outpath, 'w') as f:
        f.write('Success!')
        f.write('\n')
        f.write(f'{inpath} meets all the constraints of {constraint_path}.')

else:
    with open(outpath, 'w') as f:
        f.write('There was at least one failure.')
        f.write('\n')
        v.to_frame().to_csv(outpath)

if os.path.exists(outpath):
    print('Written %s successfully.' % outpath)
    sys.exit(0)
else:
    print('Failed to write %s.' % outpath, file=sys.stderr)
    sys.exit(1)
