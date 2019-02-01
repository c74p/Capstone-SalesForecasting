import pandas as pd
from pathlib import Path
from make_dataset import import_csvs, merge_csvs

PATH = Path('../../data/raw')
RAW_PATH = Path('../../data/processed/wrangled_dataframe.csv')

df = merge_csvs(import_csvs(PATH,
                            ignore_files=['test.csv', 'sample_submission.csv'],
                            header=0, low_memory=False))

ref_df = pd.read_csv(RAW_PATH, header=0, low_memory=False)

print(df[0].head())
print(df[0].columns)
print(len(df[0].columns))
print(ref_df.head())
print(ref_df.columns)
print(len(ref_df.columns))

for col in df[0].columns:
    if col not in ref_df.columns:
        print(col)
    # print(col)
    # print((df[0][col] == ref_df[col]).all())
