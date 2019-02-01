import pandas as pd
from pathlib import Path
from unittest import TestCase, mock

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
# sys.path.insert(0, Path('..'))
from src.data import make_dataset # NOQA

from tdda.constraints.pd.constraints import verify_df # NOQA
from tdda.referencetest import ReferenceTestCase # NOQA

# This is the test file for the src/data/make_dataset.py file.

pd.set_option('mode.chained_assignment', 'raise')  # Chained assmt = Exception

PROJ_ROOT = Path('..')


class test_Import_Csvs(TestCase):

    def setUp(self):

        # Config file directory and read_csv return values for mock
        # These may be used in the import_csvs_* series of tests below
        self.fake_files = ['a.csv', 'b.csv', 'c.csv']
        self.fake_read = ['', '', '']

        # Config empty directory and read_csv return values for mock
        # These may be used in the import_csvs_* series of tests below
        self.fake_empty_files = []
        self.fake_empty_read = ['', '', '']

    def tearDown(self):
        pass

    def test_import_csvs_pulls_all_csvs(self):
        """All available csvs in the directory should be pulled"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv', side_effect=self.fake_read):
                read = make_dataset.import_csvs('bogus_dir')
                assert read == {k: v for k, v in
                                zip(self.fake_files, self.fake_read)}

    def test_import_csvs_pulls_no_csvs_from_empty_directory(self):
        """Nothing should be returned from an empty directory"""
        with mock.patch('os.listdir', return_value=self.fake_empty_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_empty_read):
                read = make_dataset.import_csvs('bogus_dir')
                assert read == {}

    def test_import_csvs_can_ignore_files(self):
        """A single 'ignore_files=' file should be ignored"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv', side_effect=self.fake_read):
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files='b.csv')
                assert read == {'a.csv': '', 'c.csv': ''}

    def test_import_csvs_can_ignore_files_as_list(self):
        """A list of 'ignore_files=' files should be ignored"""
        with mock.patch('os.listdir', return_value=self.fake_files):
            with mock.patch('pandas.read_csv',
                            side_effect=self.fake_read) as mock_pandas:
                read = make_dataset.import_csvs('bogus_dir',
                                                ignore_files=['b.csv'])
                assert read == {'a.csv': '', 'c.csv': ''}
                # how do I assert 'ignore_files' not in kwargs when the
                # function is called?
                mock_pandas.assert_called_with('bogus_dir/c.csv')


# Constraint testing on the initial files and on the generated 'wrangled' file

class test_Merge_Csvs(TestCase):

    def setUp(self):
        CONSTRAINTS_PATH = Path('../data/interim/constraints_initial_csvs')
        CSV_PATH = Path('../data/raw')
        PROCESSED_PATH = Path('../data/processed')
        self.constraint_paths = {}
        self.csv_paths = {}
        self.filenames = ['googletrend', 'state_names', 'store_states',
                          'store', 'train', 'weather']
        self.filenames = ['weather']
        for name in self.filenames:
            self.constraint_paths[name] = \
                CONSTRAINTS_PATH / ''.join([name, '.tdda'])
            self.csv_paths[name] = CSV_PATH / ''.join([name, '.csv'])
        self.constraint_paths['wrangled_csv'] = \
            CONSTRAINTS_PATH / 'wrangled.tdda'
        self.csv_paths['wrangled'] = PROCESSED_PATH / 'wrangled_dataframe.csv'

    def tearDown(self):
        pass

    def test_input_csvs_meet_constraints(self):
        for name in self.filenames:
            df = pd.read_csv(self.csv_paths[name], header=0, low_memory=False)
            v = verify_df(df, self.constraint_paths[name])
            assert v.failures == 0

    def test_wrangled_csv_meets_constraints(self):
        wrangled_df = pd.read_csv(self.csv_paths['wrangled'], low_memory=False)
        v = verify_df(wrangled_df, self.constraint_paths['wrangled_csv'])
        assert v.failures == 0


class Test_Wrangled_Csv(ReferenceTestCase):

    def setUp(self):
        self.RAW_CSV_PATH = PROJ_ROOT / 'data' / 'raw'
        self.REF_CSV_PATH = PROJ_ROOT / 'data' / 'processed' / \
            'wrangled_dataframe.csv'

    def tearDown(self):
        pass

    def test_wrangled_csv_correct(self):

        df = make_dataset.merge_csvs(
            make_dataset.import_csvs(self.RAW_CSV_PATH,
                                     ignore_files=['test.csv',
                                                   'sample_submission.csv'],
                                     header=0,
                                     low_memory=False))

        ref_df = pd.read_csv(self.REF_CSV_PATH, header=0, low_memory=False)

        self.assertDataFramesEqual(df, ref_df)

        # Note that the path for the reference dataframe is specified in the
        # root directory in conftest.py
        # self.assertDataFrameCorrect(df, self.REF_CSV_PATH, header=0,
        #                            low_memory=False)


# Example assertions if needed
# raise NotImplementedError('Insert test code here.')
#  Examples:
# self.assertEqual(fp.readline(), 'This is a test')
# self.assertFalse(os.path.exists('a'))
# self.assertTrue(os.path.exists('a'))
# self.assertTrue('already a backup server' in c.stderr)
# self.assertIn('fun', 'disfunctional')
# self.assertNotIn('crazy', 'disfunctional')
# with self.assertRaises(Exception):
#   raise Exception('test')
#
# Unconditionally fail, e.g. in a try block that should raise
# self.fail('Exception was not raised')
