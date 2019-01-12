import os
import pandas as pd
from src.data import make_dataset
from typing import Dict, Any
import unittest


class test_XXX_Test_Group_Name(unittest.TestCase):
    def setUp(self):
        # Config filepaths
        PROJ_ROOT = os.path.abspath(os.pardir)
        self.directory = os.path.join(PROJ_ROOT, 'data', 'raw')

        # Config kwargs for test_import_csvs
        self.kwargs: Dict[str, Any] = {'header': 0, 'low_memory': False}

        self.dict_of_dataframes = make_dataset.import_csvs(self.directory,
                                                           **self.kwargs)

    def tearDown(self):
        pass

    def test_XXX_Test_Name(self):
        pass
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

    def test_import_csvs_pulls_all_csvs(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.csv'):
                self.assertIn(filename, self.dict_of_dataframes)

    # Want import_csvs() to:
    # - Find and import all the csv files in the directory
    # What could go wrong?
    # - How do we know they're imported?
    # - What if there are no csv files in there?
    # - What if there are unused csvs in there? (We don't care)
    # - What if all the right csvs are not in there?


def test_merge_all_csvs():
    pass
    # Want merge_all_csvs() to:
    # - merge all the csvs together into one, appropriately
    # What could go wrong?
    # - Not all the csvs could be there


def test_verify_csv_pull():
    pass
    # Want verify_csv_pull() to:
    # - Check the csv pull and send a message to user
    #   - Either pull was successful, or pull failed, why, and what to do next
    # What could go wrong?
    # - Not all the csvs could be there
