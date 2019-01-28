import os
from src.data import make_dataset
from unittest import TestCase


class test_Make_Dataset(TestCase):

    def setUp(self):
        # Config filepaths
        PROJ_ROOT = os.path.abspath(os.getcwd())
        directory = os.path.join(PROJ_ROOT, 'data', 'raw')

        self.df = make_dataset.import_csvs(directory)

    def tearDown(self):
        pass

    def test_Make_Dataset_happy_path(self):
        """Happy path for make_dataset.py"""
        # User story: user runs src.make_dataset() on the current directory
        # and gets a fully functional dataset, including:
        #   - number of rows is correct
        assert len(self.df) == 1050330
        self.fail('Finish the test!')
        #   - training data is in there
        #   - trend data is in there
        #   - weather data is in there
        #   - state (of the store location within Germany) data is in there
        #   - WHAT ELSE? EDIT
        # User gets a note that all files expected to be found, were found
        # add assertions here

    def test_Make_Dataset_highlight_errors(self):
        """If not on happy path, point out errors and present options"""
        # User story: user runs src.make_dataset() on a directory that's
        # missing a file and gets an error message, specifying:
        #   - Files expected to be found in there that were not found
        #   - Files expected to be found in there that *were* found
        #   - Note on what to do next - options to consider
        #   - WHAT ELSE? EDIT
        pass
        # !!!  EDIT THIS FUNCTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
