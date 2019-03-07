import os
import pytest
from tdda.referencetest import referencepytest, tag
from unittest import TestCase

import sys, os # NOQA
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + '/../')
from src.data import make_dataset # NOQA

"""NOTE ON SYSTEM TESTS: I originally started system tests in this file. As a
newb tester, in the course of writing my other tests I inter-mingled system
tests in there. 

In particular, since constraint tests and (especially) reference tests
encourage more systematic views, I got into the vein of writing more system-
oriented things in there.

It's not best practices to have the system tests in with the unit tests, but
since the system tests are already in those fileds and the whole test suite
takes ~2 minutes to run on a fast Google Cloud setup, I'm leaving this file
and moving on.
"""

class test_Make_Dataset(TestCase):

    def setUp(self):
        # Config filepaths
        self.PROJ_ROOT = os.path.abspath(os.pardir)
        self.directory = os.path.join(self.PROJ_ROOT, 'data', 'raw')

        self.df, _ = make_dataset.merge_dfs(
            make_dataset.import_csvs(self.directory,
                                     ignore_files=['test.csv',
                                                   'sample_submission.csv'],
                                     header=0, low_memory=False))

    def tearDown(self):
        pass

    def test_make_dataset_happy_path(self):
        """Happy path for make_dataset.py"""
        # User story: user runs src.make_dataset() on the current directory
        # and gets a fully functional dataset
        pass
