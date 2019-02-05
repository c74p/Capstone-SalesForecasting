import cauldron as cd
from cauldron.steptest import StepTestCase
import os
import pandas as pd
from typing import List
from unittest import mock


class TestNotebook_Data_Wrangling_S01(StepTestCase):
    """ Test class containing step unit tests for the notebook step 1.
    Note that tests on functions themselves are in tests folder - this is
    solely tests on the notebook itself. """

    def test_gets_all_csvs(self):
        """ Each of the csvs needed gets opened and placed in shared memory"""

        # Hard-coding here; it's literally what we need from this notebook
        files: List[str] = ['googletrend.csv', 'state_names.csv', 'store.csv',
                            'store_states.csv', 'train.csv', 'weather.csv']

        # List to keep track of csvs that got pulled
        csvs_pulled: List[str] = []

        # Run step 1 and pull out the shared dataframes dict
        self.run_step('S01-import.py')
        dfs_dict = cd.shared.fetch('dfs_dict')

        # Simply check that each file in files is pulled
        # Note that our /tests testing checks that the pull itself is correct
        # after merging
        # for file in files:
        for file in files:
            pulled: pd.DataFrame = dfs_dict[file]
            print(pulled)
            assert pulled is not None
            csvs_pulled.append(file)

        # not technically necessary but we already invested 5s in this
        assert csvs_pulled == files

class TestNotebook_Data_Wrangling_S02(StepTestCase):
    """ Test class containing step unit tests for the notebook step 2.
    """

    def test_gets_all_csvs(self):
        """ Since this step does nothing except show the head of a merged
        dataframe, I'm just testing for success."""

        # Run the step 
        # Only add this step if needed
        # self.run_step('S01-import.py')
        result = self.run_step('S02-merge.py')
        # Validate that the step ran successfully 
        self.assertTrue(result.success)

