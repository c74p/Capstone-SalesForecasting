import cauldron as cd
from cauldron.steptest import StepTestCase
import os
import pandas as pd
from typing import List
from unittest import mock


class TestNotebook_Data_Wrangling_S01(StepTestCase):
    """ Test class containing step unit tests for the notebook.
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

    def test_with_mock(self):
        """Mock version of the previous test. Note that this version of the
        test by itself only ran a half-second faster than the full regular
        version above. I'm leaving this in, but also including the original
        test.  Seems that maybe Cauldron tests are just very slow."""
        # """All available csvs in the directory should be pulled"""

        with mock.patch('make_dataset.import_csvs', return_value={}):
            self.run_step('S01-import.py')

        files_to_pull: List[str] = ['googletrend.csv', 'sample_submission.csv',
                                    'state_names.csv', 'store.csv',
                                    'store_states.csv', 'train.csv',
                                    'weather.csv']

        directory = cd.shared.fetch('directory')

        files_mock_pulled: List[str] = [file for file in os.listdir(directory)
                                        if file.endswith('.csv') and
                                        file != 'test.csv']

        for file in files_to_pull:
            assert file in files_mock_pulled
