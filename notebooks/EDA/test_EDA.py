# import cauldron as cd
from cauldron.steptest import StepTestCase
# import os
# import pandas as pd
# from typing import List
# from unittest import mock


class TestNotebook_EDA_S01(StepTestCase):
    """ Test class containing step unit tests for the notebook step 1.
    """

    def test_it_works(self):
        """ Since this step does nothing except show the head of a merged
        dataframe, I'm just testing for success."""

        # Run the step
        result = self.run_step('S01.py')
        # Validate that the step ran successfully
        self.assertTrue(result.success)
