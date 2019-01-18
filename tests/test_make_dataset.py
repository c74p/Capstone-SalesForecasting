import pandas as pd
from src.data import make_dataset
from unittest import TestCase, mock
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import sampled_from


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

        # Config dataframe strategies for hypothesis testing
        # These may be used in the merge_csvs_* series of tests below
        self.goog_file_vals = ["Rossmann_DE", "Rossmann_DE_BE",
                               "Rossmann_DE_BW", "Rossmann_DE_BY",
                               "Rossmann_DE_HE", "Rossmann_DE_HH",
                               "Rossmann_DE_NI", "Rossmann_DE_NW",
                               "Rossmann_DE_RP", "Rossmann_DE_SH",
                               "Rossmann_DE_SL", "Rossmann_DE_SN",
                               "Rossmann_DE_ST", "Rossmann_DE_TH"
                               ]

        self.google_raw = data_frames([
            column('file', elements=sampled_from(self.goog_file_vals)),
            column('week', dtype=pd.Timestamp),
            column('trend', elements=sampled_from(range(0, 101)))])

        # file     2072 non-null object
        # week     2072 non-null object
        # trend    2072 non-null int64

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
                # assert 'ignore_files' not in kwargs
                mock_pandas.assert_called_with('bogus_dir/c.csv')

    @given(self.google_raw())
    def test_test_of_hypothesis(self, df):
        assert self.df['file'].dtype == object

    def test_merge_csvs(self):

        self.testorama = data_frames([
                                      column('A', dtype=int),
                                      column('B', dtype=float),
                                      column('C', dtype=object)]
                                     )
        pass
    # Want merge_all_csvs() to:
    # - merge all the csvs together into one, appropriately
    # What could go wrong?
    # - Not all the csvs could be there
    # - bunch of NaNs
    # - files could incorrectly come together

    def test_verify_csv_pull(self):
        pass
    # Want verify_csv_pull() to:
    # - Check the csv pull and send a message to user
    #   - Either pull was successful, or pull failed, why, and what to do next
    # What could go wrong?
    # - Not all the csvs could be there

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
