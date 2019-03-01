from fastai.tabular import *
from hypothesis import given, example
from hypothesis.strategies import text
from io import StringIO
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

import sys, os # NOQA
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_PATH + '/../')
from src.models import preprocess # NOQA

from tdda.constraints.pd.constraints import verify_df # NOQA
from tdda.referencetest import ReferenceTestCase # NOQA

# This is the test file for the src/data/make_dataset.py file.

# Normally I want this on, but fastai.tabular.add_datepart does something that
# causes this to be an error - so I have to turn it off here
# pd.set_option('mode.chained_assignment', 'raise')
# Chained assmt = Exception

PROJ_ROOT = Path('..')


class Test_Preprocessing(TestCase):
    """Test the preprocessing steps in src/models/preprocess.py"""

    def setUp(self):

        # Use the actual test_data.csv file for testing
        # Come back later and optimize if needed
        CSV_PATH = Path('../data/interim/test_data.csv')
        self.df = pd.read_csv(CSV_PATH, low_memory=False)

    def tearDown(self):
        pass

    def test_preprocessing(self):
        """Test the preprocess() function from src/models/preprocess.py"""
        df = preprocess.preprocess(self.df)
        assert (df == df.sort_values(by=['Elapsed', 'store'])).all().all()
        assert 'week_start' not in df.columns
        assert (len(df[df.sales == 0])) == 0
        for col in ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                    'Is_month_end', 'Is_month_start', 'Is_quarter_end',
                    'Is_quarter_start', 'Is_year_end', 'Is_year_start']:
            assert col in df.columns

    def test_gather_args(self):
        """Test the gather_args function from src/models/preprocess.py"""
        args = preprocess.gather_args(preprocess.preprocess(self.df))

        assert args['path'] == Path('../../models')
        assert args['cat_names'] == \
            ['assortment', 'events', 'promo_interval', 'state',
             'state_holiday', 'store_type', 'Day', 'Dayofweek', 'Is_month_end',
             'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
             'Is_year_end', 'Is_year_start', 'Month', 'Week', 'Year']
        assert set(args['cont_names']) == \
            set(['cloud_cover', 'competition_distance', 'max_visibility_km',
                 'min_dew_point_c', 'Elapsed', 'precipitationmm',
                 'competition_open_since_month', 'max_temperature_c',
                 'dew_point_c', 'mean_temperature_c', 'promo',
                 'mean_dew_point_c', 'wind_dir_degrees', 'open', 'day_of_week',
                 'Dayofyear', 'min_humidity', 'customers', 'promo2',
                 'max_wind_speed_km_h', 'sales', 'mean_sea_level_pressureh_pa',
                 'min_sea_level_pressureh_pa', 'school_holiday',
                 'min_visibility_km', 'promo2_since_year',
                 'competition_open_since_year', 'mean_visibility_km',
                 'max_humidity', 'mean_wind_speed_km_h', 'trend',
                 'max_sea_level_pressureh_pa', 'store', 'mean_humidity',
                 'min_temperature_c', 'max_gust_speed_km_h',
                 'promo2_since_week'])
        assert args['procs'] == [FillMissing, Categorify, Normalize]
        assert args['dep_var'] == 'sales'
