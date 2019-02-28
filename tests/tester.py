from hypothesis import given, example
from hypothesis.strategies import text
from io import StringIO
import pandas as pd
from pathlib import Path
import pytest
from unittest import TestCase, mock

fake_csv = StringIO(
    """store,state,date,max_temperature_c,mean_temperature_c,\
    min_temperature_c,dew_point_c,mean_dew_point_c,min_dew_point_c,\
    max_humidity,mean_humidity,min_humidity,max_sea_level_pressureh_pa,\
    mean_sea_level_pressureh_pa,min_sea_level_pressureh_pa,\
    max_visibility_km,mean_visibility_km,min_visibility_km,\
    max_wind_speed_km_h,mean_wind_speed_km_h,max_gust_speed_km_h,\
    precipitationmm,cloud_cover,events,wind_dir_degrees,store_type,\
    assortment,competition_distance,competition_open_since_month,\
    competition_open_since_year,promo2,promo2_since_week,\
    promo2_since_year,promo_interval,day_of_week,sales,customers,open,\
    promo,state_holiday,school_holiday,trend,week_start
    1,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,c,a,1270.0,9.0,2008.0,0,\
    23.595446584938703,2011.7635726795095,None,5,4097.0,494.0,1.0,0.0,\
    0,0.0,85,2015-06-14
    56,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,c,6620.0,3.0,2012.0,1,10.0,\
    2014.0,"Mar,Jun,Sept,Dec",5,9351.0,667.0,1.0,0.0,0,0.0,85,\
    2015-06-14
    69,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,c,c,1130.0,7.224704336399474,\
    2008.6688567674114,1,40.0,2011.0,"Jan,Apr,Jul,Oct",5,6895.0,941.0,\
    1.0,0.0,0,0.0,85,2015-06-14
    77,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,c,1090.0,8.0,2009.0,1,10.0,\
    2014.0,"Jan,Apr,Jul,Oct",5,7656.0,687.0,1.0,0.0,0,0.0,85,2015-06-14
    111,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,c,7890.0,7.224704336399474,\
    2008.6688567674114,1,37.0,2009.0,"Jan,Apr,Jul,Oct",5,6039.0,600.0,\
    1.0,0.0,0,0.0,85,2015-06-14
    120,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,a,2290.0,12.0,2014.0,1,37.0,\
    2009.0,"Jan,Apr,Jul,Oct",5,5135.0,491.0,1.0,0.0,0,0.0,85,2015-06-14
    128,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,c,2000.0,7.224704336399474,\
    2008.6688567674114,1,1.0,2013.0,"Jan,Apr,Jul,Oct",5,7604.0,648.0,\
    1.0,0.0,0,0.0,85,2015-06-14
    130,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,c,a,900.0,7.224704336399474,\
    2008.6688567674114,1,13.0,2010.0,"Jan,Apr,Jul,Oct",5,4318.0,482.0,\
    1.0,0.0,0,0.0,85,2015-06-14
    135,HE,2015-06-20,17,14,11,9,7,5,88,64,37,1021,1020,1018,31.0,11.0,\
    10.0,21,13,40.0,0.0,6.0,Rain,290,d,a,5190.0,7.224704336399474,\
    2008.6688567674114,1,1.0,2013.0,"Jan,Apr,Jul,Oct",5,5823.0,595.0,\
    1.0,0.0,0,0.0,85,2015-06-14
    """)

df = pd.read_csv(fake_csv)

print(df)
