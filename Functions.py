# Ignoring warnings
import warnings

warnings.filterwarnings('ignore')

# Data Manipulation
import pandas as pd
from datetime import datetime, timedelta



def generate_dates(start_date, end_date):
    """
    Generates Dates we want to forecast
    :param start_date: the date user wants to forecast from
    :param end_date: the date user wants to forecast to
    :return:

    """
    dates = []
    current_date = start_date

    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def create_features_win(df):
    """
    Create time series features based on time series index.
    """
    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    df.index = pd.to_datetime(df['date'])
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    #df['weekofyear'] = df.index.isocalendar().week
    return df[features]

def create_features_irr(df):
    """
    Create time series features based on time series index.
    """
    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    df[' Observation period'] = pd.to_datetime(df[' Observation period'])
    df.set_index(' Observation period', inplace=True)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    #df['weekofyear'] = df.index.isocalendar().week
    return df[features]

