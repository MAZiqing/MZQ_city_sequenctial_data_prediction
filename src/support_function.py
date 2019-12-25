# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 19:48
# @Author  : MA Ziqing
# @FileName: support_function.py

import os
import argparse
from datetime import datetime
# 第三方库
import numpy as np
import pandas as pd
# import torch
# from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import metrics


def eval_metrics(y_true, y_pred):
    metrics_dict = dict()
    metrics_dict['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    metrics_dict['MSE'] = metrics.mean_squared_error(y_true, y_pred)
    # metrics_dict['MAPE'] = np.mean(np.true_divide(np.abs(y_true-y_pred), y_true))
    return metrics_dict

# def timestamp_to_datetime(x): 忽略 return datetime.fromtimestamp(x)


# def datetime_to_timestamp(x): 忽略 return datetime.timestamp(x)


def str_to_datetime(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def datetime_to_str(x):
    return datetime.strftime(x, '%Y-%m-%d %H:%M:%S')


def timestamp_to_str(x):
    x = datetime.fromtimestamp(x)
    return datetime.strftime(x, '%Y-%m-%d %H:%M:%S')


def str_to_timestamp(x):
    x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return int(datetime.timestamp(x))


def datetime_hour(x):
    return x.hour


def datetime_year(x):
    return x.year


def datetime_weekday(x):
    return x.weekday()


def datetime_dayofyear(x):
    return x.dayofyear


def datetime_weekofyear(x):
    return x.weekofyear


def datetime_season(x):
    return x.quarter


def datetime_minute(x):
    return x.minute


def datetime_month(x):
    return x.month


def normalize(df):
    # Create x, where x the 'scores' column's values as floats
    x = df.values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    return df_normalized

