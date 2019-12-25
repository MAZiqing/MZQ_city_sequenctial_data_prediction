# -*- coding: utf-8 -*-
# @Time    : 2019/12/6 18:18
# @Author  : MA Ziqing
# @FileName: arima_test.py

from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

import os
import sys
sys.path.append('..')
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from src.support_function import *
import warnings
# warnings.filterwarnings('ignore')

root_1 = '../Dataset'
df_ori = pd.read_csv(os.path.join(root_1, '3bs_8q_4p_test_resid.csv'),
                parse_dates=['datetime'])
df_train = pd.read_csv(os.path.join(root_1, '3bs_8q_4p_train_resid.csv'),
                parse_dates=['datetime'])
# Column names for all sensors
columns = ['q'+str(i) for i in range(1,12)] + ['p'+str(j) for j in range(1,8)]

# The training set is used for training the VAR model
# The valid set is only for predict and valid
dataset = df_train['q8'].interpolate().fillna(0).iloc[:1000].values
model = ARMA(dataset[:500],
            order=(2,1))
# hist_lag = 48*3
model_fitted = model.fit()
model_fitted.predict(start=1, end=2)
a = 1