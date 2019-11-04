# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 20:57
# @Author  : MA Ziqing
# @FileName: arima.py

import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pmdarima as pm
import warnings


warnings.filterwarnings("ignore")

df = pd.read_csv('./3bs_8q_4p_dataset_washed.csv',
                parse_dates=['datetime'])

station_for_pred = 'q8'

split_point = datetime.timestamp(datetime(2019,1,1))
df_train = df[df['timestamp'] < split_point]
df_test = df[df['timestamp'] > split_point]
data_train = df_train[station_for_pred].values
data_test = df_test[station_for_pred].values

model_3 = pm.ARIMA((2,1,4),
                  seasonal_order=(2,0,2,12)
                  )
model_3.fit(data_train)
print('fit success! ')

timestep_for_pred = 4*2 #数据是每十五分钟一个，预测两个小时后，即8个点以后
pred_list = []
for yi in tqdm(data_test):
    # 每次预测 15min - 2h 之间的8个点，保留最后一个点
    pred_list += [model_3.predict(n_periods=timestep_for_pred)[-1]]
    # 更新模型
    model_3.update(yi)


df_test['pred_arima'] = pred_list
df_test['pred_arima'] = df_test.shift(timestep_for_pred)
df_test.to_csv('./arima_result.csv')
print('model saved ! ')