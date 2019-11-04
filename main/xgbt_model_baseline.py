# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/9/12 10:16

from matplotlib import pyplot as plt
import numpy as np
import torch
from machine_learning.Real_data_RNN_GPUdata import GpuDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader
from pmdarima.arima import auto_arima

device = torch.device('cpu')


class Exp1(object):
    def __init__(self, future):
        device = torch.device('cpu')
        dataset = GpuDataset()
        data = dataset.data
        self.future = future
        if future:
            y = data['out_pressure'].iloc[4:].values
            x_column = [data.columns[0]] + list(data.columns[2:13]) + list(data.columns[14:25]) + ['hour']
            x = data[x_column].iloc[0:-4].values
        else:
            y = data['out_pressure'].values.transpose()
            x_column = [data.columns[0]] + list(data.columns[2:13]) + list(data.columns[14:25]) + ['hour']
            x = data[x_column].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False)
        self.y_pred = None
        self.model = GradientBoostingRegressor()

    def fit_predict(self, model_name):
        if model_name == 'xgbt':
            self.model = GradientBoostingRegressor()
        elif model_name == 'lr':
            self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        print('model fit success !')
        self.y_pred = self.model.predict(self.x_test)
        score = np.mean(np.abs(self.y_test - self.y_pred))
        plt.figure()
        plt.plot(self.y_test[0:300])
        plt.plot(self.y_pred[0:300])
        plt.legend(['true', 'pred'])
        plt.title(model_name + '+ future' + str(self.future))
        plt.savefig('../image/'+ model_name + '_future=' + str(self.future) + '.png',
                    format='png', dpi=1000)
        plt.show()
        return score

    def arima(self):
        y = self.y_test[0:3000]
        model = auto_arima(y,  error_action='ignore', trace=1,
                      seasonal=False, m=12)
        print('fit success !')
        y_pred = model.predict(n_periods=100)
        plt.plot(y_pred)
        plt.plot(self.y_test[3000:3100])
        plt.legend(['pred', 'true'])
        plt.show()

# ############## 实验一 P_all^tn, Q_all^tn 预测 P_xrl^tn #######################
# exp1 = Exp1(future=False)
# score1 = exp1.fit_predict(model_name='xgbt')
# # score = 0.004989 = 0.498937
# score2 = exp1.fit_predict(model_name='lr')
# # score = 0.00833 = 0.833

############## 实验二 P_all^tn, Q_all^tn 预测 P_xrl^(tn+1) 即15分钟之后 ###################
exp1 = Exp1(future=True)
score3 = exp1.fit_predict(model_name='xgbt')
# score = 0.005201006536829594
score4 = exp1.fit_predict(model_name='lr')
# score = 0.007055756264558367

# ############## 实验三 P_all^tn, Q_all^tn 预测 P_xrl^(tn+1) 即60分钟之后 ###################
# exp1 = Exp1(future=True)
# # score5 = exp1.fit_predict(model_name='xgbt')
# # # score = 0.0051747254
# # #         0.005175056410368635
# # score6 = exp1.fit_predict(model_name='lr')
# # # score = 0.0059010255


# exp1 = Exp1(future=False)
# exp1.arima()

a = 1
