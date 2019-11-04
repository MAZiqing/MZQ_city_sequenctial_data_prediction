# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/8/28 14:03

# 标准库
import os
import time
import argparse
from datetime import datetime
# 第三方库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 自建库
import rnn_model.model as my_model
import src.py_dataset as my_dataset

#%% --------------------- Global Variable ------------------------
print('training on cuda:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--have_cuda', type=bool, default=torch.cuda.is_available())

parser.add_argument('--epochs', type=int, default=2000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.000002, metavar='N',
                    help='lr')
#  lr = 0.000003, epoch=100, train_MSEloss: 0.000237,  train_L1_loss = 0.00996,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=300, train_MSEloss: 0.000264,  train_L1_loss = 0.010879,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=996  train_MSEloss: 0.000110    train_L1_loss: 0.006391
#  lr = 0.0000005 epoch: 676 | time: 24.430149 | train_MSEloss: 0.056815 | train_L1_loss: 0.593346|
parser.add_argument('--sequence_long', type=int, default=15)
parser.add_argument('--rolling_window', type=int, default=20)
parser.add_argument('--dataset_name', type=str, default='../Dataframe_all_configsensor_15min_20MinRolling.csv',
                    help='dataset_name')
# parser.add_argument('--pred_result_path', type=str, default='./pre_result.csv',
#                     help='dataset_name')
parser.add_argument('--result_path', type=str, default='./result/exp_result_file_2.csv')
args = parser.parse_args()
print(' lr=', args.learning_rate, ' batch_size=', args.batch_size, ' epochs=', args.epochs)


#%% --------------------- Class Trainer --------------------------

class Trainer(object):
    def __init__(self, batch_size=args.batch_size):
        self.epoch = 0
        self.num_epoch = args.epochs

        self.dataset = my_dataset.RandomDataset()
        # self.dataset_valid = my_dataset.GpuDataset(name='valid')
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      # num_workers=4
                                      )
        # self.data_loader_valid = DataLoader(self.dataset_valid,
        #                                     batch_size=batch_size,
        #                                     shuffle=False,
        #                                     )
        input_size, config_size, output_size = self.dataset.get_input_output_config_size()
        self.dataset_length = self.dataset.__len__()

        self.model = my_model.SpatioTemporelLSTM_v2(input_size, config_size, output_size)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0002) 不收敛
        self.criterion = nn.MSELoss().to(device)
        self.criterionL1 = nn.L1Loss()
        self.batch_size = batch_size
        self.train_mse_loss = []
        self.train_l1_loss = []
        self.valid_mse_loss = []
        self.valid_l1_loss = []
        self.result_df = None
        # self.first_write_result()

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_loss_l1 = 0.0
        i = 1
        for i, (input_p, input_config, input_time, label_p, time_stamp) in enumerate(self.data_loader):
            if input_p.shape[0] == args.batch_size:
                outputs_p = self.model(input_p, input_config, input_time)  # torch.Size([64, 10, 9])
                outputs_p = outputs_p.squeeze()
                if i == 100:
                    if self.epoch % 20 == 0:
                        print('label:', np.around(label_p[0].cpu().detach().numpy(), decimals=5))
                        print('output:', np.around(outputs_p[0].cpu().detach().numpy(), decimals=5))
                loss = self.criterion(outputs_p.squeeze(), label_p.squeeze())
                l1loss = self.criterionL1(outputs_p.squeeze(), label_p.squeeze())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().detach().numpy()
                total_loss_l1 += l1loss.cpu().detach().numpy()
        total_loss = total_loss / i
        total_loss_l1 = total_loss_l1 / i
        return total_loss, total_loss_l1

    def main_loop(self):
        for self.epoch in range(0, self.num_epoch):
            train_mse_loss, train_l1_loss = self.train()

    def save_model(self):
        torch.save(self.model, self.model.model_path)
        print('model_svaed ! at ', self.model.model_path)

    def load_model(self):
        self.model = torch.load(self.model.model_path)

    def save_result_df(self):
        # if os.path.exists(args.result_path):
        df_old = pd.read_csv(args.result_path, index_col='start_time')
        #     df_all = pd.concat([self.result_df, df_old], axis=1)
        # else:
        str_time = str(list(self.result_df.index)[0])
        self.result_df['start_time_str'] = str_time
        self.result_df = self.result_df.set_index('start_time_str')
        df_old.update(self.result_df)
        df_old.to_csv(args.result_path)


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load_model()
    # total_loss, total_loss_l1 = trainer.valid()
    # trainer.plot_result()
    trainer.main_loop()
    a = 1
# #
# 比较数据集
#     xnum = -10000
#     dataset_rolled = pd.read_csv('../Dataframe_all_configsensor_15min_20MinRolling.csv', index_col=0)
#     dataset = pd.read_csv('../Dataframe_all_configsensor_v2_15min.csv', index_col=0)
#     dataset_rolled.iloc[xnum:, 1].plot()
#     dataset.iloc[xnum:, 1].plot()
#     plt.legend(['rolled', 'no_roll'])
#     plt.show()
#     a = 1


#
#
#
#     #  线性回归方法 0.013562263160189653
#     data = pd.read_csv('../Virtual_Dataframe_16sensor_all.csv', index_col=0)  # shape = (51025, 39)
#
#     station_unconcerd_list = ['工农村1200压力', '大池娄压力',
#                               '张神殿村800压力', '群谊村800压力',
#                               '幸福村1200压力', '农商银行南压力',
#                               '国际机场西压力', '八大表房1200压力',
#                               ]
#     data = data.drop(columns=station_unconcerd_list)  # 51150 行
#     bool_list = None
#     # wash_lim = -100  # 还剩48715 行
#     wash_lim = 0  # 还剩17385 行
#     for i in range(0, 13 - len(station_unconcerd_list)):
#         if bool_list is None:
#             bool_list = data.iloc[:, i] > wash_lim
#         else:
#             bool_list = (bool_list & (data.iloc[:, i] > wash_lim))
#     data = data[bool_list]
#     a = 1
#
#     inputs_name = [str(i) for i in range(7, 18)]
#     inputs_2 = data[inputs_name + ['三水厂流量']].values
#     ouputs_columns = ['萧然路600压力', '三水厂压力',
#                       '市中心东路600压力',
#                       '江东大院南压力', '萧绍800压力', '北干1000压力'
#                       #  '工农村1200压力', '大池娄压力','幸福村1200压力''农商银行南压力'
#                       ]
#     outputs = data[ouputs_columns].values / 100
#     model = LinearRegression()
#     # model = SVR()
#     outputs = outputs[:, 0]
#     model.fit(inputs_2, outputs)
#     outputs_pred = model.predict(inputs_2)
#     l1 = np.mean(np.abs(outputs-outputs_pred))
#     plt.plot(outputs_pred[1000:1000 + 400], color='green')
#     plt.plot(outputs[1000: 1000 + 400])
#     plt.show()
#     a = 1
