# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/8/28 14:03

# 标准库
import os
import sys
import time
import argparse
from datetime import datetime
import warnings
# 第三方库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 自建库
sys.path.append('..')
import rnn_model.model as my_model
import src.py_dataset as my_dataset

warnings.filterwarnings("ignore")
#%% --------------------- Global Variable ------------------------
print('training on cuda:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch prediction Model')
parser.add_argument('--have_cuda', type=bool, default=torch.cuda.is_available())

parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.00003, metavar='N',
                    help='lr')
#  lr = 0.000003, epoch=100, train_MSEloss: 0.000237,  train_L1_loss = 0.00996,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=300, train_MSEloss: 0.000264,  train_L1_loss = 0.010879,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=996  train_MSEloss: 0.000110    train_L1_loss: 0.006391
#  lr = 0.0000005 epoch: 676 | time: 24.430149 | train_MSEloss: 0.056815 | train_L1_loss: 0.593346|
parser.add_argument('--encoder_sequence_length', type=int, default=40)
parser.add_argument('--decoder_sequence_length', type=int, default=4)

parser.add_argument('--rolling_window', type=int, default=20)
parser.add_argument('--dataset_path', type=str, default='../Dataset/3bs_8q_4p_dataset_washed.csv',
                    help='dataset_path')
# parser.add_argument('--pred_result_path', type=str, default='./pre_result.csv',
#                     help='dataset_name')
parser.add_argument('--result_path', type=str, default=os.path.join('../result', 'EDLSTM_nD.csv'))
args = parser.parse_args()
print(' lr=', args.learning_rate, ' batch_size=', args.batch_size, ' epochs=', args.epochs)


#%% --------------------- Class Trainer --------------------------

class Trainer(object):
    def __init__(self, batch_size=args.batch_size):
        self.epoch = 0
        self.num_epoch = args.epochs
        self.dataset = my_dataset.GpuDataset(dataset_type='train',
                                             dataset_path=args.dataset_path,
                                             encoder_sequence_length=args.encoder_sequence_length,
                                             decoder_sequence_length=args.decoder_sequence_length)
        self.dataset_valid = my_dataset.GpuDataset(dataset_type='valid',
                                                   dataset_path=args.dataset_path,
                                                   encoder_sequence_length=args.encoder_sequence_length,
                                                   decoder_sequence_length=args.decoder_sequence_length
                                                   )
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      # num_workers=4
                                      )
        self.data_loader_valid = DataLoader(self.dataset_valid,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )
        input_size, output_size, output_column = self.dataset.get_input_output_config_size()
        self.dataset_length = self.dataset.__len__()

        self.model = my_model.EDLSTM_nD(input_size, output_size, output_column, args)
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
        self.first_write_result()

    def first_write_result(self):
        self.result_df = pd.DataFrame(
            {
                'model_name': [self.model.model_name],
                'model_pkl_path': [self.model.model_path],
                'start_time': [datetime.now()],
                'device': [device],
                'learning_rate': [args.learning_rate],
                'batch size': [args.batch_size],
                'encoder_sequence_length': [args.encoder_sequence_length],
                'decoder_sequence_length': [args.decoder_sequence_length],
                'best_train_l1_loss': [1000],
                'best_train_mse_loss': [1000],
                'best_valid_l1_loss': [1000],
                'best_valid_mse_loss': [1000],
                'epoch_for_best_valid_loss': [0],
                'train_mse_curve': [[1, 2, 3]],
                'valid_mse_curve': [[1, 2, 3]]
            }
        )
        self.result_df = self.result_df.set_index('start_time')
        if os.path.exists(args.result_path):
            df_old = pd.read_csv(args.result_path, index_col='start_time')
            df_all = pd.concat([df_old, self.result_df], sort=False)
        else:
            df_all = self.result_df
        df_all.to_csv(args.result_path)

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_loss_l1 = 0.0
        i = 1
        for i, (input_p, input_time, label_p, decoder_time) in enumerate(self.data_loader):
            if input_p.shape[0] == args.batch_size:
                outputs_p = self.model(input_p, input_time, label_p, decoder_time)  # torch.Size([64, 10, 9])
                label_p = label_p[:, -4:]
                if i == 100:
                    if self.epoch % 20 == 0:
                        print('label:', np.around(label_p[0].cpu().detach().numpy(), decimals=5))
                        print('output:', np.around(outputs_p[0].squeeze().cpu().detach().numpy(), decimals=5))
                loss = self.criterion(outputs_p.squeeze(), label_p.squeeze())
                l1loss = self.criterionL1(outputs_p.squeeze(), label_p.squeeze())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().detach().numpy()
                total_loss_l1 += l1loss.cpu().detach().numpy()

        self.train_mse_loss.append(total_loss / i)
        self.result_df['train_mse_curve'] = [self.train_mse_loss]
        if i % 5 == 0:
            self.save_result_df()
        self.train_l1_loss.append(total_loss_l1 / i)
        return total_loss/i, total_loss_l1/i

    def valid(self):
        self.model.eval()
        valid_total_loss = 0.0
        valid_total_loss_l1 = 0.0
        index = 1
        for index, (input_p, input_time, label_p, decoder_time) in enumerate(self.data_loader_valid):
            outputs_p = self.model(input_p, input_time, label_p, decoder_time)
            outputs_p = outputs_p.squeeze()
            label_p = label_p[:, -4:]
            loss = self.criterion(outputs_p, label_p)
            l1loss = self.criterionL1(outputs_p, label_p)
            valid_total_loss += loss.cpu().detach().numpy()
            valid_total_loss_l1 += l1loss.cpu().detach().numpy()

        self.valid_mse_loss.append(valid_total_loss / index)
        self.result_df['valid_mse_curve'] = [self.valid_mse_loss]
        if index % 5 == 0:
            self.save_result_df()
        self.valid_l1_loss.append(valid_total_loss_l1 / index)

        # df = pd.DataFrame({'pred': outputs_p_list, 'true': label_p_list, 'time_stamp': time_stamp_list})
        # df.to_csv('./' + self.model.model_name + '_pred_result.csv')
        return valid_total_loss/index, valid_total_loss_l1/index

    def main_loop(self):
        best_valid_loss = 100000
        valid_mse_loss = 100
        valid_l1_loss = 100
        for self.epoch in range(0, self.num_epoch):
            since = time.time()

            train_mse_loss, train_l1_loss = self.train()
            valid_mse_loss, valid_l1_loss = self.valid()

            print('epoch: {:} | time: {:2f} | train_MSEloss: {:4f} | train_L1_loss: {:4f}| '
                  'lr: {:4f}| best_valid_MSEloss: {:4f} | valid_L1_loss: {:4f}'.format(
                self.epoch, time.time()-since, train_mse_loss, train_l1_loss,
                args.learning_rate, valid_mse_loss, valid_l1_loss
            ))

            if valid_mse_loss < best_valid_loss:
                self.save_model()
                best_valid_loss = valid_mse_loss
                self.result_df['best_train_l1_loss'] = train_l1_loss
                self.result_df['best_train_mse_loss'] = train_mse_loss
                self.result_df['best_valid_l1_loss'] = valid_l1_loss
                self.result_df['best_valid_mse_loss'] = valid_mse_loss
                self.result_df['epoch_for_best_valid_loss'] = self.epoch
                self.save_result_df()

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
