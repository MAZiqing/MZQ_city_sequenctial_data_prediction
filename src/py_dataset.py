# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/9/20 10:16

from src.support_function import *
# 标准库
import os
import argparse
from datetime import datetime
# 第三方库
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# 自建库


seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %% ---------------- Class GpuDataset()----------------------------
class GpuDataset(Dataset):
    def __init__(self, dataset_type, dataset_path,
                 encoder_sequence_length, decoder_sequence_length):
        # self.args = args
        self.dataset_type = dataset_type
        self.encoder_sequence_length = encoder_sequence_length
        self.decoder_sequence_length = decoder_sequence_length
        # self.delta_t = args_dataset.sequence_long
        self.data = pd.read_csv(dataset_path)  # shape = (93715, 28)
        self.flow_column = ['q' + str(i) for i in range(1, 12)]
        self.pressure_column = ['p' + str(i) for i in range(1, 8)]
        self.pressure_column_out = 'q8'
        # self.time_column = ['hour', 'weekday', 'season', 'month', 'weekofyear']
        self.time_column = ['hour', 'weekday', 'month', 'weekofyear']
        self.data_pre_treat()
        self.inputs_p_q = torch.tensor(self.data[self.pressure_column + self.flow_column].values,
                                       dtype=torch.float32).to(device)
        self.inputs_time = torch.tensor(self.data[self.time_column].values,
                                        dtype=torch.long).to(device)
        self.output_p = torch.tensor(self.data['out_pressure'].values,
                                     dtype=torch.float32).to(device)
        print('dataset prepared !')

    def get_input_output_config_size(self):
        input_size = len(self.flow_column + self.pressure_column)
        output_size = 1
        output_column = list(self.data.columns).index(self.pressure_column_out)
        return input_size, output_size, output_column

    def data_pre_treat(self):
        split_point = datetime.timestamp(datetime(2019, 1, 1))
        if self.dataset_type == 'train':
            self.data = self.data[self.data['timestamp'] < split_point]
        elif self.dataset_type == 'valid':
            self.data = self.data[self.data['timestamp'] > split_point]
        self.data['datetime'] = self.data['datetime'].apply(str_to_datetime)
        self.data['season'] = self.data['datetime'].apply(datetime_season)
        self.data['weekday'] = self.data['datetime'].apply(datetime_weekday)
        self.data['hour'] = self.data['datetime'].apply(datetime_hour)
        self.data['month'] = self.data['datetime'].apply(datetime_month)
        self.data['weekofyear'] = self.data['datetime'].apply(datetime_weekofyear)
        self.data['out_pressure'] = self.data[self.pressure_column_out]
        self.data.fillna(0)
        df_normalized = normalize(self.data[self.flow_column +
                                            self.pressure_column +
                                            ['out_pressure']])
        self.data.update(df_normalized)
        a = 1

    def __len__(self):
        return self.data.shape[0] - 3 * (self.encoder_sequence_length + self.decoder_sequence_length)

    def __getitem__(self, idx):
        inputs_p_q = self.inputs_p_q[idx: idx + self.encoder_sequence_length]
        # inputs_config = self.inputs_config[idx: idx + self.delta_t]
        inputs_time = self.inputs_time[idx: idx + self.encoder_sequence_length]
        label_p = self.output_p[idx: idx + self.encoder_sequence_length + self.decoder_sequence_length]
        # aim_p = self.output_p[idx: idx + self.encoder_sequence_length]
        decoder_time = self.inputs_time[idx + self.encoder_sequence_length:
                                        idx + self.encoder_sequence_length + self.decoder_sequence_length]
        return inputs_p_q, inputs_time, label_p, decoder_time

#%%


class RandomDataset(object):
    def __init__(self):
        self.delta_t = 15
        self.inputs_p_q = torch.ones(10000, 15, dtype=torch.float32)
        self.inputs_config = torch.ones(10000, 11, dtype=torch.float32)
        self.inputs_time = torch.ones(10000, 3, dtype=torch.long)
        # self.time_stamp = torch.ones(1, 10000, dtype=torch.float32)
        self.output_p = torch.ones(10000, 1, dtype=torch.float32)

    @staticmethod
    def get_input_output_config_size():
        input_size = 15
        config_size = 11

        # output_size = len(self.pressure_column)
        output_size = 1
        return input_size, config_size, output_size

    def __len__(self):
        return 9000

    def __getitem__(self, idx):
        inputs_p_q = self.inputs_p_q[idx: idx + self.delta_t]
        inputs_config = self.inputs_config[idx: idx + self.delta_t]
        inputs_time = self.inputs_time[idx: idx + self.delta_t]
        output_pressure = self.output_p[idx + self.delta_t + 1]
        time_stamp = 1
        # torch.Size([10]) -> torch.Size([10, 1])
        return inputs_p_q, inputs_config, inputs_time, output_pressure, time_stamp


class GpuResidualDataset(Dataset):
    def __init__(self, dataset_type, dataset_path,
                 encoder_sequence_length, decoder_sequence_length):
        # self.args = args
        self.dataset_type = dataset_type
        self.encoder_sequence_length = encoder_sequence_length
        self.decoder_sequence_length = decoder_sequence_length
        # self.delta_t = args_dataset.sequence_long
        self.data = pd.read_csv(dataset_path, index_col=0)  # shape = (93715, 28)
        self.flow_column = ['q' + str(i) for i in range(1, 12)]
        self.pressure_column = ['p' + str(i) for i in range(1, 8)]
        self.pressure_column_out = 'q8'
        self.continue_column = self.flow_column + \
                               self.pressure_column
        # self.time_column = ['hour', 'weekday', 'season', 'month', 'weekofyear']
        self.time_column = ['hour', 'weekday', 'month', 'weekofyear']
        self.df_resid = None
        self.df_seasonal = None
        self.data_pre_treat()

        self.inputs_p_q = torch.tensor(self.df_resid[self.pressure_column + self.flow_column].values,
                                       dtype=torch.float32).to(device)
        self.inputs_time = torch.tensor(self.df_resid[self.time_column].values,
                                        dtype=torch.long).to(device)
        self.output_p = torch.tensor(self.df_resid[self.pressure_column_out].values,
                                     dtype=torch.float32).to(device)
        print('dataset prepared !')

    def get_input_output_config_size(self):
        input_size = len(self.flow_column + self.pressure_column)
        output_size = 1
        output_column = list(self.data.columns).index(self.pressure_column_out)
        return input_size, output_size, output_column

    def data_pre_treat(self):
        split_point = datetime.timestamp(datetime(2019, 1, 1))
        if self.dataset_type == 'train':
            self.data = self.data[self.data['timestamp'] < split_point]
        elif self.dataset_type == 'valid':
            self.data = self.data[self.data['timestamp'] > split_point]
        self.data['datetime'] = self.data['datetime'].apply(str_to_datetime)
        self.data['season'] = self.data['datetime'].apply(datetime_season)
        self.data['weekday'] = self.data['datetime'].apply(datetime_weekday)
        self.data['hour'] = self.data['datetime'].apply(datetime_hour)
        self.data['month'] = self.data['datetime'].apply(datetime_month)
        self.data['weekofyear'] = self.data['datetime'].apply(datetime_weekofyear)
        # self.data['out_pressure'] = self.data[self.pressure_column_out]
        self.data.fillna(0)
        df_normalized = normalize(self.data[self.continue_column])
        self.data.update(df_normalized)
        self.data[self.continue_column] = self.data[self.continue_column].diff()
        self.data = self.data.dropna()
        ss_decomposition = seasonal_decompose(x=self.data[self.continue_column], model='additive', freq=48)
        self.df_seasonal = ss_decomposition.seasonal
        self.df_resid = self.data
        self.df_resid[self.continue_column] = self.data[self.continue_column] - self.df_seasonal
        a = 1

    def __len__(self):
        return self.data.shape[0] - 3 * (self.encoder_sequence_length + self.decoder_sequence_length)

    def __getitem__(self, idx):
        inputs_p_q = self.inputs_p_q[idx: idx + self.encoder_sequence_length]
        # inputs_config = self.inputs_config[idx: idx + self.delta_t]
        inputs_time = self.inputs_time[idx: idx + self.encoder_sequence_length]
        label_p = self.output_p[idx: idx + self.encoder_sequence_length + self.decoder_sequence_length]
        # aim_p = self.output_p[idx: idx + self.encoder_sequence_length]
        decoder_time = self.inputs_time[idx + self.encoder_sequence_length:
                                        idx + self.encoder_sequence_length + self.decoder_sequence_length]
        return inputs_p_q, inputs_time, label_p, decoder_time

