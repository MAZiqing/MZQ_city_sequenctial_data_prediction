# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/9/20 10:16

import os
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.support_function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WDSDataset(Dataset):
    def __init__(self, dataset_type, encoder_sequence_length, decoder_sequence_length, target_sensor):
        '''
        :param dataset_type: str, dataset type
        :param encoder_sequence_length: int, the length of encoder
        :param decoder_sequence_length: int, the length of decoder
        :param target_sensor: str, the id of target sensor
        '''
        self.encoder_sequence_length = encoder_sequence_length
        self.decoder_sequence_length = decoder_sequence_length
        if dataset_type == 'train':
            self.data = pd.read_csv(os.path.join('../Dataset', 'data_train.csv'),
                                    parse_dates=['datetime'])
        elif dataset_type == 'valid':
            self.data = pd.read_csv(os.path.join('../Dataset', 'data_valid.csv'),
                                    parse_dates=['datetime'])
        elif dataset_type == 'test':
            self.data = pd.read_csv(os.path.join('../Dataset', 'data_test.csv'),
                                    parse_dates=['datetime'])
        else:
            raise Exception('dataset_type error')

        self.flow_column = ['q' + str(i) for i in range(1, 12)]
        self.pressure_column = ['p' + str(i) for i in range(1, 8)]
        self.pressure_column_out = target_sensor
        self.data.fillna(0)
        self.inputs_p_q = torch.tensor(self.data[self.pressure_column + self.flow_column].values,
                                       dtype=torch.float32).to(device)
        self.output_p = torch.tensor(self.data[self.pressure_column_out].values,
                                     dtype=torch.float32).to(device)
        print('dataset prepared !')

    def get_input_size(self):
        input_size = len(self.flow_column + self.pressure_column)
        output_column = (self.pressure_column + self.flow_column).index(self.pressure_column_out)
        return input_size, output_column

    def __len__(self):
        return self.data.shape[0] - 1 * (self.encoder_sequence_length + self.decoder_sequence_length)

    def __getitem__(self, idx):
        inputs_p_q = self.inputs_p_q[idx: idx + self.encoder_sequence_length + self.decoder_sequence_length]
        label_p = self.output_p[idx: idx + self.encoder_sequence_length + self.decoder_sequence_length]
        datetime_i = self.data['datetime'].iloc[idx + self.encoder_sequence_length]
        return inputs_p_q, label_p, datetime.timestamp(datetime_i)

