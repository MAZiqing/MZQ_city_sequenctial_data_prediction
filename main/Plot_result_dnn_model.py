# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/9/10 18:02

# 标准库
from itertools import islice
# 第三方库
import pandas as pd
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pylab import mpl
# 自建库
from machine_learning.Real_data_RnnSpatioAtt_GPUdata_time import args
from machine_learning.src.py_dataset import GpuDataset

device = torch.device('cpu')


def plot_result():
    model = torch.load('./dnn_model/RnnAtt_time.pkl', map_location=torch.device('cpu'))
    model.eval()
    pred = []
    true = []
    dataset = GpuDataset()
    data_loader_plot = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  # num_workers=4
                                  )
    plot_column = 0  # 萧然路600 HD72_309_A1
    k = 2
    t_range = 2
    for input_p, input_config, input_time, label_p in islice(data_loader_plot, k * t_range, (k + 1) * t_range):
        # input_p = input_p.to(device)
        # input_config = input_config.to(device)
        # input_time = input_time.to(device)
        # label_p = label_p.to(device)
    # for i, (input_p, input_config, input_time, label_p) in enumerate(data_loader_plot):
        # print(i, end='\r')
        output_p = model(input_p, input_config, input_time)
        for i, j in zip(output_p[:, 0, 0], label_p):
            pred.append(float(i))
            true.append(float(j))
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(pred, color='blue')
    plt.plot(true, color='orange')
    # plt.xlim([0, 60*48])
    plt.legend(['predicted_value', 'true_value'])
    plt.title(list(dataset.data.columns)[plot_column] + '(萧然路600压力传感器)')
    plt.xlabel('样本点')
    plt.ylabel('压力(MPa)')
    # plt.ylim([0.22, 0.33])
    plt.savefig('../image/dl_model_prediction_result.png', format='png', dpi=1000)
    plt.show()


def plot_data_wash_result():
    dataset = GpuDataset()
    df = dataset.data
    inter = list(df.columns)

    df = df.set_index('datetime')
    # plt.figure()
    # df['out_pressure'].plot()
    # plt.ylabel('Pressure(MPa)')
    # plt.ylim([0.25, 0.32])
    # plt.savefig('XRL600_data_washed_all.png', format='png')
    # plt.show()

    plt.figure()
    df['out_pressure'].tail(500).plot()
    plt.ylabel('Pressure(MPa)')
    plt.ylim([0.25, 0.32])
    plt.savefig('XRL600_data_washed.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 4 * 25])


    for index, i in enumerate(list(df.columns)[0:25]):
        plt.subplot(25, 1, index+1)
        df[i].plot()
        plt.title(i)
    plt.show()


def plot_out_with_pump_raising():
    model = torch.load(args.model_name, map_location=torch.device('cpu'))
    model.eval()
    pred = []
    true = []
    dataset = GpuDataset()
    data_loader_plot = DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  # num_workers=4
                                  )
    plot_column = 0  # 萧然路600 HD72_309_A1
    k = 3
    t_range = 4 * 24 * 4
    for input_p, input_config, input_time, label_p in islice(data_loader_plot, k * t_range, k * t_range + 1):
        output_list = []
        output_p = model(input_p, input_config, input_time)
        output_list.append(output_p)
        config_changed = input_config
        for i in range(0, 11):
            if torch.sum(config_changed[0, :, i]) == 0:
                config_changed[0, :, i] = config_changed[0, :, i] + 1
            output_p_2 = model(input_p, config_changed, input_time)
            output_list.append(output_p_2)
        plt.plot(output_list, '-o')
        plt.show()

        pred.append(float(output_p[0]))
        true.append(float(label_p))
    plt.show()


def plot_from_csv():
    df_result = pd.read_csv('./RnnAtt_time_pred_result.csv', index_col='time_stamp') # <class 'tuple'>: (19955, 3)
    dataset = GpuDataset()
    df = dataset.data  # <class 'tuple'>: (20000, 31)
    df_end = df.join(df_result, how='left')
    df_end = df_end.set_index('datetime')
    start = 10000
    end = start + 200
    df_end['pred'].iloc[start: end].plot()
    df_end['true'].iloc[start: end].plot()
    plt.legend(['pred', 'true'])
    plt.show()
    a = 1


if __name__ == '__main__':
    plot_data_wash_result()
    # plot_result()
    # plot_from_csv()
    # plot_out_with_pump_raising()
    a = 1
