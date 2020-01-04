# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/8/28 14:03

# 标准库
import time
import warnings
import sys

sys.path.append('..')
# 第三方库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 自建库
import src.model as my_model
import src.py_dataset as my_dataset
from src.support_function import *

# %% --------------------- Global Variable ------------------------
warnings.filterwarnings('ignore')
print('training on cuda:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser(description='Time Series Model')
parser.add_argument('--version', type=str, default='v9')
parser.add_argument('--have_cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--mission_name', type=str, default='train_test')
parser.add_argument('--target_sensor', type=str, default='q8')

parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024
                    , metavar='N',
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='N',
                    help='lr')
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--num_hidden_state', type=int, default=64)

#  lr = 0.000003, epoch=100, train_MSEloss: 0.000237,  train_L1_loss = 0.00996,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=300, train_MSEloss: 0.000264,  train_L1_loss = 0.010879,  model = './DNN_model_best.pkl'
#  lr = 0.0000009 epoch=996  train_MSEloss: 0.000110    train_L1_loss: 0.006391
#  lr = 0.0000005 epoch: 676 | time: 24.430149 | tra
#  in_MSEloss: 0.056815 | train_L1_loss: 0.593346|
parser.add_argument('--encoder_sequence_length', type=int, default=60)
parser.add_argument('--decoder_sequence_length', type=int, default=10)

# parser.add_argument('--rolling_window', type=int, default=20)
parser.add_argument('--dataset_path', type=str, default='../Dataset/3bs_8q_4p_dataset_washed.csv',
                    help='dataset_path')
parser.add_argument('--model_path', type=str, default=os.path.join('../src', 'saved_pkl_model'),
                    help='dataset_name')
parser.add_argument('--result_path', type=str, default=os.path.join('../result', 'STAttention.csv'))
args = parser.parse_args()

print(args.mission_name,
      'version=', args.version,
      'target_sensor =', args.target_sensor,
      'lr =', args.learning_rate,
      'seed =', seed,
      'batch_size =', args.batch_size,
      'epochs =', args.epochs,
      'encoder_length =', args.encoder_sequence_length)


# %% --------------------- Class Trainer --------------------------
class Trainer(object):
    def __init__(self, batch_size=args.batch_size):
        self.epoch = 0
        self.num_epoch = args.epochs
        self.dataset = my_dataset.WDSDataset(dataset_type='train',
                                             encoder_sequence_length=args.encoder_sequence_length,
                                             decoder_sequence_length=args.decoder_sequence_length,
                                             target_sensor=args.target_sensor)
        if args.mission_name == 'train_valid':
            self.dataset_valid = my_dataset.WDSDataset(dataset_type='valid',
                                                       encoder_sequence_length=args.encoder_sequence_length,
                                                       decoder_sequence_length=args.decoder_sequence_length,
                                                       target_sensor=args.target_sensor)
        elif args.mission_name == 'train_test':
            self.dataset_valid = my_dataset.WDSDataset(dataset_type='test',
                                                       encoder_sequence_length=args.encoder_sequence_length,
                                                       decoder_sequence_length=args.decoder_sequence_length,
                                                       target_sensor=args.target_sensor)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      )
        self.data_loader_valid = DataLoader(self.dataset_valid,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            )
        self.dataset_length = self.dataset.__len__()
        input_size, output_column = self.dataset.get_input_size()
        if args.version == 'v3':
            self.model = my_model.DS_RNN_II(input_size, output_size, output_column, args)
        elif args.version == 'v4':
            self.model = my_model.dSTA_RNN(input_size, output_size, output_column, args)
            # self.model = my_model.STAttention_notf_v4(input_size, output_size, output_column, args)
        elif args.version == 'v5':
            self.model = my_model.hDS_RNN(input_size, output_size, output_column, args)

        elif args.version == 'v6':
            self.model = my_model.STAttention_notf_v6(input_size, output_size, output_column, args)
        elif args.version == 'v7':
            self.model = my_model.STAttention_notf_v7(input_size, output_size, output_column, args)
        elif args.version == 'v8':
            self.model = my_model.DS_RNN(input_size, output_size, output_column, args)
        elif args.version == 'v9':
            self.model = my_model.DSTP_RNN(input_size, args, output_column)

        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0002) 不收敛
        self.criterion = nn.MSELoss().to(device)
        self.criterionL1 = nn.L1Loss().to(device)
        self.batch_size = batch_size
        self.train_mse_loss = []
        self.train_l1_loss = []
        self.valid_mse_loss = []
        self.valid_l1_loss = []
        self.result_df = None
        # self.first_write_result()

    def first_write_result(self):
        self.result_df = pd.DataFrame(
            {
                'model_name': [self.model.model_name + args.version
                               + '_' + args.mission_name
                               + '_' + 'new_recon'],
                'model_pkl_path': [self.model.model_path],
                'start_time': [datetime.now()],
                'device': [device],
                'learning_rate': [args.learning_rate],
                'seed': [seed],
                'batch size': [args.batch_size],
                'encoder_sequence_length': [args.encoder_sequence_length],
                'decoder_sequence_length': [args.decoder_sequence_length],
                'best_train_l1_loss': [1000],
                'best_train_mse_loss': [1000],
                'best_valid_l1_loss': [1000],
                'best_valid_mse_loss': [1000],
                'epoch_for_best_valid_loss': [0],
                'train_mse_curve': [[1, 2, 3]],
                'valid_mse_curve': [[1, 2, 3]],
                'num_hidden_state': args.num_hidden_state,
                'num_layer': args.num_layer,
                'recon_error': [1000]
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
        for i, (input_p, label_p, datetime_i) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            if input_p.shape[0] == args.batch_size:
                outputs_p = self.model(input_p, label_p)  # torch.Size([64, 10, 9])
                label_p = label_p[:, -args.decoder_sequence_length:]
                # outputs_p = outputs_p.squeeze()
                if i == 100:
                    if self.epoch % 10 == 0:
                        print('label:', np.around(label_p[0].cpu().detach().numpy(), decimals=5))
                        print('output:', np.around(outputs_p[0].cpu().detach().numpy(), decimals=5))
                loss = self.criterion(outputs_p, label_p)
                l1loss = self.criterionL1(outputs_p, label_p)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().detach().numpy()
                total_loss_l1 += l1loss.cpu().detach().numpy()

        self.train_mse_loss.append(total_loss / i)
        self.result_df['train_mse_curve'] = [self.train_mse_loss]
        return total_loss / i, total_loss_l1 / i

    def valid(self):
        self.model.eval()
        valid_total_loss = 0.0
        valid_total_loss_l1 = 0.0
        total_loss_0 = 0.0
        index = 1
        datetime_list = []
        outputs_p_list = []
        for index, (input_p, label_p, datetime_i) in enumerate(self.data_loader_valid):
            if input_p.shape[0] == args.batch_size:
                outputs_p = self.model(input_p, label_p)

                SA_score = torch.mean(torch.cat(self.model.spatial_att_score_list, dim=2), dim=0).cpu().detach().numpy()
                TA_score = torch.mean(torch.cat(self.model.temporal_att_score_list, dim=1), dim=0).cpu().detach().numpy()
                df_SA_score = pd.DataFrame(SA_score, index=self.dataset.continue_column)
                df_SA_score.to_csv(os.path.join('../result', 'visu', self.model.model_name + '_SA.csv'))
                df_TA_score = pd.DataFrame(TA_score)
                df_TA_score.to_csv(os.path.join('../result', 'visu', self.model.model_name + '_TA.csv'))

                # fig, ax = plt.subplots(1, 1)
                # img = ax.imshow(SA_score)
                # ax.set_yticks([i+0.5 for i in list(range(0, 18))])
                # ax.set_yticklabels(self.dataset.continue_column)
                # plt.savefig('./test.png')
                # plt.imshow(TA_score)

                label_p = label_p[:, -args.decoder_sequence_length:]
                loss = self.criterion(outputs_p, label_p)
                l1loss = self.criterionL1(outputs_p, label_p)
                l1loss_0 = self.criterion(torch.zeros(label_p.shape, dtype=torch.float32).to(device),
                                          label_p)
                valid_total_loss += loss.cpu().detach().numpy()
                valid_total_loss_l1 += l1loss.cpu().detach().numpy()
                total_loss_0 += l1loss_0.cpu().detach().numpy()

                datetime_list += [i.cpu().detach().numpy() for i in datetime_i]
                outputs_p_list += [i.cpu().detach().numpy() for i in outputs_p]
        self.valid_mse_loss.append(valid_total_loss / index)
        # self.result_df['valid_mse_curve'] = [self.valid_mse_loss]
        return valid_total_loss / index, valid_total_loss_l1 / index, total_loss_0 / index, datetime_list, outputs_p_list

    def main_loop(self):
        best_valid_mse_loss = 100000
        best_epoch = 1
        for self.epoch in range(0, self.num_epoch):
            since = time.time()
            train_mse_loss, train_l1_loss = self.train()
            valid_mse_loss, valid_l1_loss, valid_mse_loss_0, datetime_list, outputs_p_list = self.valid()
            # if self.epoch % 5 == 0:
            print('epoch: {:} | time: {:2f} | train_MSEloss: {:4f} | train_L1_loss: {:4f} |'
                  ' best_valid_MSEloss: {:4f}/{:4f} | valid_L1_loss: {:4f}'.format(
                self.epoch, time.time() - since, train_mse_loss, train_l1_loss,
                best_valid_mse_loss, valid_mse_loss_0, valid_l1_loss
            ))
            if valid_mse_loss < best_valid_mse_loss:
                recon_list = self.reconstruct(datetime_list, outputs_p_list)
                # self.save_model()
                best_valid_mse_loss = valid_mse_loss
                best_epoch = self.epoch
                self.result_df['best_valid_mse_loss'] = best_valid_mse_loss
                self.result_df['best_train_l1_loss'] = train_l1_loss
                self.result_df['best_train_mse_loss'] = train_mse_loss
                self.result_df['best_valid_l1_loss'] = valid_l1_loss
                self.result_df['recon_error'] = [recon_list]
                # self.save_result_df()
            self.result_df['epoch_for_best_valid_loss'] = str(best_epoch) + '/' + str(self.epoch)
            # if self.epoch % 5 == 0:
            #     self.save_result_df()

    def save_model(self):
        torch.save(self.model, os.path.join(args.model_path, self.model.model_name+args.target_sensor+'.pkl'))
        print('model_svaed ! at ', self.model.model_path)

    def load_model(self):

        model_path = os.path.join(args.model_path, self.model.model_name+args.target_sensor+'.pkl')
        # torch.save(self.model, model_path)
        # print('model_svaed ! at ', model_path)
        self.model = torch.load(model_path)
        print('model_loaded: ', model_path)

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

    def reconstruct(self, timestamp_list, resid_pred_list):
        if args.mission_name == 'train_valid':
            df1 = pd.read_csv(os.path.join('../Dataset', 'data_valid.csv'),
                              parse_dates=['datetime'])
        elif args.mission_name == 'train_test':
            df1 = pd.read_csv(os.path.join('../Dataset', 'data_test.csv'),
                              parse_dates=['datetime'])
        df1.index = df1['datetime'].apply(datetime.timestamp)
        resid_columns = ['resi1', 'resi2', 'resi3', 'resi4']
        df2 = pd.DataFrame(np.array(resid_pred_list), index=np.array(timestamp_list), columns=resid_columns)
        df3 = df1.join(df2)
        start = 200  # start > encoder length (100)
        end = -200  # end < batch_size
        df3['temp'] = 0
        df3['0'] = 0
        mean_mse = 0
        mean_mae = 0
        # plt.figure(figsize=(200, 5))
        # plt.plot(df3['q8_ori'].iloc[start:end])
        return_list = []
        for index, i in enumerate(resid_columns):
            df3['temp'] += df3[i].copy()
            pred = (df3['temp']).shift(index).iloc[start:end].values
            true = df3[args.target_sensor].rolling(index+1).sum().iloc[start:end].values
            eval_dict = eval_metrics(true, pred)
            return_list += [eval_dict]
            print(eval_dict)
            # df3['pre_' + i] = (df3['temp'] + df3['q8_seasonal'].shift(-index).rolling(index + 1).sum() +
            #                    df3['q8_ori'].copy().shift(-1)).shift(index)
            # eval_dict = eval_metrics(df3['pre_' + i].iloc[start:end].values, df3['q8_ori'].iloc[start:end].values)
            # print(eval_dict)
            # if index < 2:
            #     plt.plot(df3['pre_' + i].iloc[start:end])
            mean_mae += eval_dict['MAE'] / 4
            mean_mse += eval_dict['MSE'] / 4
        print('mae', mean_mae, 'mse', mean_mse)
        return_list += [mean_mae, mean_mse]
        return return_list
        # plt.savefig('./test.png')
        # plt.show()
        # plt.figure()
        # df3['q8'].iloc[start:300].plot()
        # df3['resi1'].shift(0).iloc[start:300].plot()
        # df3.to_csv('./test.csv')
        # plt.savefig('./test2.png')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_model()
    valid_mse_loss, valid_l1_loss, valid_mse_loss_0, datetime_list, outputs_p_list = trainer.valid()
    recon_list = trainer.reconstruct(datetime_list, outputs_p_list)

    # total_loss, total_loss_l1 = trainer.valid()
    # trainer.plot_result()
    # trainer.main_loop()
    a = 1