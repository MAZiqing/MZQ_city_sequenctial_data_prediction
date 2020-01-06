# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/8/28 14:03

import time
import warnings
import sys
# sys.path.append('..')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import src.model as my_model
import src.py_dataset as my_dataset
from src.support_function import *

# %% --------------------- Global Variable ------------------------
warnings.filterwarnings('ignore')
print('training on cuda:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2017
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Time Series Model')
version = 'v7'
parser.add_argument('--version', type=str, default=version)
parser.add_argument('--mission_name', type=str, default='train_test')
parser.add_argument('--target_sensor', type=str, default='q8')
parser.add_argument('--model_path', type=str, default=os.path.join('../src', 'saved_pkl_model'),
                    help='dataset_name')
if version == 'v1':
    filename = 'ANN.csv'
if version == 'v2':
    filename = 'Seq2Seq.csv'
if version == 'v3':
    filename = 'Transformer.csv'
else:
    filename = 'STAttention.csv'
parser.add_argument('--result_path', type=str, default=os.path.join('../result', 'model_compare', filename))
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64
                    , metavar='N',
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N',
                    help='lr')
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--num_hidden_state', type=int, default=64)
parser.add_argument('--encoder_sequence_length', type=int, default=60)
parser.add_argument('--decoder_sequence_length', type=int, default=4)
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
    def __init__(self):
        self.batch_size = args.batch_size
        self.epoch = 0
        self.num_epoch = args.epochs
        self.dataset = my_dataset.WDSDataset(dataset_type='train',
                                             encoder_sequence_length=args.encoder_sequence_length,
                                             decoder_sequence_length=args.decoder_sequence_length,
                                             target_sensor=args.target_sensor)
        if args.mission_name == 'train_valid':
            dataset_type = 'valid'
        else:
            dataset_type = 'test'
        self.dataset_valid_test = my_dataset.WDSDataset(dataset_type=dataset_type,
                                                        encoder_sequence_length=args.encoder_sequence_length,
                                                        decoder_sequence_length=args.decoder_sequence_length,
                                                        target_sensor=args.target_sensor)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.data_loader_valid = DataLoader(self.dataset_valid_test,
                                            batch_size=self.batch_size,
                                            shuffle=False)
        input_size, output_column = self.dataset.get_input_size()
        if args.version == 'v1':
            self.model = my_model.ANN(input_size, args)
        elif args.version == 'v2':
            self.model = my_model.Seq2Seq(input_size, args)
        elif args.version == 'v3':
            self.model = my_model.Transformer(input_size, args)
        elif args.version == 'v4':
            self.model = my_model.DS_RNN(input_size, args)
        elif args.version == 'v5':
            self.model = my_model.DS_RNN_II(input_size, args)
        elif args.version == 'v6':
            self.model = my_model.DSTP_RNN(input_size, args, output_column)
        elif args.version == 'v7':
            self.model = my_model.hDS_RNN(input_size, args)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.mse_loss = nn.MSELoss().to(device)
        self.l1_loss = nn.L1Loss().to(device)
        self.train_mse_loss = []
        self.train_l1_loss = []
        self.valid_mse_loss = []
        self.valid_l1_loss = []
        self.result_df = None
        self.first_write_result()

    def first_write_result(self):
        self.result_df = pd.DataFrame({
                'model_name': [self.model.model_name + args.version
                               + '_' + args.mission_name],
                'start_time': [str(datetime.now())],
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
                'recon_error': [1000],
                'target_sensor': args.target_sensor,
                'detail': self.model.details,
                'recon_mse': [100],
                'recon_mae': [100]
        })
        self.result_df = self.result_df.set_index('start_time')
        if os.path.exists(args.result_path):
            df_old = pd.read_csv(args.result_path, index_col='start_time', encoding='gbk')
            df_all = pd.concat([df_old, self.result_df], sort=False)
        else:
            df_all = self.result_df
        df_all.to_csv(args.result_path, encoding='gbk')

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
                loss = self.mse_loss(outputs_p, label_p)
                l1loss = self.l1_loss(outputs_p, label_p)
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
                label_p = label_p[:, -args.decoder_sequence_length:]
                loss = self.mse_loss(outputs_p, label_p)
                l1loss = self.l1_loss(outputs_p, label_p)
                l1loss_0 = self.mse_loss(torch.zeros(label_p.shape, dtype=torch.float32).to(device),
                                        label_p)
                valid_total_loss += loss.cpu().detach().numpy()
                valid_total_loss_l1 += l1loss.cpu().detach().numpy()
                total_loss_0 += l1loss_0.cpu().detach().numpy()

                datetime_list += [i.cpu().detach().numpy() for i in datetime_i]
                outputs_p_list += [i.cpu().detach().numpy() for i in outputs_p]
        self.valid_mse_loss.append(valid_total_loss / index)
        self.result_df['valid_mse_curve'] = [self.valid_mse_loss]
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
                recon_list, mean_mse, mean_mae = self.reconstruct(datetime_list, outputs_p_list)
                self.save_model()
                best_valid_mse_loss = valid_mse_loss
                best_epoch = self.epoch
                self.result_df['best_valid_mse_loss'] = best_valid_mse_loss
                self.result_df['best_train_l1_loss'] = train_l1_loss
                self.result_df['best_train_mse_loss'] = train_mse_loss
                self.result_df['best_valid_l1_loss'] = valid_l1_loss
                self.result_df['recon_error'] = [recon_list]
                self.result_df['recon_mse'] = [mean_mse]
                self.result_df['recon_mae'] = [mean_mae]
                # self.save_result_df()
            self.result_df['epoch_for_best_valid_loss'] = str(best_epoch) + '/' + str(self.epoch)
            if self.epoch % 5 == 0:
                self.save_result_df()

    def save_model(self):
        model_path = os.path.join(args.model_path, self.model.model_name+args.target_sensor+'.pkl')
        torch.save(self.model, model_path)
        print('model_svaed ! at ', model_path)

    def load_model(self):
        self.model = torch.load(self.model, os.path.join(args.model_path, self.model.model_name+args.target_sensor+'.pkl'))
        # self.model = torch.load(self.model.model_path, map_location='cpu')
        print('model_loaded: ', self.model.model_path)

    def save_result_df(self):
        df_old = pd.read_csv(args.result_path, index_col='start_time', encoding='gbk')
        str_time = str(list(self.result_df.index)[0])
        self.result_df['start_time_str'] = str_time
        self.result_df = self.result_df.set_index('start_time_str')
        df_old.update(self.result_df)
        df_old.to_csv(args.result_path, encoding='gbk')

    def reconstruct(self, timestamp_list, resid_pred_list):
        if args.mission_name == 'train_valid':
            df1 = pd.read_csv(os.path.join('../Dataset', 'data_valid.csv'),
                              parse_dates=['datetime'])
        elif args.mission_name == 'train_test':
            df1 = pd.read_csv(os.path.join('../Dataset', 'data_test.csv'),
                              parse_dates=['datetime'])
        df1.index = df1['datetime'].apply(datetime.timestamp)
        resid_columns = list(range(1, args.decoder_sequence_length+1))
        df2 = pd.DataFrame(np.array(resid_pred_list), index=np.array(timestamp_list), columns=resid_columns)
        df3 = df1.join(df2)
        start = 200  # start > encoder length (100)
        end = -200  # end < batch_size
        df3['temp'] = 0
        df3['0'] = 0
        mean_mse = 0
        mean_mae = 0
        return_list = []
        for index, i in enumerate(resid_columns):
            df3['temp'] += df3[i].copy()
            pred = (df3['temp']).shift(index).iloc[start:end].values
            true = df3[args.target_sensor].rolling(index+1).sum().iloc[start:end].values
            eval_dict = eval_metrics(true, pred)
            return_list += [eval_dict]
            print(eval_dict)
            mean_mae += eval_dict['MAE'] / 4
            mean_mse += eval_dict['MSE'] / 4
        print('mae', mean_mae, 'mse', mean_mse)
        # return_list += [mean_mae, mean_mse]
        return return_list, mean_mse, mean_mae


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load_model()
    # trainer.valid()
    # total_loss, total_loss_l1 = trainer.valid()
    # trainer.plot_result()
    trainer.main_loop()
    a = 1