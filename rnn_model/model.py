# Copyright(c) 2009 IW
# All rights reserved
#
# @Author  : maziqing<maziqing@interns.ainnovation.com>
# @Time    : 2019/8/28 14:03

# 标准库
import os
# 第三方库
import torch
import torch.nn as nn
# 自建库
import src.py_dataset as my_dataset
# print('training on cuda:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM_1D(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(LSTM_1D, self).__init__()
        self.model_name = 'LSTM_1D'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model', self.model_name+'.pkl')
        self.args = args
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        # self.input_size = input_p_q_size + 5 + 3 + 3
        self.hidden_size = 64
        self.output_size = output_size
        self.LSTM = nn.LSTM(num_layers=2,
                            input_size=1,
                            hidden_size=self.hidden_size,
                            dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, input_time, aim_p, decoder_time):
        x = aim_p[:, 0:self.args.encoder_sequence_length]  # torch.Size([64, 40+8])
        x, (h, c) = self.LSTM(x.unsqueeze(2))  # x: torch.Size([64, 40, 32])
        out = self.linear(self.Tanh(x))
        return out  # torch.Size([64, 40+8-1, 1])


class ED_LSTM_1D(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(ED_LSTM_1D, self).__init__()
        self.model_name = 'ED_LSTM_1D'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model', self.model_name+'.pkl')
        self.args = args
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        # self.input_size = input_p_q_size + 5 + 3 + 3
        self.hidden_size = 64
        self.output_size = output_size
        self.LSTM = nn.LSTM(num_layers=3,
                            input_size=1,
                            hidden_size=self.hidden_size,
                            dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, input_time, aim_p, decoder_time):
        x = aim_p[:, 0:self.args.encoder_sequence_length]  # torch.Size([64, 40+8])
        x, (h, c) = self.LSTM(x.unsqueeze(2))  # x: torch.Size([64, 40, 32])
        out = self.linear(x)
        decode_in = out[:, -1:, :]
        out_i_list = [out]
        for i in range(0, self.args.decoder_sequence_length-1):
            out_i, (h, c) = self.LSTM(decode_in, (h[:, -1:, :].contiguous(), c[:, -1:, :].contiguous()))
            out_i = self.linear(out_i)
            decode_in = out_i
            out_i_list += [out_i]
        y = torch.cat(out_i_list, dim=1)
        return y  # torch.Size([64, 40+8-1, 1])


class ANN(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(ANN, self).__init__()
        self.model_name = 'ANN'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model', 'ANN.pkl')
        self.args = args
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        # self.input_size = input_p_q_size + 5 + 3 + 3
        self.hidden_size = 64
        self.output_size = output_size
        self.linear_layer = nn.Sequential(
            # nn.BatchNorm1d((input_p_q_size+3)*args.encoder_sequence_length),
            nn.Linear((input_p_q_size+3)*args.encoder_sequence_length, 256),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
            # output_size * args.decoder_sequence_length)
        )
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, input_time):
        x = torch.cat([input_p_q, input_time.float()], dim=2).view(input_p_q.shape[0], -1)
        y = self.linear_layer(x)
        return y  # torch.Size([64, 8])


class EDLSTM_nD(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(EDLSTM_nD, self).__init__()
        self.model_name = 'EDLSTM_nD'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        self.args = args
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.input_size = input_p_q_size + 5 + 3 + 4 + 7
        self.hidden_size = 64
        self.output_size = output_size
        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=2,
                               batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=output_size + 5 + 3 + 4 + 7,
                                        hidden_size=self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, output_size)

        self.embed_hour = nn.Embedding(24, 5)
        self.embed_weekday = nn.Embedding(8, 3)
        # self.embed_season = nn.Embedding(5, 3)
        self.embed_month = nn.Embedding(13, 4)
        self.embed_weekofyear = nn.Embedding(54, 7)
        self.Tanh = nn.Tanh()
        # self.BN = nn.BatchNorm1d(15)
        # print('BN')

    def forward(self, input_p_q, input_time, label_p, decoder_time):
        batch_size = input_p_q.shape[0]
        # sequence_length = input_p_q.shape[1]
        input_hour = self.embed_hour(input_time[:, :, 0]).squeeze(2)
        input_weekday = self.embed_weekday(input_time[:, :, 1]).squeeze(2)
        input_month = self.embed_month(input_time[:, :, 2]).squeeze(2)
        input_weekofyear = self.embed_weekofyear(input_time[:, :, 3]).squeeze(2)
        inputs = torch.cat([input_p_q, input_hour, input_weekday,
                            input_month, input_weekofyear], dim=2)
        output, (h_state, c_state) = self.encoder(inputs)
        # out, h_state = self.decoder(code)

        hi = h_state[-1]
        ci = c_state[-1]
        output_vector = torch.zeros(batch_size, self.args.decoder_sequence_length, self.output_size, device=device)
        # atte_score_2 = torch.zeros(batch_size, sequence_length, 1, device=device)
        # decode_input = decoder_p_last.unsqueeze(1)

        input_hour = self.embed_hour(decoder_time[:, :, 0]).squeeze(2)
        input_weekday = self.embed_weekday(decoder_time[:, :, 1]).squeeze(2)
        # input_season = self.embed_season(decoder_time[:, :, 2]).squeeze(2)
        input_month = self.embed_month(decoder_time[:, :, 2]).squeeze(2)
        input_weekofyear = self.embed_weekofyear(decoder_time[:, :, 3]).squeeze(2)
        decoder_p_first = label_p[:, self.args.encoder_sequence_length]\
            .squeeze()

        for i in range(0, self.args.decoder_sequence_length):
            decoder_inputs = torch.cat([decoder_p_first.unsqueeze(1), input_hour[:, i, :],
                                        input_weekday[:, i, :], input_month[:, i, :],
                                        input_weekofyear[:, i, :]], dim=1)
            hi, ci = self.decoder_cell(decoder_inputs, (hi, ci))
            output = self.out_layer(hi)
            decoder_p_first = output.squeeze()
            output_vector[:, i, :] = output # torch.Size([32, 30, 64])
        return output_vector
        # torch.Size([64, 10, 34]) = [batch, seq, fe


class SpatioTemporelLSTM(nn.Module):
    def __init__(self, input_p_q_size, config_size, output_size, hidden_size=64):

        super(SpatioTemporelLSTM, self).__init__()
        self.model_name = 'SpatioTemporelLSTM'
        self.model_path = './dnn_model/SpatioTemporelLSTM.pkl'
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.input_size = input_p_q_size + 5 + 10 + 3 + 3
        self.hidden_size = hidden_size
        self.embed_config = nn.Linear(config_size, 10)
        self.embed_hour = nn.Embedding(24, 5)
        self.embed_weekday = nn.Embedding(8, 3)
        self.embed_season = nn.Embedding(5, 3)
        self.Tanh = nn.Tanh()

        self.BN = nn.BatchNorm1d(15)
        # print('BN')

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            num_layers=1,
            hidden_size=1
        )
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_encoder = nn.LSTMCell(hidden_size, hidden_size).to(device)

        self.temperal_attention_score = nn.Linear(hidden_size * 3, 1)
        self.spatial_attention_score = nn.Linear(self.input_size + hidden_size * 2, self.input_size)
        self.att_score = None
        self.init()

    def init(self):
        gain = 1  # gain_original = 0.06 扩大了十倍
        # self.LSTMcell_spatial_encoder.data.xvaier_uniform_(gain)
        # self.LSTMcell_temporal_encoder.xvaier_uniform_(gain)
        self.temperal_attention_score.weight.data.uniform_(-gain, gain)
        self.temperal_attention_score.bias.data.zero_()
        a = torch.abs(self.spatial_attention_score.weight.data).mean()
        self.spatial_attention_score.weight.data.uniform_(-gain, gain)
        self.spatial_attention_score.bias.data.zero_()
        a2 = torch.abs(self.spatial_attention_score.weight.data).mean()
        b = 1

    def forward(self, input_p_q, input_config, input_time):
        batch_size = input_p_q.shape[0]
        sequence_length = input_p_q.shape[1]

        input_config = self.embed_config(input_config)  # torch.Size([64, 10, 11]) -> torch.Size([64, 10, 10])
        input_hour = self.embed_hour(input_time[:, :, 0]).squeeze(2)
        input_weekday = self.embed_weekday(input_time[:, :, 1]).squeeze(2)
        input_season = self.embed_season(input_time[:, :, 2]).squeeze(2)
        inputs = torch.cat([input_p_q, input_config, input_hour, input_weekday, input_season],
                           dim=2)  # torch.Size([64, 10, 34]) = [batch, seq, feat]

        # Spatial attention
        hi = torch.zeros(batch_size, self.hidden_size, device=device)
        ci = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, sequence_length, self.hidden_size, device=device)
        atte_score = torch.zeros(batch_size, sequence_length, self.input_size, device=device)
        for i in range(0, sequence_length):
            atte_score_i = self.Tanh(
                self.spatial_attention_score(torch.cat([hi, ci, inputs[:, i, :]], dim=1)))  # [B, 28]
            atte_score[:, i, :] = atte_score_i
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i)
            hi, ci = self.LSTMcell_spatial_encoder(inputs_i, (hi, ci))
            # inputs[:, i, :] torch.Size([32, 28])
            mid_output[:, i, :] = hi  # torch.Size([32, 30, 64])

        mid_output = self.BN(mid_output)

        # Temporal attention
        hi = torch.zeros(batch_size, self.hidden_size, device=device)
        ci = torch.zeros(batch_size, self.hidden_size, device=device)
        output = torch.zeros(batch_size, sequence_length, self.hidden_size, device=device)
        atte_score_2 = torch.zeros(batch_size, sequence_length, 1, device=device)
        for i in range(0, sequence_length):
            hi, ci = self.LSTMcell_temporal_encoder(mid_output[:, i, :], (hi, ci))
            atte_score_2[:, i, :] = self.Tanh(
                self.temperal_attention_score(torch.cat([hi, ci, mid_output[:, i, :]], dim=1)))
            output[:, i, :] = hi  # torch.Size([32, 30, 64])
        output_1d, h_state = self.decoder(output)
        out = torch.bmm(output_1d.transpose(1, 2), atte_score_2.softmax(dim=1))
        # torch.Size([32, 30, 1]) * torch.Size([32, 30, 1])
        return out


class RNN(nn.Module):
    def __init__(self, input_p_q_size, config_size, output_size, hidden_size=64):
        super(RNN, self).__init__()
        self.embed_config = nn.Linear(config_size, 10)
        self.embed_hour = nn.Embedding(24, 5)
        self.activ = nn.Tanh()
        # self.activ = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm1d(my_dataset.args_dataset.sequence_long)
        self.p_encoder = nn.RNN(
            input_size=input_p_q_size + 5 + 10,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True)
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(args.sequence_long),
            nn.Linear(hidden_size, int(hidden_size/2)),
            self.activ,
            nn.Linear(int(hidden_size/2), output_size),
        )

    def forward(self, input_p_q, input_config, input_time):
        input_config = self.embed_config(input_config)  # torch.Size([64, 10, 11]) -> torch.Size([64, 10, 10])
        input_time = self.embed_hour(input_time).squeeze(2)
        inputs = torch.cat([input_p_q, input_config, input_time], dim=2) # torch.Size([64, 10, 34])
        # inputs = self.BN(inputs)
        # input_p  torch.Size([1, 5, 12])
        # inputs_con  torch.Size([1, 11])
        # out = self.trans(inputs)
        output, h_state = self.p_encoder(inputs)  # torch.Size([64, 10, 34]) -> torch.Size([1, 5, 64])
        feature = output[:, -1, :]  # torch.Size([64, 1, 64])
        # feature = output[:, :, :]  # torch.Size([64, 10, 64])
        out = self.decoder(feature)
        return out


class Transfomer(nn.Module):
    def __init__(self, input_p_q_size, config_size, output_size, hidden_size=64):
        super(Transfomer, self).__init__()
        self.embed_config = nn.Linear(config_size, 10)
        self.embed_hour = nn.Embedding(24, 5)
        self.activ = nn.Tanh()
        # self.activ = nn.ReLU(inplace=True)
        # self.BN = nn.BatchNorm1d(args.sequence_long)
        self.trans = nn.Transformer(d_model=input_p_q_size + 5 + 10,
                                    nhead=4,
                                    dim_feedforward=256,
                                    num_encoder_layers=4,
                                    num_decoder_layers=4)
        # self.p_encoder = nn.LSTM(
        #     input_size=input_p_q_size + 5 + 10,
        #     hidden_size=hidden_size,
        #     num_layers=4,
        #     batch_first=True)
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(args.sequence_long),
            nn.Linear(input_p_q_size + 5 + 10, 1),
            # self.activ,
            # nn.Linear(int(hidden_size/2), output_size),
        )
        # self.temperal_attention_score = nn.Linear(hidden_size, 1)
        # self.att_score = None

    def forward(self, input_p_q, input_config, input_time):
        input_config = self.embed_config(input_config)  # torch.Size([64, 10, 11]) -> torch.Size([64, 10, 10])
        input_time = self.embed_hour(input_time).squeeze(2)
        inputs = torch.cat([input_p_q, input_config, input_time],
                           dim=2)  # torch.Size([64, 10, 34]) = [batch, seq, feat]
        src = inputs.transpose(0, 1)  # [seq=30, B=64, feat=34]
        tgt = inputs[:, -1, :].unsqueeze(1).transpose(0, 1)  # [seq=1, B=64, feat=34]
        output = self.trans(src, tgt)

        # output, h_state = self.p_encoder(inputs)

        # atte_score = self.temperal_attention_score(output)
        # # B * 10 * 1 = torch.Size([B, 10, 64] * 34 * 1
        # self.att_score = atte_score
        #
        # output_att = torch.bmm(atte_score.transpose(1, 2), output)
        out = self.decoder(output)
        return out.squeeze()


class SpatioTemporelLSTM_v2(nn.Module):
    def __init__(self, input_p_q_size, config_size, output_size, hidden_size=64):
        super(SpatioTemporelLSTM_v2, self).__init__()
        self.model_name = 'SpatioTemporelLSTM'
        self.model_path = './dnn_model/SpatioTemporelLSTM.pkl'
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.input_size = input_p_q_size + 5 + 10 + 3 + 3
        self.hidden_size = hidden_size
        self.embed_config = nn.Linear(config_size, 10)
        self.embed_hour = nn.Embedding(24, 5)
        self.embed_weekday = nn.Embedding(8, 3)
        self.embed_season = nn.Embedding(5, 3)
        self.Tanh = nn.Tanh()
        # self.BN = nn.BatchNorm1d(15)
        # print('BN')
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(hidden_size+1, 1).to(device)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, 1)

        self.temporal_attention_Wd = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wd2 = nn.Linear(2, self.hidden_size, bias=False)
        self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.att_score = None
#        self.init()

    def forward(self, input_p_q, input_config, input_time):
        batch_size = input_p_q.shape[0]
        sequence_length_encode = input_p_q.shape[1]
        sequence_length_decoder = 16

        input_config = self.embed_config(input_config)  # torch.Size([64, 10, 11]) -> torch.Size([64, 10, 10])
        input_hour = self.embed_hour(input_time[:, :, 0]).squeeze(2)
        input_weekday = self.embed_weekday(input_time[:, :, 1]).squeeze(2)
        input_season = self.embed_season(input_time[:, :, 2]).squeeze(2)
        inputs = torch.cat([input_p_q, input_config, input_hour, input_weekday, input_season],
                           dim=2)  # torch.Size([64, 10, 34]) = [batch, seq, feat]

        # Spatial attention
        hi = torch.zeros(batch_size, self.hidden_size, device=device)
        ci = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, sequence_length_encode, self.hidden_size, device=device)
        atte_score = torch.zeros(batch_size, sequence_length_encode, self.input_size, device=device)
        for i_decode in range(0, sequence_length_encode):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([hi, ci], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i_decode, :])
                ))  # [B, 28]
            atte_score[:, i_decode, :] = atte_score_i
            inputs_i = torch.mul(inputs[:, i_decode, :], atte_score_i)
            hi, ci = self.LSTMcell_spatial_encoder(inputs_i, (hi, ci))
            # inputs[:, i, :] torch.Size([32, 28])
            mid_output[:, i_decode, :] = hi  # torch.Size([32, 30, 64])

        # mid_output = self.BN(mid_output)

        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, 1, device=device)
        ci = torch.zeros(batch_size, 1, device=device)
        decode_output = torch.zeros(batch_size, sequence_length_decoder, 1, device=device)
        inputs_before = mid_output[:, -1, 2].unsqueeze(1)

        atte_score_2 = torch.zeros(batch_size, sequence_length_encode, sequence_length_decoder, device=device)  # [B, 30, 16]
        for i_decode in range(0, sequence_length_decoder):
            for i_encode in range(0, sequence_length_encode):
                # hi, ci = self.LSTMcell_temporal_decoder(mid_output[:, i, :], (hi, ci))
                atte_score_2[:, i_encode, i_decode] = torch.squeeze(
                    self.temporal_attention_Vd(
                    self.Tanh(
                        self.temporal_attention_Wd2(torch.cat([hi, ci], dim=1)) +
                        self.temporal_attention_Wd(mid_output[:, i_encode, :])
                )))
            atte_score_22 = atte_score_2[:, :, i_decode]\
                .repeat(mid_output.size(2), 1, 1).permute(1, 2, 0)
            inputs_mid = torch.mul(mid_output, atte_score_22)  # [8,15,64]
            hi, ci = self.LSTMcell_temporal_decoder(
                torch.cat([inputs_mid.sum(dim=1), inputs_before], dim=1),
                (hi, ci))
            inputs_before = hi
            decode_output[:, i_decode, :] = hi  # torch.Size([32, 30, 1])
        # output_1d, h_state = self.decoder(output)
        # out = torch.bmm(output.transpose(1, 2), atte_score_2.softmax(dim=1))
        # torch.Size([32, 30, 1]) * torch.Size([32, 30, 1])
        return decode_output



from sklearn.model_selection import train_test_split
#
# class RnnAtt(nn.Module):
#     def __init__(self, input_p_q_size, config_size, output_size, hidden_size=64):
#         self.model_name = 'RnnAtt'
#         self.model_path = './dnn_model/RnnAtt.pkl'
#         print('model_name:', self.model_name, 'saved at:', self.model_path)
#         super(RnnAtt, self).__init__()
#         self.input_size = input_p_q_size + 5 + 10
#         self.hidden_size = hidden_size
#         self.embed_config = nn.Linear(config_size, 10)
#         self.embed_hour = nn.Embedding(24, 5)
#         # self.embed_weekday = nn.Embedding(8, 3)
#         # self.embed_season = nn.Embedding(5, 3)
#         self.Tanh = nn.Tanh()
#         # self.activ = nn.ReLU(inplace=True)
#         # self.BN = nn.BatchNorm1d(args.sequence_long)
#         # self.trans = nn.Transformer()
#         # self.LSTM_encoder = nn.LSTM(
#         #     input_size=input_p_q_size + 5 + 10,
#         #     hidden_size=hidden_size,
#         #     num_layers=3,
#         #     batch_first=True)
#         self.decoder = nn.LSTM(
#             input_size=hidden_size,
#             num_layers=1,
#             hidden_size=1
#         )
#         self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
#         self.LSTMcell_temporal_encoder = nn.LSTMCell(hidden_size, hidden_size).to(device)
#
#         self.temperal_attention_score = nn.Linear(hidden_size * 3, 1)
#         self.spatial_attention_score = nn.Linear(self.input_size + hidden_size * 2, self.input_size)
#         self.att_score = None
#
#     def forward(self, input_p_q, input_config, input_time):
#         batch_size = input_p_q.shape[0]
#         sequence_length = input_p_q.shape[1]
#         input_config = self.embed_config(input_config)  # torch.Size([64, 10, 11]) -> torch.Size([64, 10, 10])
#         input_hour = self.embed_hour(input_time[:, :, 0]).squeeze(2)
#         # input_weekday = self.embed_weekday(input_time[:, :, 1]).squeeze(2)
#         # input_season = self.embed_season(input_time[:, :, 2]).squeeze(2)
#         inputs = torch.cat([input_p_q, input_config, input_hour],
#                            dim=2)  # torch.Size([64, 10, 34]) = [batch, seq, feat]
#
#         # Spatial attention
#         hi = torch.zeros(batch_size, self.hidden_size, device=device)
#         ci = torch.zeros(batch_size, self.hidden_size, device=device)
#         mid_output = torch.zeros(batch_size, sequence_length, self.hidden_size, device=device)
#         atte_score = torch.zeros(batch_size, sequence_length, self.input_size, device=device)
#         for i in range(0, sequence_length):
#             atte_score_i = self.Tanh(self.spatial_attention_score(torch.cat([hi, ci, inputs[:, i, :]], dim=1)))  # [B, 28]
#             atte_score[:, i, :] = atte_score_i
#             inputs_i = torch.mul(inputs[:, i, :], atte_score_i)
#             hi, ci = self.LSTMcell_spatial_encoder(inputs_i, (hi, ci))
#             # inputs[:, i, :] torch.Size([32, 28])
#             mid_output[:, i, :] = hi  # torch.Size([32, 30, 64])
#         # mid_output = self.BN(mid_output)
#
#         # Temporal attention
#         hi = torch.zeros(batch_size, self.hidden_size, device=device)
#         ci = torch.zeros(batch_size, self.hidden_size, device=device)
#         output = torch.zeros(batch_size, sequence_length, self.hidden_size, device=device)
#         atte_score_2 = torch.zeros(batch_size, sequence_length, 1, device=device)
#         for i in range(0, sequence_length):
#             hi, ci = self.LSTMcell_temporal_encoder(mid_output[:, i, :], (hi, ci))
#             atte_score_2[:, i, :] = self.Tanh(self.temperal_attention_score(torch.cat([hi, ci, mid_output[:, i, :]], dim=1)))
#             output[:, i, :] = hi  # torch.Size([32, 30, 64])
#         output_1d, h_state = self.decoder(output)
#         out = torch.bmm(output_1d.transpose(1, 2), atte_score_2.softmax(dim=1))
#         # torch.Size([32, 30, 1]) * torch.Size([32, 30, 1])
#         return out