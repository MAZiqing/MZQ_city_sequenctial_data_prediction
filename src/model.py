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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class hDS_RNN(nn.Module):
    def __init__(self, input_size, args):
        '''
        :param input_size: int, input size
        :param args: global variables
        '''
        super(hDS_RNN, self).__init__()
        self.model_name = 'hDS_RNN'
        self.details = 'None'

        # hyper-parameters of the model
        self.n_inp = input_size
        self.n_hid = args.num_hidden_state
        self.n_hid_deco = self.n_hid
        self.T_enco = args.encoder_sequence_length
        self.m_enco = args.num_hidden_state
        self.T_deco = args.decoder_sequence_length

        # spatial attention weights
        self.Ue = nn.Linear(self.T_enco, self.m_enco)
        self.Ue_ = nn.Linear(self.n_inp, self.m_enco)
        self.We = nn.Linear(self.n_hid * 2, self.m_enco)
        self.Ve = nn.Linear(self.m_enco, 1)

        # temporal attention weights
        self.Ud = nn.Linear(self.n_hid, self.n_hid)
        self.Wd = nn.Linear(2 * self.n_hid_deco, self.n_hid)
        self.Vd = nn.Linear(self.n_hid, 1)

        # LSTM layer of encoder, decoder and a middle layer
        self.lstm_s_encoder = nn.LSTMCell(self.n_inp,
                                          self.n_hid).to(device)
        self.lstm_t_decoder = nn.LSTMCell(self.n_hid_deco,
                                          self.n_hid_deco).to(device)
        self.LSTM_mid = nn.LSTM(input_size=self.n_hid,
                                hidden_size=self.n_hid,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        # regressor of the output layer
        self.regressor = nn.Linear(self.n_hid_deco, 1)
        self.Tanh = nn.Tanh()

        self.spatial_att_score_list = []
        self.temporal_att_score_list = []

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        # get the inputs for the encoder
        inputs = input_p_q[:, :self.T_enco, :]

        # # Spatial attention
        h = torch.zeros(batch_size, self.n_hid, device=device)
        c = torch.zeros(batch_size, self.n_hid, device=device)
        self.spatial_att_score_list = []
        mid_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        for i in range(0, self.T_enco):
            s_att_score = self.Ve(
                self.Tanh(
                    self.We(torch.cat([h, c], dim=1)).
                    repeat(self.n_inp, 1, 1).permute(1, 0, 2) +
                    self.Ue(inputs.transpose(1, 2)) +
                    self.Ue_(inputs[:, i, :])
                    .repeat(self.n_inp, 1, 1).permute(1, 0, 2)
                )).squeeze()
            self.spatial_att_score_list += [s_att_score.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], s_att_score.softmax(dim=1))
            h, c = self.lstm_s_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h

        # mid layer
        mid_output, hh = self.LSTM_mid(mid_output)

        # Temporal attention
        hi = torch.zeros(batch_size, self.n_hid_deco, device=device)
        ci = torch.zeros(batch_size, self.n_hid_deco, device=device)
        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, self.T_deco+6):
            t_att_score_ = self.Wd(torch.cat([hi, ci], dim=1)).\
                               repeat(self.T_enco, 1, 1).transpose(0, 1) + \
                               self.Ud(mid_output)
            t_att_score = self.Vd(self.Tanh(t_att_score_))
            self.temporal_att_score_list += [t_att_score.transpose(1, 2).softmax(dim=2)]
            decoder_in = torch.bmm(t_att_score.transpose(1, 2), mid_output)
            hi, ci = self.lstm_t_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        # output regressor
        out = torch.cat(decode_output, dim=1)
        out = self.regressor(out)
        out = out[:, -self.T_deco:, :].squeeze()
        return out


class DS_RNN(nn.Module):
    def __init__(self, n_inp, args):
        '''
        :param input_size: int, input size
        :param args: global variables
        '''
        # A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
        super(DS_RNN, self).__init__()
        self.model_name = 'DS_RNN'
        self.details = 'None'

        # hyper-parameters of the model
        self.n_inp = n_inp
        self.n_hid = args.num_hidden_state
        self.n_hid_deco = self.n_hid
        self.T_enco = args.encoder_sequence_length
        self.T_deco = args.decoder_sequence_length

        # spatial attention weights
        self.Ue = nn.Linear(self.T_enco, self.T_enco)
        self.We = nn.Linear(self.n_hid * 2, self.T_enco, bias=False)
        self.Ve = nn.Linear(self.T_enco, 1)

        # temporal attention weights
        self.Ud = nn.Linear(self.n_hid, self.n_hid)
        self.Wd = nn.Linear(2 * self.n_hid_deco, self.n_hid, bias=False)
        self.Vd = nn.Linear(self.n_hid, 1)

        # LSTM layer of encoder and decoder
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.n_inp,
                                                    self.n_hid).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.n_hid_deco,
                                                     self.n_hid_deco).to(device)

        # regressor of the output layer
        self.regressor = nn.Linear(self.n_hid_deco, 1)
        self.Tanh = nn.Tanh()

        self.spatial_att_score_list = []
        self.temporal_att_score_list = []

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        # get inputs for the encoder
        inputs = input_p_q[:, :self.T_enco, :]

        # # Spatial attention
        h = torch.zeros(batch_size, self.n_hid, device=device)
        c = torch.zeros(batch_size, self.n_hid, device=device)
        self.spatial_att_score_list = []
        mid_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        for i in range(0, self.T_enco):
            atte_score_i = self.Ve(
                self.Tanh(
                    self.We(torch.cat([h, c], dim=1)).
                    repeat(self.n_inp, 1, 1).permute(1, 0, 2) +
                    self.Ue(inputs.transpose(1, 2))
                )).squeeze()
            self.spatial_att_score_list += [atte_score_i.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h

        # Temporal attention
        hi = torch.zeros(batch_size, self.n_hid_deco, device=device)
        ci = torch.zeros(batch_size, self.n_hid_deco, device=device)

        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, self.T_deco):
            atte_score_2_x = self.Wd(torch.cat([hi, ci], dim=1)).repeat(self.T_enco, 1, 1).transpose(0, 1) + \
                             self.Ud(mid_output)
            atte_score_2 = self.Vd(self.Tanh(atte_score_2_x))
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(dim=2)]
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2).softmax(dim=2), mid_output)
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = self.regressor(torch.cat(decode_output, dim=1))
        out = out[:, -self.T_deco:, :].squeeze()
        return out


class DS_RNN_II(nn.Module):
    def __init__(self, input_size, args):
        '''
        :param input_size: int, input size
        :param args: global variables
        '''
        # Modified form [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction]
        super(DS_RNN_II, self).__init__()
        self.model_name = 'DS_RNN_II'
        self.details = 'None'

        # hyper-parameters of the model
        self.args = args
        self.n_inp = input_size
        self.n_hid = args.num_hidden_state
        self.T_enco = args.encoder_sequence_length
        self.T_deco = args.decoder_sequence_length

        # spatial attention weights
        self.Ue = nn.Linear(self.n_inp, self.n_hid)
        self.We = nn.Linear(self.n_hid * 2, self.n_hid, bias=False)
        self.Ve = nn.Linear(self.n_hid, self.n_inp)

        # temporal attention weights
        self.Ud = nn.Linear(self.n_hid, self.n_hid)
        self.Wd = nn.Linear(2 * self.n_hid, self.n_hid, bias=False)
        self.Vd = nn.Linear(self.n_hid, 1)

        # LSTM layer of the encoder, decoder and mid layer
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.n_inp, self.n_hid).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.n_hid, self.n_hid).to(device)
        self.LSTM_mid = nn.LSTM(input_size=self.n_hid,
                                hidden_size=self.n_hid,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        # regressor of the output layer
        self.regressor = nn.Linear(self.n_hid, 1)
        self.Tanh = nn.Tanh()

        self.spatial_att_score_list = []
        self.temporal_att_score_list = []

    def forward(self, inputs, label_p):
        batch_size = inputs.shape[0]
        # get inputs for the encoder
        inputs = inputs[:, :self.T_enco, :]

        # # Spatial attention
        h = torch.zeros(batch_size, self.n_hid, device=device)
        c = torch.zeros(batch_size, self.n_hid, device=device)
        mid_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        self.spatial_att_score_list = []
        for i in range(0, self.T_enco):
            atte_score_i = self.Ve(
                self.Tanh(
                    self.We(torch.cat([h, c], dim=1)) +
                    self.Ue(inputs[:, i, :])
                ))
            self.spatial_att_score_list += [atte_score_i.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h

        # middle layer
        mid_output, hh = self.LSTM_mid(mid_output)

        # Temporal attention
        hi = torch.zeros(batch_size, self.n_hid, device=device)
        ci = torch.zeros(batch_size, self.n_hid, device=device)
        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, self.T_deco+4):
            atte_score_2_x = self.Wd(torch.cat([hi, ci], dim=1)).repeat(self.T_deco, 1, 1).transpose(0, 1) + \
                             self.Ud(mid_output)
            atte_score_2 = self.Vd(self.Tanh(atte_score_2_x))
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(dim=2)]
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output)
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]
        out = torch.cat(decode_output, dim=1)
        out = self.regressor(out)
        out = out[:, -self.T_deco:, :]
        return out.squeeze()


class DSTP_RNN(nn.Module):
    def __init__(self, input_size, args, output_column):
        # DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks
        # for long-term and multivariate time series prediction
        super(DSTP_RNN, self).__init__()
        self.model_name = 'DSTP_RNN'
        self.details = 'None'
        self.output_column = output_column

        # hyper-parameters of the model
        self.n_inp = input_size
        self.n_hid = args.num_hidden_state
        self.T_enco = args.encoder_sequence_length
        self.T_deco = args.decoder_sequence_length

        # spatial attention weights stage 1
        self.Ue1 = nn.Linear(self.T_enco, self.T_enco)
        self.We1 = nn.Linear(self.n_hid * 2, self.T_enco, bias=False)
        self.Ve1 = nn.Linear(self.T_enco, 1)

        # spatial attention weights stage 2
        self.Ue2 = nn.Linear(self.T_enco, self.T_enco)
        self.We2 = nn.Linear(self.n_hid * 2, self.T_enco, bias=False)
        self.Ve2 = nn.Linear(self.T_enco, 1)

        # temporal attention weights
        self.Ud = nn.Linear(self.n_hid, self.n_hid)
        self.Wd = nn.Linear(2 * self.n_hid, self.n_hid, bias=False)
        self.Vd = nn.Linear(self.n_hid, 1)

        # LSTM layer of encoder and decoder
        self.encoder1 = nn.LSTMCell(self.n_inp - 1,
                                    self.n_hid).to(device)
        self.encoder2 = nn.LSTMCell(self.n_hid + 1,
                                    self.n_hid).to(device)
        self.decoder = nn.LSTMCell(self.n_hid,
                                   self.n_hid).to(device)

        # regressor
        self.regressor = nn.Linear(self.n_hid, 1)
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        # get inputs without target series
        inputs = input_p_q[:, :self.T_enco, list(range(0, self.output_column)) + list(range(self.output_column+1, 18))]
        labels_p = label_p[:, :self.T_enco]

        # # Spatial attention phase 1
        h1 = torch.zeros(batch_size, self.n_hid, device=device)
        c1 = torch.zeros(batch_size, self.n_hid, device=device)
        mid_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        for i in range(0, self.T_enco):
            atte_score_i = self.Ve1(
                self.Tanh(
                    self.We1(torch.cat([h1, c1], dim=1)).
                    repeat(self.n_inp - 1, 1, 1).permute(1, 0, 2) +
                    self.Ue1(inputs.transpose(1, 2))
                )).squeeze()
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h1, c1 = self.encoder1(inputs_i, (h1, c1))
            mid_output[:, i, :] = h1

        # Spatial attention phase 2
        mid_output = torch.cat([mid_output, labels_p.unsqueeze(2)], dim=2)
        h = torch.zeros(batch_size, self.n_hid, device=device)
        c = torch.zeros(batch_size, self.n_hid, device=device)
        final_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        for i in range(0, self.T_enco):
            atte_score_i = self.Ve2(
                self.Tanh(
                    self.We2(torch.cat([h, c], dim=1)).
                    repeat(self.n_hid + 1, 1, 1).permute(1, 0, 2) +
                    self.Ue2(mid_output.transpose(1, 2))
                )).squeeze()
            inputs_i2 = torch.mul(mid_output[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.encoder2(inputs_i2, (h, c))
            final_output[:, i, :] = h

        # Temporal attention
        hi = torch.zeros(batch_size, self.h_hid, device=device)
        ci = torch.zeros(batch_size, self.h_hid, device=device)
        decode_output = []
        for i_decoder in range(0, self.T_deco+6):
            atte_score_2_x = self.Wd(torch.cat([hi, ci], dim=1)).repeat(self.T_enco, 1, 1).transpose(0, 1) + \
                             self.Ud(final_output)
            atte_score_2 = self.Vd(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2).softmax(dim=2), final_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        # regressor
        out = torch.cat(decode_output, dim=1)
        out = self.regressor(out)
        out = out[:, -self.T_deco:, :]
        return out.squeeze()


class ANN(nn.Module):
    def __init__(self, input_size, args):
        super(ANN, self).__init__()
        self.model_name = 'ANN_nD'
        self.T_enco = args.encoder_sequence_length
        self.hidden_size = args.num_hidden_state
        self.linear_layer = nn.Sequential(
            nn.Linear(input_size * args.encoder_sequence_length, self.hidden_size * 4),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hidden_size, args.decoder_sequence_length)
        )
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, targets):
        inputs = input_p_q[:, :self.T_enco, :]
        x = inputs.view(inputs.shape[0], -1)
        y = self.linear_layer(x)
        return y


class Seq2Seq(nn.Module):
    # Encoder-Decoder LSTM for multivariate time series (n-dimension, no time feature needed)
    def __init__(self, input_size, args):
        super(Seq2Seq, self).__init__()
        self.model_name = 'Seq2Seq'
        self.T_enco = args.encoder_sequence_length
        self.T_deco = args.decoder_sequence_length
        self.Tanh = nn.Tanh()
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=args.num_hidden_state,
                               num_layers=args.num_layer,
                               batch_first=True,
                               dropout=0.5)
        self.decoder_cell = nn.LSTMCell(input_size=1,
                                        hidden_size=args.num_hidden_state)
        self.out_layer = nn.Linear(args.num_hidden_state, 1)

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        output, (h_state, c_state) = self.encoder(inputs)
        h = h_state[-1]
        c = c_state[-1]
        output_vector = torch.zeros(batch_size, self.T_deco, self.n_out, device=device)
        decoder_p_first = targets[:, self.T_enco - 1].unsqueeze(1)
        for i in range(0, self.T_deco):
            decoder_inputs = decoder_p_first
            h, c = self.decoder_cell(decoder_inputs, (h, c))
            output = self.out_layer(h)
            decoder_p_first = output
            output_vector[:, i, :] = output
        return output_vector


class Transformer(nn.Module):
    def __init__(self, input_p_q_size, args):
        super(Transformer, self).__init__()
        self.model_name = 'transformer'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.activ = nn.Tanh()
        self.src_mask = None
        self.tgt_mask = None
        self.encoder_in = nn.Linear(input_p_q_size, args.num_hidden_state)
        self.transformer = nn.Transformer(d_model=args.num_hidden_state,
                                          nhead=8,
                                          num_encoder_layers=args.num_layer,
                                          num_decoder_layers=args.num_layer,
                                          dim_feedforward=args.num_hidden_state,
                                          dropout=0.1)
        self.decoder_in = nn.Linear(1, args.num_hidden_state)
        self.decoder = nn.Sequential(
            nn.Linear(args.num_hidden_state, 1))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_p_q, label_p):
        src = input_p_q[:, :-self.args.decoder_sequence_length, :].transpose(0, 1)
        tgt = label_p[:, -self.args.decoder_sequence_length - 1: -1].unsqueeze(2).transpose(0, 1)

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            device = tgt.device
            self.tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
        src = self.encoder_in(src)
        tgt = self.decoder_in(tgt)
        out = self.transformer(src, tgt, self.src_mask, self.tgt_mask)
        out = self.decoder(out).transpose(0, 1)
        return out.squeeze()
