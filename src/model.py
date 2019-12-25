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


class ED_LSTM_1D(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(ED_LSTM_1D, self).__init__()
        self.model_name = 'ED_LSTM_1D'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model', self.model_name + '.pkl')
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
        for i in range(0, self.args.decoder_sequence_length - 1):
            out_i, (h, c) = self.LSTM(decode_in, (h[:, -1:, :].contiguous(), c[:, -1:, :].contiguous()))
            out_i = self.linear(out_i)
            decode_in = out_i
            out_i_list += [out_i]
        y = torch.cat(out_i_list, dim=1)
        return y  # torch.Size([64, 40+8-1, 1])


class ANN_nD(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(ANN_nD, self).__init__()
        self.model_name = 'ANN_nD'
        self.output_column = output_column
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        self.args = args
        self.hidden_size = args.num_hidden_state
        self.output_size = output_size
        self.linear_layer = nn.Sequential(
            # nn.BatchNorm1d((input_p_q_size+3)*args.encoder_sequence_length),
            nn.Linear(input_p_q_size * args.encoder_sequence_length, self.hidden_size * 4),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Tanh(),

            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Tanh(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Tanh(),

            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Tanh(),
            nn.Linear(self.hidden_size, 4)
        )
        self.Tanh = nn.Tanh()

    def forward(self, input_p_q, input_time):
        inputs = input_p_q
        x = inputs.view(input_p_q.shape[0], -1)
        y = self.linear_layer(x)
        return y  # torch.Size([64, 8])


class EDLSTM(nn.Module):
    # Encoder-Decoder LSTM for multivariate time series (n-dimension, no time feature needed)
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(EDLSTM, self).__init__()
        self.model_name = 'EDLSTM'
        self.n_out = output_size
        self.T_enco = args.encoder_sequence_length
        self.T_deco = args.decoder_sequence_length
        self.Tanh = nn.Tanh()
        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=args.num_hidden_state,
                               num_layers=args.num_layer,
                               batch_first=True,
                               dropout=0.5)
        self.decoder_cell = nn.LSTMCell(input_size=output_size,
                                        hidden_size=args.num_hidden_state)
        self.out_layer = nn.Linear(args.num_hidden_state, output_size)

    def forward(self, input_p_q, input_time, label_p, decoder_time):
        batch_size = input_p_q.shape[0]
        output, (h_state, c_state) = self.encoder(input_p_q)
        h = h_state[-1]
        c = c_state[-1]
        output_vector = torch.zeros(batch_size, self.T_deco, self.n_out, device=device)
        decoder_p_first = label_p[:, self.T_enco - 1].unsqueeze(1)
        for i in range(0, self.T_deco):
            decoder_inputs = decoder_p_first
            h, c = self.decoder_cell(decoder_inputs, (h, c))
            output = self.out_layer(h)
            decoder_p_first = output
            output_vector[:, i, :] = output  # torch.Size([32, 30, 64])
        return output_vector


class Transformer(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_columns, args):
        super(Transformer, self).__init__()
        self.model_name = 'transformer'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.activ = nn.Tanh()
        self.src_mask = None
        self.tgt_mask = None
        # encoder_layer = nn.TransformerEncoderLayer(d_model=input_p_q_size, nhead=2,
        #                                            dim_feedforward=args.num_hidden_state,
        # dropout=0.1)
        # self.trans = nn.TransformerEncoder(encoder_layer,
        #                                    num_layers=4)
        self.encoder_in = nn.Linear(input_p_q_size, args.num_hidden_state)
        self.transformer = nn.Transformer(d_model=args.num_hidden_state,
                                          nhead=8,
                                          num_encoder_layers=args.num_layer,
                                          num_decoder_layers=args.num_layer,
                                          dim_feedforward=args.num_hidden_state,
                                          dropout=0.1)
        self.decoder_in = nn.Linear(output_size, args.num_hidden_state)
        self.decoder = nn.Sequential(
            nn.Linear(args.num_hidden_state, 1))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_p_q, label_p):
        src = input_p_q[:, :-self.args.decoder_sequence_length, :].transpose(0, 1)
        # [40 * B * 18]
        tgt = label_p[:, -self.args.decoder_sequence_length - 1: -1].unsqueeze(2).transpose(0, 1)
        # [4 * B * 1]
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            device = tgt.device
            self.tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
        src = self.encoder_in(src)
        tgt = self.decoder_in(tgt)
        out = self.transformer(src, tgt, self.src_mask, self.tgt_mask)
        out = self.decoder(out).transpose(0, 1)
        return out.squeeze()


class STAttention(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(STAttention, self).__init__()
        self.model_name = 'STAttention'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size + 5 + 3 + 4 + 7
        hidden_size = 64
        self.hidden_size = hidden_size
        self.embed_hour = nn.Embedding(24, 5)
        self.embed_weekday = nn.Embedding(8, 3)
        self.embed_month = nn.Embedding(13, 4)
        self.embed_weekofyear = nn.Embedding(54, 7)
        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(hidden_size + 1, 1).to(device)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)

        self.temporal_attention_Wd = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wd2 = nn.Linear(2, self.hidden_size, bias=False)
        self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.att_score = None

    def forward(self, input_p_q, input_time, label_p, decoder_time):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length

        input_hour = self.embed_hour(input_time[:, :, 0]).squeeze(2)
        input_weekday = self.embed_weekday(input_time[:, :, 1]).squeeze(2)
        input_month = self.embed_month(input_time[:, :, 2]).squeeze(2)
        input_weekofyear = self.embed_weekofyear(input_time[:, :, 3]).squeeze(2)
        inputs = torch.cat([input_p_q, input_hour, input_weekday,
                            input_month, input_weekofyear], dim=2)
        # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        # atte_score = torch.zeros(batch_size, sequence_length_encoder, self.input_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i, :])
                ))
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, 1, device=device)
        ci = torch.zeros(batch_size, 1, device=device)

        decode_output = []
        decoder_p_in = label_p[:, len_enco - 1].unsqueeze(1)
        for i_decoder in range(0, len_deco):
            atte_score_2 = self.temporal_attention_Vd(
                self.Tanh(
                    self.temporal_attention_Wd2(torch.cat([hi, ci], dim=1))
                    .repeat(40, 1, 1).transpose(0, 1) +
                    self.temporal_attention_Wd(mid_output)))
            # torch.Size([32, 40, 1])
            hi, ci = self.LSTMcell_temporal_decoder(
                torch.cat([
                    torch.bmm(atte_score_2.transpose(1, 2), mid_output).squeeze(),
                    # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
                    decoder_p_in], dim=1), (hi, ci))
            decode_output += [hi]
        return torch.cat(decode_output, dim=1)


class STAttention_notf_v4(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        # 对mid_output直接全连接
        super(STAttention_notf_v4, self).__init__()
        self.model_name = 'STAttention_notf'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        hidden_size = args.num_hidden_state
        self.hidden_size = hidden_size
        self.decoder_hidden_state = self.hidden_size

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state, self.decoder_hidden_state).to(device)

        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)
        self.att_score = None

        self.linear1 = nn.Linear(self.hidden_size, 1)
        self.linear2 = nn.Linear(args.encoder_sequence_length,
                                 args.decoder_sequence_length)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        inputs = input_p_q

        # # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i, :])
                ))
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        mid_output, hh = self.LSTM_mid(mid_output)
        # [B * 40 * 18]
        out = self.Tanh(self.linear1(mid_output))
        out = self.linear2(out.squeeze())
        return out.squeeze()


class STAttention_notf_v5(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        # v5 的特点，将mid_output 以hi和ci的方式加入
        super(STAttention_notf_v5, self).__init__()
        self.model_name = 'STAttention_notf'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        hidden_size = args.num_hidden_state
        self.hidden_size = hidden_size
        self.decoder_hidden_state = self.hidden_size

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state, self.decoder_hidden_state).to(device)

        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.LSTM_out = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        # self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        self.att_score = None

        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        inputs = input_p_q

        # # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i, :])
                ))
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        mid_output, (hi, ci) = self.LSTM_mid(mid_output)
        hi = hi.squeeze()
        ci = ci.squeeze()
        # Temporal attention 和论文里的一毛一样了

        decode_output = []
        # decoder_p_in = label_p[:, len_enco-1].unsqueeze(1)
        for i_decoder in range(0, len_deco):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                                     1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = torch.cat(decode_output, dim=1)
        # out, hhh = self.LSTM_out(out)
        out = self.regressor_2(out)
        return out.squeeze()


class STAttention_notf_v6(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        # v6 的特点，将decoder从4延长到44，以mid_output
        super(STAttention_notf_v6, self).__init__()
        self.model_name = 'STAttention_notf_i=20'
        self.model_path = os.path.join('../rnn_model', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        hidden_size = args.num_hidden_state
        self.hidden_size = hidden_size
        self.decoder_hidden_state = self.hidden_size

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state + 1, self.decoder_hidden_state).to(
            device)

        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.LSTM_out = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        # self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        # self.spatial_att_score_list = []
        # self.temporal_att_score = []

        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        inputs = input_p_q

        # # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i, :])
                ))
            # self.temporal_att_score += [atte_score_i.softmax(dim=1)]
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            # self.spatial_att_score_list += [atte_score_i.softmax(dim=1)]
            # [B * 18] = [B * 18] * [B * 18]
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        mid_output, (hi, ci) = self.LSTM_mid(mid_output)
        # hi = hi.squeeze()
        # ci = ci.squeeze()

        hi = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        ci = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        # Temporal attention 和论文里的一毛一样了

        decode_output = []
        i = 20
        for i_decoder in range(len_deco + len_enco-i, len_deco + len_enco):
        # for i_decoder in range(0, len_deco):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                            1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            # decoder_in = torch.cat([decoder_in.squeeze(), label_p[:, len_enco + i_decoder - 1].unsqueeze(1)], dim=1)
            decoder_in = torch.cat([decoder_in.squeeze(), label_p[:, i_decoder].unsqueeze(1)], dim=1)
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in, (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        # out = torch.cat(decode_output, dim=1)
        out = torch.cat(decode_output[-len_deco-1: -1], dim=1)
        out = self.regressor_2(out)
        return out.squeeze()


class STAttention_notf_v7(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        # v7 的特点，将decoder从4延长到44，以mid_output
        # a dynamic global attention
        super(STAttention_notf_v7, self).__init__()
        self.model_name = 'STAttention_notf_i=10'
        self.model_path = os.path.join('../src', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state + 1, self.decoder_hidden_state).to(
            device)
        self.LSTMcell_target_encoder = nn.LSTMCell(output_size+self.hidden_size, self.hidden_size).to(device)

        self.LSTM_target = nn.LSTM(input_size=output_size+self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0.5)
        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.LSTM_out = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Wt = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        # self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []
        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []
        inputs = input_p_q
        look_back = 10

        # # Spatial attention
        ht = torch.zeros(batch_size, self.hidden_size, device=device)
        ct = torch.zeros(batch_size, self.hidden_size, device=device)
        decoder_in = torch.zeros(batch_size, self.hidden_size, device=device)
        label_p_i = label_p[:, len_deco + len_enco-look_back]
        # [B]
        target_list = []
        self.spatial_att_score_list = []
        for it in range(len_deco + len_enco-look_back, len_deco + len_enco):
            x = torch.cat([label_p_i.unsqueeze(1), decoder_in], dim=1)
            ht, ct = self.LSTMcell_target_encoder(x, (ht, ct))
            # [B * 64]
            target = self.regressor_2(ht)
            if it < len_enco:
                label_p_i = label_p[:, it+1]
            else:
                target_list += [target]
                label_p_i = target.squeeze()
            he = torch.zeros(batch_size, self.hidden_size, device=device)
            ce = torch.zeros(batch_size, self.hidden_size, device=device)
            mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
            for i in range(0, len_enco):
                atte_score_i = self.spatial_attention_Vd(
                    self.Tanh(
                        self.spatial_attention_We(torch.cat([he, ce], dim=1)) +
                        self.spatial_attention_Wi(inputs[:, i, :]) +
                        self.spatial_attention_Wt(torch.cat([ht, ct], dim=1))
                        # [B * 64]
                    ))
                # self.temporal_att_score += [atte_score_i.softmax(dim=1)]
                inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
                self.spatial_att_score_list += [atte_score_i.softmax(dim=1).unsqueeze(2)]
                # self.spatial_att_score_list += [atte_score_i.softmax(dim=1)]
                # [B * 18] = [B * 18] * [B * 18]
                he, ce = self.LSTMcell_spatial_encoder(inputs_i, (he, ce))
                mid_output[:, i, :] = he  # torch.Size([32, 30, 64])
            mid_output, (hi, ci) = self.LSTM_mid(mid_output)
            x = self.temporal_attention_Wh(torch.cat([ht, ct], dim=1)).repeat(len_enco, 1, 1).transpose(0, 1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(x))
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(2)]
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output).squeeze()

        out = torch.cat(target_list, dim=1)
        return out.squeeze()


class DS_RNN(nn.Module):
    def __init__(self, n_inp, output_size, output_column, args):
        super(DS_RNN, self).__init__()
        self.model_name = 'DS_RNN'
        self.details = 'nomid_i=4_2sf'
        # DS_RNN论文里的设计 v8 从 notf 那里改过来的
        # self.model_path = os.path.join('../src', 'saved_pkl_model',
        #                                self.model_name + '.pkl')
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.n_inp = n_inp
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size
        T_enco = args.encoder_sequence_length

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.n_inp,
                                                    self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state,
                                                     self.decoder_hidden_state).to(device)

        self.spatial_attention_Wi = nn.Linear(T_enco, T_enco)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, T_enco, bias=False)
        self.spatial_attention_Vd = nn.Linear(T_enco, 1)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        self.regressor = nn.Linear(self.decoder_hidden_state, 1)
        # self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
        #                         hidden_size=self.hidden_size,
        #                         num_layers=1,
        #                         batch_first=True,
        #                         dropout=0.5)
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        inputs = input_p_q[:, :len_enco, :]
        # [B * 40 * 18]

        # # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        self.spatial_att_score_list = []
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)).
                    repeat(self.n_inp, 1, 1).permute(1, 0, 2) +
                    self.spatial_attention_Wi(inputs.transpose(1, 2))
                )).squeeze()
            self.spatial_att_score_list += [atte_score_i.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        # mid_output, hh = self.LSTM_mid(mid_output)
        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        ci = torch.zeros(batch_size, self.decoder_hidden_state, device=device)

        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, len_deco):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                                     1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(dim=2)]
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2).softmax(dim=2), mid_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = torch.cat(decode_output, dim=1)
        # out, hhh = self.LSTM_out(out)
        out = self.regressor(out)
        out = out[:, -len_deco:, :]
        return out.squeeze()


class DS_RNN_II(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(DS_RNN_II, self).__init__()
        self.model_name = 'DS_RNN_II'
        self.details = 'mid_noTsf_i=8'# == STAttention_notf v3
        # V2 加了一层
        # V3 又加了一层
        # V4 是全连接层代替decoder
        # V5 是 code 通过 hi 与 ci 输入
        # V6 是加长decoder
        # self.model_path = os.path.join('../src', 'saved_pkl_model',
                                       # self.model_name + '.pkl')
        print('model_name:', self.model_name)
        self.args = args
        self.input_size = input_p_q_size
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []
        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state, self.decoder_hidden_state).to(device)

        self.spatial_attention_Wi = nn.Linear(self.input_size, self.hidden_size)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.spatial_attention_Vd = nn.Linear(self.hidden_size, self.input_size)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)

        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.regressor = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, inputs, label_p):
        batch_size = inputs.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length

        # # Spatial attention
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        self.spatial_att_score_list = []
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h, c], dim=1)) +
                    self.spatial_attention_Wi(inputs[:, i, :])
                ))
            self.spatial_att_score_list += [atte_score_i.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            h, c = self.LSTMcell_spatial_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        mid_output, hh = self.LSTM_mid(mid_output)

        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        ci = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, len_deco+4):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                                     1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(dim=2)]
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]
        out = torch.cat(decode_output, dim=1)
        out = self.regressor(out)
        out = out[:, -len_deco:, :]
        return out.squeeze()


class DS_RNN_III(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(DS_RNN_III, self).__init__()
        self.model_name = 'DS_RNN_III_best'
        self.details = 'no_tem_softm'
        self.n_inp = input_p_q_size
        self.n_hid = args.num_hidden_state
        self.n_hid_deco = self.n_hid
        self.T_enco = args.encoder_sequence_length
        self.m_enco = args.num_hidden_state
        self.T_deco = args.decoder_sequence_length

        self.Tanh = nn.Tanh()
        self.lstm_s_encoder = nn.LSTMCell(self.n_inp,
                                          self.n_hid).to(device)
        self.lstm_t_decoder = nn.LSTMCell(self.n_hid_deco,
                                          self.n_hid_deco).to(device)
        self.s_att_Wi = nn.Linear(self.T_enco, self.m_enco)
        self.s_att_Wj = nn.Linear(self.n_inp, self.m_enco)
        self.s_att_We = nn.Linear(self.n_hid * 2, self.m_enco)
        self.s_att_Vd = nn.Linear(self.m_enco, 1)
        self.t_att_Wx = nn.Linear(self.n_hid, self.n_hid)
        self.t_att_Wh = nn.Linear(2 * self.n_hid_deco, self.n_hid)
        self.t_att_V = nn.Linear(self.n_hid, 1)
        self.regressor = nn.Linear(self.n_hid_deco, 1)
        self.LSTM_mid = nn.LSTM(input_size=self.n_hid,
                                hidden_size=self.n_hid,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        inputs = input_p_q[:, :self.T_enco, :]
        # [B * 40 * 18]

        # # Spatial attention
        h = torch.zeros(batch_size, self.n_hid, device=device)
        c = torch.zeros(batch_size, self.n_hid, device=device)
        self.spatial_att_score_list = []
        mid_output = torch.zeros(batch_size, self.T_enco, self.n_hid, device=device)
        for i in range(0, self.T_enco):
            s_att_score = self.s_att_Vd(
                self.Tanh(
                    self.s_att_We(torch.cat([h, c], dim=1)).
                    repeat(self.n_inp, 1, 1).permute(1, 0, 2) +
                    self.s_att_Wi(inputs.transpose(1, 2)) +
                    self.s_att_Wj(inputs[:, i, :])
                    .repeat(self.n_inp, 1, 1).permute(1, 0, 2)
                )).squeeze()
            self.spatial_att_score_list += [s_att_score.softmax(dim=1).unsqueeze(2)]
            inputs_i = torch.mul(inputs[:, i, :], s_att_score.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h, c = self.lstm_s_encoder(inputs_i, (h, c))
            mid_output[:, i, :] = h  # torch.Size([32, 30, 64])

        mid_output, hh = self.LSTM_mid(mid_output)
        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, self.n_hid_deco, device=device)
        ci = torch.zeros(batch_size, self.n_hid_deco, device=device)

        decode_output = []
        self.temporal_att_score_list = []
        for i_decoder in range(0, self.T_deco+6):
            t_att_score_ = self.t_att_Wh(torch.cat([hi, ci], dim=1)).\
                               repeat(self.T_enco, 1, 1).transpose(0, 1) + \
                               self.t_att_Wx(mid_output)
            t_att_score = self.t_att_V(self.Tanh(t_att_score_))
            # torch.Size([B, 40, 1])
            self.temporal_att_score_list += [t_att_score.transpose(1, 2).softmax(dim=2)]
            decoder_in = torch.bmm(t_att_score.transpose(1, 2), mid_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.lstm_t_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = torch.cat(decode_output, dim=1)
        # out, hhh = self.LSTM_out(out)
        out = self.regressor(out)
        out = out[:, -self.T_deco:, :]
        return out.squeeze()


class DSTP_RNN(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(DSTP_RNN, self).__init__()
        self.model_name = 'DSTP_RNN'
        self.details = 'i=10_nomid_sf'
        # DS_RNN论文里的设计 v8 从 notf 那里改过来的
        self.model_path = os.path.join('../src', 'saved_pkl_model',
                                       self.model_name + '.pkl')
        print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size
        enco_len = args.encoder_sequence_length

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size-1,
                                                    self.hidden_size).to(device)
        self.LSTMcell_spatial_encoder2 = nn.LSTMCell(self.hidden_size + 1,
                                                    self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state,
                                                     self.decoder_hidden_state).to(device)

        self.spatial_attention_Wi = nn.Linear(enco_len, enco_len)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, enco_len, bias=False)
        self.spatial_attention_Vd = nn.Linear(enco_len, 1)

        self.spatial_attention_Wi2 = nn.Linear(enco_len, enco_len)
        self.spatial_attention_We2 = nn.Linear(self.hidden_size * 2, enco_len, bias=False)
        self.spatial_attention_Vd2 = nn.Linear(enco_len, 1)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        # self.att_score = None

        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        # inputs = input_p_q[:, :len_enco, :]
        inputs = input_p_q[:, :len_enco, list(range(0, 14)) + list(range(15, 18))]
        # assert input_p_q[:, :, 14] == label_p, 'bug 1!'
        labels_p = label_p[:, :len_enco]
        # [B * 40 * 18]

        # # Spatial attention phase 1
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd(
                self.Tanh(
                    self.spatial_attention_We(torch.cat([h1, c1], dim=1)).
                    repeat(self.input_size-1, 1, 1).permute(1,0,2) +
                    self.spatial_attention_Wi(inputs.transpose(1, 2))
                )).squeeze()
            inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h1, c1 = self.LSTMcell_spatial_encoder(inputs_i, (h1, c1))
            mid_output[:, i, :] = h1  # torch.Size([32, 30, 64])

        # Spatial attention phase 2
        mid_output = torch.cat([mid_output, labels_p.unsqueeze(2)], dim=2)
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        final_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd2(
                self.Tanh(
                    self.spatial_attention_We2(torch.cat([h, c], dim=1)).
                    repeat(self.hidden_size+1, 1, 1).permute(1, 0, 2) +
                    self.spatial_attention_Wi2(mid_output.transpose(1, 2))
                )).squeeze()
            inputs_i2 = torch.mul(mid_output[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h, c = self.LSTMcell_spatial_encoder2(inputs_i2, (h, c))
            final_output[:, i, :] = h

        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        ci = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        decode_output = []
        for i_decoder in range(0, len_deco+6):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                                     1) + \
                             self.temporal_attention_Wx(final_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2).softmax(dim=2), final_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = torch.cat(decode_output, dim=1)
        # out, hhh = self.LSTM_out(out)
        out = self.regressor_2(out)
        out = out[:, -len_deco:, :]
        return out.squeeze()


class DSTP_RNN_II(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        super(DSTP_RNN_II, self).__init__()
        self.model_name = 'DSTP_RNN_II'
        # DS_RNN论文里的设计 v8 从 notf 那里改过来的
        self.args = args
        self.input_size = input_p_q_size
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size
        enco_len = args.encoder_sequence_length

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder11 = nn.LSTMCell(self.input_size - 1,
                                                      self.hidden_size).to(device)
        self.LSTMcell_spatial_encoder12 = nn.LSTMCell(self.input_size,
                                                      self.hidden_size).to(device)
        self.LSTMcell_spatial_encoder2 = nn.LSTMCell(self.hidden_size * 2 + 1,
                                                     self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state,
                                                     self.decoder_hidden_state).to(device)

        self.spatial_attention_Wi11 = nn.Linear(enco_len, enco_len)
        self.spatial_attention_We11 = nn.Linear(self.hidden_size * 2, enco_len, bias=False)
        self.spatial_attention_Vd11 = nn.Linear(enco_len, 1)

        self.spatial_attention_Wi12 = nn.Linear(enco_len, enco_len)
        self.spatial_attention_We12 = nn.Linear(self.hidden_size * 2, enco_len, bias=False)
        self.spatial_attention_Vd12 = nn.Linear(enco_len, 1)

        self.spatial_attention_Wi2 = nn.Linear(enco_len, enco_len)
        self.spatial_attention_We2 = nn.Linear(self.hidden_size * 2, enco_len, bias=False)
        self.spatial_attention_Vd2 = nn.Linear(enco_len, 1)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        # self.att_score = None

        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        inputs11 = input_p_q[:, :len_enco, list(range(0, 14)) + list(range(15, 18))]
        inputs12 = input_p_q[:, :len_enco, :]
        labels_p = label_p[:, :len_enco]
        # [B * 40 * 18]

        # # Spatial attention phase 11
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output11 = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd11(
                self.Tanh(
                    self.spatial_attention_We11(torch.cat([h1, c1], dim=1)).
                    repeat(self.input_size-1, 1, 1).permute(1,0,2) +
                    self.spatial_attention_Wi11(inputs11.transpose(1, 2))
                )).squeeze()
            inputs_i = torch.mul(inputs11[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h1, c1 = self.LSTMcell_spatial_encoder11(inputs_i, (h1, c1))
            mid_output11[:, i, :] = h1  # torch.Size([32, 30, 64])

        # # Spatial attention phase 12
        h12 = torch.zeros(batch_size, self.hidden_size, device=device)
        c12 = torch.zeros(batch_size, self.hidden_size, device=device)
        mid_output12 = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd12(
                self.Tanh(
                    self.spatial_attention_We12(torch.cat([h12, c12], dim=1)).
                    repeat(self.input_size, 1, 1).permute(1, 0, 2) +
                    self.spatial_attention_Wi12(inputs12.transpose(1, 2))
                )).squeeze()
            inputs_i = torch.mul(inputs12[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h12, c12 = self.LSTMcell_spatial_encoder12(inputs_i, (h12, c12))
            mid_output12[:, i, :] = h12  # torch.Size([32, 30, 64])

        # Spatial attention phase 2
        mid_output = torch.cat([mid_output11, mid_output12, labels_p.unsqueeze(2)], dim=2)
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        final_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
        for i in range(0, len_enco):
            atte_score_i = self.spatial_attention_Vd2(
                self.Tanh(
                    self.spatial_attention_We2(torch.cat([h, c], dim=1)).
                    repeat(self.hidden_size*2
                           +1, 1, 1).permute(1, 0, 2) +
                    self.spatial_attention_Wi2(mid_output.transpose(1, 2))
                )).squeeze()
            inputs_i2 = torch.mul(mid_output[:, i, :], atte_score_i.softmax(dim=1))
            # [B * 18]                  [B * 18] * [B * 18]
            h, c = self.LSTMcell_spatial_encoder2(inputs_i2, (h, c))
            final_output[:, i, :] = h

        # Temporal attention 和论文里的一毛一样了
        hi = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        ci = torch.zeros(batch_size, self.decoder_hidden_state, device=device)
        decode_output = []
        for i_decoder in range(0, len_deco+6):
            atte_score_2_x = self.temporal_attention_Wh(torch.cat([hi, ci], dim=1)).repeat(len_enco, 1, 1).transpose(0,
                                                                                                                     1) + \
                             self.temporal_attention_Wx(final_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(atte_score_2_x))
            # torch.Size([B, 40, 1])
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2).softmax(dim=2), final_output)
            # [B * 1 * 64] = [B * 1 * 40] * [B * 40 * 64]
            hi, ci = self.LSTMcell_temporal_decoder(decoder_in.squeeze(), (hi, ci))
            decode_output += [hi.unsqueeze(1)]

        out = torch.cat(decode_output, dim=1)
        # out, hhh = self.LSTM_out(out)
        out = self.regressor_2(out)
        out = out[:, -len_deco:, :]
        return out.squeeze()


class dSTA_RNN(nn.Module):
    def __init__(self, input_p_q_size, output_size, output_column, args):
        # v7 的特点，将decoder从4延长到44，以mid_output
        # a dynamic global attention
        super(dSTA_RNN, self).__init__()
        self.model_name = 'dSTA_RNN_correcti=8'
        # self.model_path = os.path.join('../src', 'saved_pkl_model',
        #                                self.model_name + '.pkl')
        # print('model_name:', self.model_name, 'saved at:', self.model_path)
        self.args = args
        self.input_size = input_p_q_size
        self.hidden_size = args.num_hidden_state
        self.decoder_hidden_state = self.hidden_size

        self.Tanh = nn.Tanh()
        self.LSTMcell_spatial_encoder = nn.LSTMCell(self.input_size, self.hidden_size).to(device)
        self.LSTMcell_temporal_decoder = nn.LSTMCell(self.decoder_hidden_state + 1, self.decoder_hidden_state).to(
            device)
        self.LSTMcell_target_encoder = nn.LSTMCell(output_size+self.hidden_size, self.hidden_size).to(device)

        self.LSTM_target = nn.LSTM(input_size=output_size+self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   dropout=0.5)
        self.LSTM_mid = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)
        self.LSTM_out = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5)

        self.spatial_attention_Wi = nn.Linear(args.encoder_sequence_length, args.encoder_sequence_length)
        self.spatial_attention_We = nn.Linear(self.hidden_size * 2, args.encoder_sequence_length, bias=False)
        self.spatial_attention_Wt = nn.Linear(self.hidden_size * 2, args.encoder_sequence_length, bias=False)
        self.spatial_attention_Vd = nn.Linear(args.encoder_sequence_length, 1)

        self.temporal_attention_Wx = nn.Linear(self.hidden_size, self.hidden_size)
        self.temporal_attention_Wh = nn.Linear(2 * self.decoder_hidden_state, self.hidden_size, bias=False)
        # self.temporal_attention_Vd = nn.Linear(self.hidden_size, 1)
        self.temporal_attention_V = nn.Linear(self.hidden_size, 1)
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []
        self.regressor_2 = nn.Linear(self.decoder_hidden_state, 1)

    def forward(self, input_p_q, label_p):
        batch_size = input_p_q.shape[0]
        len_enco = self.args.encoder_sequence_length
        len_deco = self.args.decoder_sequence_length
        self.spatial_att_score_list = []
        self.temporal_att_score_list = []
        # inputs = input_p_q
        inputs = input_p_q[:, :len_enco, :]
        look_back = 8

        # # Spatial attention
        ht = torch.zeros(batch_size, self.hidden_size, device=device)
        ct = torch.zeros(batch_size, self.hidden_size, device=device)
        decoder_in = torch.zeros(batch_size, self.hidden_size, device=device)
        label_p_i = label_p[:, len_deco + len_enco-look_back]
        # [B]
        target_list = []
        self.spatial_att_score_list = []
        for it in range(len_deco + len_enco-look_back, len_deco + len_enco-1):
            x = torch.cat([label_p_i.unsqueeze(1), decoder_in], dim=1)
            ht, ct = self.LSTMcell_target_encoder(x, (ht, ct))
            # [B * 64]
            target = self.regressor_2(ht)
            if it+1 < len_enco:
                label_p_i = label_p[:, it+1]
            else:
                target_list += [target]
                label_p_i = target.squeeze()
            he = torch.zeros(batch_size, self.hidden_size, device=device)
            ce = torch.zeros(batch_size, self.hidden_size, device=device)
            mid_output = torch.zeros(batch_size, len_enco, self.hidden_size, device=device)
            for i in range(0, len_enco):
                atte_score_i = self.spatial_attention_Vd(
                    self.Tanh(
                        self.spatial_attention_We(torch.cat([he, ce], dim=1)).
                        repeat(self.input_size, 1, 1).permute(1, 0, 2) +
                        self.spatial_attention_Wi(inputs.transpose(1, 2)) +
                        self.spatial_attention_Wt(torch.cat([ht, ct], dim=1)).
                        repeat(self.input_size, 1, 1).permute(1, 0, 2)
                        # [B * 64]
                    ))
                # self.temporal_att_score += [atte_score_i.softmax(dim=1)]
                inputs_i = torch.mul(inputs[:, i, :], atte_score_i.softmax(dim=1).squeeze())
                # [B * 18] = [B * 18] * [B * 18]
                self.spatial_att_score_list += [atte_score_i.softmax(dim=1)]
                he, ce = self.LSTMcell_spatial_encoder(inputs_i, (he, ce))
                mid_output[:, i, :] = he  # torch.Size([32, 30, 64])
            mid_output, (hi, ci) = self.LSTM_mid(mid_output)
            x = self.temporal_attention_Wh(torch.cat([ht, ct], dim=1)).repeat(len_enco, 1, 1).transpose(0, 1) + \
                             self.temporal_attention_Wx(mid_output)
            atte_score_2 = self.temporal_attention_V(self.Tanh(x))
            self.temporal_att_score_list += [atte_score_2.transpose(1, 2).softmax(2)]
            decoder_in = torch.bmm(atte_score_2.transpose(1, 2), mid_output).squeeze()

        out = torch.cat(target_list, dim=1)
        return out.squeeze()

