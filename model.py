#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/11/5 10:27

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dgl_gnn import UnsupervisedGAT, UnsupervisedGIN
from module.gps_transformer_layer import Encoder as Transformer
import dgl


def get_dict_info_batch(input_id, features_dict):
    """
    batched dict info
    """
    # input_id = [1, batch size]
    input_id = input_id.reshape(-1)
    features = torch.index_select(features_dict, dim=0, index=input_id)
    return features


def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        pred = x_exp / (x_exp_sum + 1e-6)
        pred = torch.clip(pred, 1e-6, 1)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / (x_exp_sum + 1e-6)
    return output_custom


def mask_graph_log_softmax(g, log_flag=True):
    lg = g.ndata['lg']
    w = g.ndata['w']

    maxes = dgl.max_nodes(g, 'lg')
    maxes = dgl.broadcast_nodes(g, maxes)

    x_exp = torch.exp(lg - maxes) * w
    g.ndata['lg'] = x_exp
    x_exp_sum = dgl.sum_nodes(g, 'lg')
    x_exp_sum = dgl.broadcast_nodes(g, x_exp_sum)

    if log_flag:
        pred = x_exp / (x_exp_sum + 1e-6)
        pred = torch.clip(pred, 1e-6, 1)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / (x_exp_sum + 1e-6)
    return output_custom


class RoadGNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.gnn_type = parameters.gnn_type
        self.node_input_dim = parameters.id_emb_dim
        self.node_hidden_dim = parameters.hid_dim
        self.num_layers = parameters.num_layers
        if self.gnn_type == 'gat':
            self.gnn = UnsupervisedGAT(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        else:
            self.gnn = UnsupervisedGIN(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, g, x, readout=True):
        '''
        :param x: road emb id with size [node size, id dim]
        :return: road hidden emb with size [graph size, hidden dim] if readout
                 else [node size, hidden dim]
        '''
        x = self.dropout(self.gnn(g, x))
        if not readout:
            return x

        g.ndata['x'] = x
        if 'w' in g.ndata:
            return dgl.mean_nodes(g, 'x', weight='w'), g
        else:
            return dgl.mean_nodes(g, 'x'), g


class Extra_MLP(nn.Module):
    """
        MLP with tanh activation function.
    """

    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc_out(x))
        return out


class Encoder(nn.Module):
    """
        Trajectory Encoder.
        Set online_feature_flag=False.
        Keep pro_features_flag (hours and holiday information).
        Encoder: RNN + MLP
    """

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.pro_features_flag = parameters.pro_features_flag
        self.device = parameters.device
        self.grid_flag = parameters.grid_flag
        self.transformer_layers = parameters.transformer_layers
        self.gnn_type = parameters.gnn_type

        input_dim = 3
        if self.online_features_flag:
            input_dim += parameters.online_dim
        if self.dis_prob_mask_flag:
            input_dim += parameters.hid_dim
        if self.grid_flag:
            input_dim += parameters.id_emb_dim // 2

        self.fc_in = nn.Linear(input_dim, self.hid_dim)
        self.pred_out = nn.Linear(self.hid_dim, 1)

        self.transformer = Transformer(self.gnn_type, self.hid_dim, self.transformer_layers,
                                       self.device)
        # self.final_encoder = EncoderLayer(self.hid_dim)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, g, pro_features):
        # src = [src len, batch size, 3]
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n
        # src_len = [batch size]
        max_src_len = src.size(0)
        bs = src.size(1)
        mask3d = torch.zeros(bs, max_src_len, max_src_len).to(self.device)
        mask2d = torch.zeros(bs, max_src_len).to(self.device)
        for i in range(bs):
            mask3d[i, :src_len[i], :src_len[i]] = 1
            mask2d[i, :src_len[i]] = 1
        src = self.fc_in(src)
        src = src.transpose(0, 1)

        outputs, g = self.transformer(src, g, mask3d, mask2d)
        g.ndata['lg'] = self.pred_out(g.ndata['x'])
        g.ndata['lg'] = mask_graph_log_softmax(g)

        # outputs = self.final_encoder(outputs, mask, norm=True)
        outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]

        # idx = [i for i in range(bs)]
        # hidden = outputs[[i - 1 for i in src_len], idx, :].unsqueeze(0)
        assert outputs.size(0) == max_src_len

        for i in range(bs):
            outputs[src_len[i]:, i, :] = 0
        hidden = torch.mean(outputs, dim=0).unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hid dim]

        return outputs, hidden, g


class Attention(nn.Module):
    """
        Calculate the attention score of the sequence with respect to the query vector.
        hidden: [1, batch size, hid dim] represents to query vector.
        encoder_outputs: [src len, batch size, hid dim * num directions] represents to key/value vectors.
        :return [batch size, src len] represents to attention score with sum of dim 1 to 1.
    """

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e6)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    """
        Trajectory Decoder.
        Set online_feature_flag=False.
        Keep tandem_fea_flag (road network static feature).

        Decoder: Attention + RNN
        If calculate attention, calculate the attention between current hidden vector and encoder output.
        Feed rid embedding, hidden vector, input rate into rnn to get the next prediction.
    """

    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        rnn_input_dim = self.hid_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim

        type_input_dim = self.hid_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
            nn.Linear(type_input_dim, self.hid_dim),
            nn.ReLU()
        )

        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network

        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float.
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        input_id = input_id.squeeze(1)  # cannot use squeeze() bug for batch size = 1
        # input_id = [batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(torch.index_select(self.emb_id, index=input_id, dim=0)).unsqueeze(0)
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate,
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        if not (output == hidden).all():
            import pdb
            pdb.set_trace()
        assert (output == hidden).all()

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)),
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(torch.index_select(self.emb_id, index=max_id, dim=0))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)), dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden


class Seq2SeqMulti(nn.Module):
    """
    Trajectory Seq2Seq Model.
    """

    def __init__(self, encoder, decoder, device, parameters):
        super().__init__()
        self.id_size = parameters.id_size
        self.hid_dim = parameters.hid_dim
        self.grid_num = parameters.grid_num
        self.id_emb_dim = parameters.id_emb_dim
        self.grid_flag = parameters.grid_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.subg = parameters.subg
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))
        self.device = device
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.rn_grid_dict = parameters.rn_grid_dict
        self.pad_rn_grid, _ = self.merge(self.rn_grid_dict)
        # self.grid_len = [fea.shape[0] - 1 for fea in self.rn_grid_dict]
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid_dict])

        self.gnn = RoadGNN(parameters)
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim)
        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti

        self.params = parameters


    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths


    def forward(self, src, src_len, trg_id, trg_rate, trg_len,
                constraint_mat_trg, pro_features,
                online_features_dict, rid_features_dict, constraint_graph_src,
                src_gps_seqs, teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        constraint_src = [src len, batch size, id size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        Return:
        ------
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        max_src_len = src.size(0)
        batch_size = trg_id.size(1)

        # road representation
        max_grid_len = self.pad_rn_grid.size(1)
        rn_grid = self.pad_rn_grid.reshape(-1, 2)
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)

        # change to pad_packed_sequence
        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                              batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_emb = grid_output.reshape(-1, self.id_emb_dim)
        assert grid_emb.size(0) == self.emb_id.size(0)
        # grid_emb = grid_output[self.grid_len, range(len(self.grid_len)), :]  # [rid, dim]

        input_road = torch.index_select(self.emb_id, index=self.subg.ndata['id'].long(), dim=0)
        input_grid = torch.index_select(grid_emb, index=self.subg.ndata['id'].long(), dim=0)
        input_emb = F.leaky_relu(input_road + input_grid)
        # input_emb = torch.cat((input_road, input_grid), dim=-1)
        # finish changing

        road_emb, _ = self.gnn(self.subg, input_emb)
        road_emb = road_emb.reshape(-1, self.hid_dim)
        self.decoder.emb_id = road_emb  # [id size, hidden dim]

        assert self.dis_prob_mask_flag
        input_cons = torch.index_select(road_emb, index=constraint_graph_src.ndata['id'].long(),
                                        dim=0)

        constraint_graph_src.ndata['x'] = input_cons
        cons_emb = dgl.mean_nodes(constraint_graph_src, 'x', weight='w')
        cons_emb = cons_emb.reshape(batch_size, max_src_len, -1).transpose(0, 1)

        if self.grid_flag:
            grid_input = src[:, :, :2].reshape(-1, 2).cpu().numpy()
            grid_emb = self.grid_id[grid_input[:, 0].tolist(), grid_input[:, 1].tolist(), :]
            grid_emb = grid_emb.reshape(max_src_len, batch_size, -1)
            src = torch.cat((cons_emb, grid_emb, src), dim=-1)
        else:
            src = torch.cat((cons_emb, src), dim=-1)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hiddens, g = self.encoder(src, src_len, constraint_graph_src, pro_features)

        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None

        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                    encoder_outputs, hiddens, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    constraint_mat_trg, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate, g

    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                  attn_mask, constraint_mat[t], pro_features,
                                                                  online_features, rid_features)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1

        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = -100
            outputs_id[i][trg_len[i]:, 0] = 0  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate
