import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday

import numpy as np
import torch

from utils.spatial_func import distance
from utils.trajectory_func import Trajectory
from utils.parse_traj import ParseMMTraj
from model_utils import get_constraint_mask, get_gps_subgraph
from utils.graph_func import *
import dgl


class Dataset(torch.utils.data.Dataset):
    """
    customize a dataset for PyTorch
    """

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        """
        Remove norm_grid_poi_dict, norm_grid_rnfea_dict, weather_dict.
        parameter.ds_type: ['uniform', 'random'].
        parameter.keep_ratio: float [0.0625, 0.125, 0.25].
        parameter.win_size: set to a large interger, larger than the max length of trajectory, default to 1000.
        parameter.grid_size: size of each grid, default to 50.
        parameter.time_span: time interval between two consecutive points [10, 12, 15].
        """
        self.parameters = parameters
        self.sub_graphs = parameters.subgs
        self.rn = rn
        self.mbr = mbr  # MBR of all trajectories
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span
        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.mode = mode
        # above should be [num_seq, len_seq(unpadded)]
        self.get_data(trajs_dir, parameters.win_size, parameters.ds_type, parameters.keep_ratio)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.src_grid_seqs)

    def __getitem__(self, index):
        """Generate one sample of data"""
        src_grid_seq = self.src_grid_seqs[index]
        src_gps_seq = self.src_gps_seqs[index]
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]

        src_grid_seq = self.add_token(src_grid_seq)
        src_gps_seq = self.add_token(src_gps_seq)
        trg_gps_seq = self.add_token(trg_gps_seq)
        trg_rid = self.add_token(trg_rid)
        trg_rate = self.add_token(trg_rate)
        src_pro_fea = torch.tensor(self.src_pro_feas[index])
        src_length = torch.tensor([len(src_grid_seq)])
        trg_length = torch.tensor([len(trg_gps_seq)])

        if self.parameters.dis_prob_mask_flag:
            constraint_mat_trg, constraint_mat_src = get_constraint_mask(
                src_grid_seq.unsqueeze(0), src_gps_seq.unsqueeze(0), src_length, trg_length, self.rn, self.parameters)
            constraint_mat_trg = constraint_mat_trg.squeeze(0)
            constraint_mat_src = constraint_mat_src.squeeze(0)
            constraint_graph_src = get_gps_subgraph(constraint_mat_src, src_grid_seq, trg_rid, self.parameters)
        else:
            constraint_mat_trg = torch.zeros(len(trg_length), self.parameters.id_size)
            constraint_graph_src = [empty_graph() for _ in range(len(src_grid_seq))]

        return src_grid_seq, src_gps_seq, src_pro_fea, trg_gps_seq, trg_rid, trg_rate, \
               constraint_mat_trg, constraint_graph_src

    def add_token(self, sequence):
        """
        Append start element(sos in NLP) for each sequence. And convert each list to tensor.
        """
        new_sequence = []
        dimension = len(sequence[0])
        start = [0] * dimension  # pad 0 as start of rate sequence
        new_sequence.append(start)
        new_sequence.extend(sequence)
        new_sequence = torch.tensor(new_sequence)
        return new_sequence

    def get_data(self, trajs_dir, win_size, ds_type, keep_ratio):
        parser = ParseMMTraj(self.rn)
        if self.mode == 'train':
            src_file = os.path.join(trajs_dir, 'train/train_input.txt')
            trg_file = os.path.join(trajs_dir, 'train/train_output.txt')
        elif self.mode == 'valid':
            src_file = os.path.join(trajs_dir, 'valid/valid_input.txt')
            trg_file = os.path.join(trajs_dir, 'valid/valid_output.txt')
        elif self.mode == 'test':
            src_file = os.path.join(trajs_dir, f'test/test_input.txt')
            trg_file = os.path.join(trajs_dir, f'test/test_output.txt')
        else:
            raise NotImplementedError

        src_trajs = parser.parse(src_file, is_target=False)
        trg_trajs = parser.parse(trg_file, is_target=True)
        assert len(src_trajs) == len(trg_trajs)

        for (src_traj, trg_traj) in tqdm(zip(src_trajs, trg_trajs)):
            if self.mode != 'test':
                _, _, _, ls_grid_seq_ls, ls_gps_seq_ls, features_ls = self.parse_traj(src_traj, win_size, ds_type,
                                                                            keep_ratio=keep_ratio)
            else:
                _, _, _, ls_grid_seq_ls, ls_gps_seq_ls, features_ls = self.parse_traj(src_traj, win_size, ds_type,
                                                                                      keep_ratio=1)

            mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, _, _, _ = self.parse_traj(trg_traj, win_size, ds_type,
                                                                            keep_ratio=1)

            self.trg_gps_seqs.extend(mm_gps_seq_ls)
            self.trg_rids.extend(mm_eids_ls)
            self.trg_rates.extend(mm_rates_ls)
            self.src_grid_seqs.extend(ls_grid_seq_ls)
            self.src_gps_seqs.extend(ls_gps_seq_ls)
            self.src_pro_feas.extend(features_ls)
            assert len(mm_gps_seq_ls) == len(mm_eids_ls) == len(mm_rates_ls)

        assert len(self.trg_gps_seqs) == len(self.trg_rids) == len(self.trg_rates) == \
                   len(self.src_gps_seqs) == len(self.src_grid_seqs) == len(self.src_pro_feas), \
                'The number of source and target sequence must be equal.'
        del src_trajs
        del trg_trajs

    def parse_traj(self, traj, win_size, ds_type, keep_ratio):
        """
        Split traj based on length.
        Preprocess ground truth (map-matched) Trajectory(), get gps sequence, rid list and rate list.
        Down sample original Trajectory(), get ls_gps, ls_grid sequence and profile features
        Args:
        -----
        traj:
            Trajectory()
        win_size:
            window size of length for a single high sampling trajectory
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_steps element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        --------
        new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls
        """
        new_trajs = self.get_win_trajs(traj, win_size)

        mm_gps_seq_ls, mm_eids_ls, mm_rates_ls = [], [], []
        ls_grid_seq_ls, ls_gps_seq_ls, features_ls = [], [], []

        for tr in new_trajs:
            tmp_pt_list = tr.pt_list

            # get target sequence
            mm_gps_seq, mm_eids, mm_rates = self.get_trg_seq(tmp_pt_list)

            # get source sequence
            if keep_ratio != 1:
                ds_pt_list = self.downsample_traj(tmp_pt_list, ds_type, keep_ratio)
            else:
                ds_pt_list = tmp_pt_list

            ls_grid_seq, ls_gps_seq, hours, ttl_t = self.get_src_seq(ds_pt_list)
            features = self.get_pro_features(ds_pt_list, hours)

            mm_gps_seq_ls.append(mm_gps_seq)
            mm_eids_ls.append(mm_eids)
            mm_rates_ls.append(mm_rates)
            ls_grid_seq_ls.append(ls_grid_seq)
            ls_gps_seq_ls.append(ls_gps_seq)
            features_ls.append(features)

        return mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls

    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        if len_pt_list < win_size:
            return [traj]

        num_win = len_pt_list // win_size
        last_traj_len = len_pt_list % win_size + 1
        new_trajs = []
        for w in range(num_win):
            # if last window is large enough then split to a single trajectory
            if w == num_win and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            elif w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            # else split trajectories based on the window size
            else:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]
                # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(tmp_pt_list)
            new_trajs.append(new_traj)
        return new_trajs

    def get_src_seq(self, ds_pt_list):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        first_pt = ds_pt_list[0]
        last_pt = ds_pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_normalized_t(first_pt, last_pt, time_interval)
        for ds_pt in ds_pt_list:
            hours.append(ds_pt.time.hour)
            t = self.get_normalized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            locgrid_xid, locgrid_yid = self.gps2grid(ds_pt, self.mbr, self.grid_size)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])

        return ls_grid_seq, ls_gps_seq, hours, ttl_t

    def get_trg_seq(self, tmp_pt_list):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is None:
                return None, None, None
            else:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
                mm_eids.append([self.rn.valid_edge_one[candi_pt.eid]])  # keep the same format as seq
                mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates

    def get_pro_features(self, ds_pt_list, hours):
        holiday = is_holiday(ds_pt_list[0].time) * 1
        hour = {'hour': np.bincount(hours).argmax()}  # find most frequent hours as hour of the trajectory
        features = self.one_hot(hour) + [holiday]
        return features

    def gps2grid(self, pt, mbr, grid_size):
        """
        mbr:
            MBR class.
        grid size:
            int. in meter
        """
        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size

        lat = pt.lat
        lng = pt.lng
        locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
        locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1

        return locgrid_x, locgrid_y

    def get_normalized_t(self, first_pt, current_pt, time_interval):
        """
        calculate normalized t from first and current pt
        return time index (normalized time)
        """
        t = int(1 + ((current_pt.time - first_pt.time).seconds / time_interval))
        return t

    @staticmethod
    def get_distance(pt_list):
        dist = 0.0
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist

    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        """
        Down sample trajectory
        Args:
        -----
        pt_list:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        elif ds_type == 'random':
            sampled_inds = sorted(
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        return new_pt_list

    @staticmethod
    def one_hot(data):
        one_hot_dict = {'hour': 24}
        for k, v in data.items():
            encoded_data = [0] * one_hot_dict[k]
            encoded_data[v] = 1
        return encoded_data


# Use for DataLoader
def collate_fn(data):
    """
    Reference: https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
    Creates mini-batch tensors from the list of tuples (src_seq, src_pro_fea, trg_seq, trg_rid, trg_rate).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
    -----
    data: list of tuple (src_seq, src_pro_fea, trg_seq, trg_rid, trg_rate), from dataset.__getitem__().
        - src_seq: torch tensor of shape (?,2); variable length.
        - src_pro_fea: torch tensor of shape (1,64) # concatenate all profile features
        - trg_seq: torch tensor of shape (??,2); variable length.
        - trg_rid: torch tensor of shape (??); variable length.
        - trg_rate: torch tensor of shape (??); variable length.
    Returns:
    --------
    src_grid_seqs:
        torch tensor of shape (batch_size, padded_length, 3)
    src_gps_seqs:
        torch tensor of shape (batch_size, padded_length, 3).
    src_pro_feas:
        torch tensor of shape (batch_size, feature_dim) unnecessary to pad
    src_lengths:
        list of length (batch_size); valid length for each padded source sequence.
    trg_seqs:
        torch tensor of shape (batch_size, padded_length, 2).
    trg_rids:
        torch tensor of shape (batch_size, padded_length, 1).
    trg_rates:
        torch tensor of shape (batch_size, padded_length, 1).
    trg_lengths:
        list of length (batch_size); valid length for each padded target sequence.
    constraint_mat:
        torch tensor of shape (batch_size, padded_length, is_size)
    pre_grids:
        torch tensor of shape (batch_size, padded_length, 3)
    next_grids:
        torch tensor of shape (batch_size, padded_length, 3)
    """

    def merge(sequences, pad_value=0.0, pad=False):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        if not pad:
            padded_seqs = torch.zeros(len(sequences), max(lengths), dim)
        else:
            padded_seqs = torch.zeros(len(sequences), max(lengths), dim) + pad_value

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def batch_graph(graphs):
        lengths = [len(graph) for graph in graphs]
        padded_graphs = [empty_graph() for _ in range(len(graphs) * max(lengths))]
        for i in range(len(graphs)):
            padded_graphs[i * max(lengths): i * max(lengths) + lengths[i]] = graphs[i]
        return dgl.batch(padded_graphs), lengths

    # sort a list by source sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_grid_seqs, src_gps_seqs, src_pro_feas, trg_gps_seqs, trg_rids, trg_rates, \
    constraint_mat_trgs, constraint_graph_srcs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_grid_seqs, src_lengths = merge(src_grid_seqs)
    src_gps_seqs, _ = merge(src_gps_seqs)
    src_pro_feas = torch.tensor([list(src_pro_fea) for src_pro_fea in src_pro_feas])
    trg_gps_seqs, trg_lengths = merge(trg_gps_seqs)
    trg_rids, _ = merge(trg_rids)
    trg_rates, _ = merge(trg_rates)

    constraint_mat_trgs, _ = merge(constraint_mat_trgs, pad_value=1e-6, pad=True)
    constraint_graph_srcs, _ = batch_graph(constraint_graph_srcs)

    return src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, \
           constraint_mat_trgs, constraint_graph_srcs