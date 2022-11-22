import torch
from utils.mbr import MBR
from utils.spatial_func import LAT_PER_METER, LNG_PER_METER
from utils.spatial_func import SPoint, EARTH_MEAN_RADIUS_METER
from utils.graph_func import empty_graph
import dgl


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_dis_prob_vec(gps, rn, parameters, search_dist=None, beta=None):
    """
    Args:
    -----
    gps: [SPoint, tid]
    """
    if search_dist is None:
        search_dist = parameters.search_dist
    if beta is None:
        beta = parameters.beta
    cons_vec = torch.zeros(parameters.id_size)
    mbr = MBR(gps[0].lat - search_dist * LAT_PER_METER,
              gps[0].lng - search_dist * LNG_PER_METER,
              gps[0].lat + search_dist * LAT_PER_METER,
              gps[0].lng + search_dist * LNG_PER_METER)
    candis = rn.get_candidates(gps[0], mbr)
    if candis is not None:
        for candi_pt in candis:
            new_rid = rn.valid_edge_one[candi_pt.eid]
            cons_vec[new_rid] = exp_prob(beta, candi_pt.error)
    else:
        cons_vec = torch.ones(parameters.id_size)
    return cons_vec


import math


def exp_prob(beta, x):
    """
    error distance weight.
    """
    return math.exp(-pow(x, 2) / pow(beta, 2))


def get_reachable_inds(parameters):
    reachable_inds = list(range(parameters.id_size))

    return reachable_inds


def get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths, trg_lengths, rn, parameters):
    max_trg_len = max(trg_lengths)
    max_src_len = max(src_lengths)
    batch_size = src_grid_seqs.size(0)
    constraint_mat_trg = torch.zeros(batch_size, max_trg_len, parameters.id_size) + 1e-6
    constraint_mat_src = torch.zeros(batch_size, max_src_len, parameters.id_size)

    for bs in range(batch_size):
        # first src gps
        pre_t = 1
        pre_gps = [SPoint(src_gps_seqs[bs][pre_t][0].tolist(),
                          src_gps_seqs[bs][pre_t][1].tolist()),
                   pre_t]

        if parameters.dis_prob_mask_flag:
            constraint_mat_src[bs][pre_t] = get_dis_prob_vec(pre_gps, rn, parameters,
                                                             parameters.neighbor_dist, parameters.gamma)
            constraint_mat_trg[bs][pre_t] = get_dis_prob_vec(pre_gps, rn, parameters)
        else:
            reachable_inds = get_reachable_inds(parameters)
            constraint_mat_trg[bs][pre_t][reachable_inds] = 1

        # missed gps
        for i in range(2, src_lengths[bs]):
            cur_t = int(src_grid_seqs[bs, i, 2].tolist())
            cur_gps = [SPoint(src_gps_seqs[bs][i][0].tolist(),
                              src_gps_seqs[bs][i][1].tolist()),
                       cur_t]

            time_diff = cur_t - pre_t
            reachable_inds = get_reachable_inds(parameters)

            for t in range(pre_t + 1, cur_t):
                constraint_mat_trg[bs][t][reachable_inds] = 1

            # middle src gps
            if parameters.dis_prob_mask_flag:
                constraint_mat_src[bs][i] = get_dis_prob_vec(cur_gps, rn, parameters,
                                                             parameters.neighbor_dist, parameters.gamma)
                constraint_mat_trg[bs][cur_t] = get_dis_prob_vec(cur_gps, rn, parameters)
            else:
                reachable_inds = get_reachable_inds(parameters)
                constraint_mat_trg[bs][cur_t][reachable_inds] = 1
            pre_t = cur_t

    constraint_mat_trg = torch.clip(constraint_mat_trg, 1e-6, 1)
    return constraint_mat_trg, constraint_mat_src


def get_gps_subgraph(constraint_mat_src, src_grid_seq, trg_rid, parameters):
    total_g = parameters.g
    gps_subgraph = [empty_graph()]
    for i in range(1, constraint_mat_src.size(0)):
        sub = dgl.DGLGraph()
        nodes = torch.where(constraint_mat_src[i] > 0)[0].numpy().tolist()
        if trg_rid[src_grid_seq[i][-1]] not in nodes:
            nodes.append(trg_rid[src_grid_seq[i][-1]].item())
        _, neighbor = total_g.out_edges(nodes)
        nodes = list(set.union(set(nodes), set(neighbor.numpy().tolist())))
        sub.add_nodes(len(nodes))
        sub.ndata['id'] = torch.tensor(nodes)
        nmap = {}
        for (k, rid) in enumerate(nodes):
            nmap[rid] = k
        src, dst, w = [], [], []
        for rid in nodes:
            w.append(constraint_mat_src[i][rid])
            _, neighbor = total_g.out_edges([rid])
            for nrid in neighbor:
                if nrid.item() in nmap:
                    if rid != nrid.item():
                        src.append(nmap[rid])
                        dst.append(nmap[nrid.item()])
        sub.add_edges(src, dst)
        # sub.ndata['w'] = torch.tensor(w).reshape(-1, 1) / sum(w)
        sub.ndata['w'] = torch.tensor(w).reshape(-1, 1)
        sub.ndata['gt'] = torch.zeros_like(sub.ndata['w'])
        sub.ndata['gt'][nmap[trg_rid[src_grid_seq[i][-1]].item()], :] = 1
        sub = dgl.add_self_loop(sub)
        gps_subgraph.append(sub)
    return gps_subgraph
