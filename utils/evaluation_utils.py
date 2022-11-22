def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s

    return new_seq


def memoize(fn):
    '''
    Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function
    is called repeatedly with the same arguments.
    '''
    cache = dict()

    def wrapped(*v):
        key = tuple(v)  # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]

    return wrapped

from utils.shortest_path_func import SPSolver
def seq2path(seq: list, spsolver: SPSolver) -> list:
    path = set()
    for (st,en) in zip(seq[:-1], seq[1:]):
        subpath = spsolver.sp(st, en)
        for rid in subpath:
            path.add(rid)
    return list(path)



def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''

    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []

    return lcs_(len(xs), len(ys))


def cal_id_acc(predict, target):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len]
        target = [seq len]
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    assert len(predict) == len(target)
    ttl = len(predict)
    cnt = np.sum(np.array(predict) == np.array(target))

    # compute average rid accuracy
    shr_trg_ids = shrink_seq(target)
    shr_pre_ids = shrink_seq(predict)
    correct_id_num = len(lcs(shr_trg_ids, shr_pre_ids))
    ttl_trg_id_num = len(shr_trg_ids)
    ttl_pre_id_num = len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    if rid_precision + rid_recall < 1e-6:
        rid_f1 = 0
    else:
        rid_f1 = 2 * rid_recall * rid_precision / (rid_precision + rid_recall)
    return rid_acc, rid_recall, rid_precision, rid_f1


def cal_id_acc_v2(sp_solver: SPSolver, predict, target):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    Use actual path instead of sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len]
        target = [seq len]
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    assert len(predict) == len(target)
    ttl = len(predict)
    cnt = np.sum(np.array(predict) == np.array(target))

    # compute average rid accuracy
    shr_trg_ids = seq2path(shrink_seq(target), sp_solver)
    shr_pre_ids = seq2path(shrink_seq(predict), sp_solver)
    correct_id_num = len(lcs(shr_trg_ids, shr_pre_ids))
    ttl_trg_id_num = len(shr_trg_ids)
    ttl_pre_id_num = len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    if rid_precision + rid_recall < 1e-6:
        rid_f1 = 0
    else:
        rid_f1 = 2 * rid_recall * rid_precision / (rid_precision + rid_recall)
    return rid_acc, rid_recall, rid_precision, rid_f1


from utils.spatial_func import *


def cal_rn_dis_loss(sp_solver, predict_gps, predict_id, target_gps, target_id, rn_flag=True):
    """
    Calculate road network based MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        sp_solver: shortest path solver
        predict_gps = [seq len, 2]
        predict_id = [seq len]
        target_gps = [seq len, 2]
        target_id = [seq len]

        predict and target have been removed sos
    Returns:
    -------
        MAE in meter.
        RMSE in meter.
    """
    ls_dis = []
    rn_ls_dis = []
    assert len(predict_id) == len(predict_gps) == len(target_id) == len(target_gps)
    trg_len = len(predict_id)

    for i in range(trg_len):
        ls_dis.append(distance(SPoint(*predict_gps[i]), SPoint(*target_gps[i])))
        if rn_flag:
            rn_ls_dis.append(sp_solver.cal_sp_dist(SPoint(*predict_gps[i]), SPoint(*target_gps[i]),
                                               int(predict_id[i]), int(target_id[i])))

    ls_dis = np.array(ls_dis)
    rn_ls_dis = np.array(rn_ls_dis)

    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis ** 2).mean())
    if rn_flag:
        rn_mae = rn_ls_dis.mean()
        rn_rmse = np.sqrt((rn_ls_dis ** 2).mean())
        return mae, rmse, rn_mae, rn_rmse
    else:
        return mae, rmse, None, None

import torch
def cal_id_acc_batch(predict, target, trg_len, rn, inverse_flag=True, reduction='mean'):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len, batch size, id one hot output dim] in torch
        target = [seq len, batch size, 1] in torch
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, id dim]
    target = target.permute(1, 0)  # [batch size, seq len, 1]
    bs = predict.size(0)
    sl = predict.size(1)
    target = target.reshape(bs, sl)

    predict = predict.argmax(dim=-1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    rid_acc_bs = []
    rid_recall_bs = []
    rid_precision_bs = []
    rid_f1_bs = []

    for i in range(bs):
        tmp_predict = predict[i, :trg_len[i]]
        tmp_target = target[i, :trg_len[i]]
        if inverse_flag:
            tmp_predict = [rn.valid_to_origin_one[rid] for rid in tmp_predict]
            tmp_target = [rn.valid_to_origin_one[rid] for rid in tmp_target]
        rid_acc, rid_recall, rid_precision, rid_f1 = cal_id_acc(tmp_predict, tmp_target)
        rid_acc_bs.append(rid_acc)
        rid_recall_bs.append(rid_recall)
        rid_precision_bs.append(rid_precision)
        rid_f1_bs.append(rid_f1)

    if reduction == 'mean':
        return np.mean(rid_acc_bs), np.mean(rid_recall_bs), np.mean(rid_precision_bs), np.mean(rid_f1_bs)
    elif reduction == 'sum':
        return np.sum(rid_acc_bs), np.sum(rid_recall_bs), np.sum(rid_precision_bs), np.sum(rid_f1_bs)
    else:
        return rid_acc_bs, rid_recall_bs, rid_precision_bs, rid_f1_bs


def toseq(rn, rids, rates):
    """
    Convert batched rids and rates to gps sequence.
    Args:
    -----
    rn_dict:
        use for rate2gps()
    rids:
        [trg len, batch size, id one hot dim] in torch
    rates:
        [trg len, batch size] in torch
    Returns:
    --------
    seqs:
        [trg len, batch size, 2] in torch
    """

    batch_size = rids.shape[1]
    trg_len = rids.size(0)
    seqs = torch.zeros(trg_len, batch_size, 2)
    rates = rates.detach().cpu().numpy()
    rids = rids.detach().cpu().numpy()

    for i in range(1, trg_len):
        for bs in range(batch_size):
            if rids[i][bs].argmax() != 0:
                rid = rn.valid_to_origin_one[rids[i][bs].argmax()]
                rate = rates[i][bs]
                pt = rate2gps(rn, rid, rate)
                seqs[i][bs][0] = pt.lat
                seqs[i][bs][1] = pt.lng
            else:
                seqs[i][bs][0] = (rn.zone_range[0] + rn.zone_range[2]) / 2
                seqs[i][bs][1] = (rn.zone_range[1] + rn.zone_range[3]) / 2
    return seqs


def cal_rn_dis_loss_batch(sp_solver, rn, predict_gps, predict_id, target_gps, target_id, trg_len, rn_flag=False,
                          inverse_flag=True, reduction='mean'):
    """
    Calculate road network based MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        predict_gps = [seq len, batch size, 2]
        predict_id = [seq len, batch size, id one hot output dim]
        target_gps = [seq len, batch size, 2]
        target_id = [seq len, batch size]
        trg_len = [batch size]  if not considering target length, the loss will smaller than the real one.

        predict and target have been removed sos
    Returns:
    -------
        MAE of a batch in meter.
        RMSE of a batch in meter.
    """
    batch_size = target_id.size(1)
    predict_gps = predict_gps.permute(1, 0, 2).detach().cpu().numpy()
    predict_id = predict_id.permute(1, 0, 2).argmax(dim=-1).detach().cpu().numpy()
    target_gps = target_gps.permute(1, 0, 2).detach().cpu().numpy()
    target_id = target_id.permute(1, 0).detach().cpu().numpy()

    mae_bs = []
    rmse_bs = []
    rn_mae_bs = []
    rn_rmse_bs = []

    for i in range(batch_size):
        tmp_predict_gps = predict_gps[i, :trg_len[i], :]
        tmp_predict_id = predict_id[i, :trg_len[i]]
        tmp_target_gps = target_gps[i, :trg_len[i], :]
        tmp_target_id = target_id[i, :trg_len[i]]
        if inverse_flag:
            tmp_predict_id = [rn.valid_to_origin_one[rid] for rid in tmp_predict_id]
            tmp_target_id = [rn.valid_to_origin_one[rid] for rid in tmp_target_id]

        mae, rmse, rn_mae, rn_rmse = cal_rn_dis_loss(sp_solver, tmp_predict_gps, tmp_predict_id,
                                                     tmp_target_gps, tmp_target_id, rn_flag)
        mae_bs.append(mae)
        rmse_bs.append(rmse)
        rn_mae_bs.append(rn_mae)
        rn_rmse_bs.append(rn_rmse)
    if rn_flag:
        if reduction == 'mean':
            return np.mean(mae_bs), np.mean(rmse_bs), np.mean(rn_mae_bs), np.mean(rn_rmse_bs)
        elif reduction == 'sum':
            return np.sum(mae_bs), np.sum(rmse_bs), np.sum(rn_mae_bs), np.sum(rn_rmse_bs)
        else:
            return mae_bs, rmse_bs, rn_mae_bs, rn_rmse_bs
    else:
        if reduction == 'mean':
            return np.mean(mae_bs), np.mean(rmse_bs), None, None
        elif reduction == 'sum':
            return np.sum(mae_bs), np.sum(rmse_bs), None, None
        else:
            return mae_bs, rmse_bs, None, None

