import dgl
import torch
from queue import Queue
from sklearn.neighbors import KDTree
from utils.spatial_func import rate2gps
from utils.candidate_point import CandidatePoint


def get_total_graph(rn):
    src, dst = [], []
    g = dgl.DGLGraph()
    g.add_nodes(rn.valid_edge_cnt_one)
    for rid in rn.valid_edge:
        for nrid in rn.edgeDict[rid]:
            if nrid in rn.valid_edge:
                src.append(rn.valid_edge_one[rid])
                dst.append(rn.valid_edge_one[nrid])
    g.add_edges(src, dst)
    g = dgl.add_self_loop(g)
    return g


def get_sub_graphs(rn, max_deps=3):
    num_nodes = rn.valid_edge_cnt_one
    g = [dgl.DGLGraph() for _ in range(num_nodes)]
    g[0].add_nodes(1)
    g[0] = dgl.add_self_loop(g[0])
    g[0].ndata['id'] = torch.tensor([0])
    for i in range(1, num_nodes):
        rid = rn.valid_to_origin_one[i]
        rset = set()
        cset = set()
        src, dst = [], []
        q = Queue()
        q.put((rid, 0))
        rset.add(rid)
        while not q.empty():
            rid, dep = q.get()
            if dep == max_deps:
                continue
            if rid in cset:
                continue
            cset.add(rid)
            for nrid in rn.edgeDict[rid]:
                if nrid in rn.valid_edge:
                    src.append(rid)
                    dst.append(nrid)
                    rset.add(nrid)
                    q.put((nrid, dep + 1))
        g[i].add_nodes(len(rset))
        rset = list(rset)
        mset = [rn.valid_edge_one[rid] for rid in rset]
        rmap = {}
        for j in range(len(rset)):
            rmap[rset[j]] = j
        src = [rmap[rid] for rid in src]
        dst = [rmap[rid] for rid in dst]
        g[i].add_edge(src, dst)
        g[i].ndata['id'] = torch.tensor(mset)
        g[i] = dgl.add_self_loop(g[i])
    return g


def empty_graph(add_weight=True):
    g = dgl.DGLGraph()
    g.add_nodes(1)
    g = dgl.add_self_loop(g)
    g.ndata['id'] = torch.tensor([0])
    if add_weight:
        g.ndata['w'] = torch.tensor([[1]]).float()
    return g


def fetch_poi(rn, poi):
    poi_list = []
    poi_point = []
    poi_rn = {}
    for rid in rn.valid_edge:
        poi_rn[rid] = []
    for cluster in poi:
        for cand in cluster:
            if cand.eid in rn.valid_edge:
                poi_rn[cand.eid].append(cand)
    missing = 0
    for rid in rn.valid_edge:
        if len(poi_rn[rid]) == 0:
            missing += 1
            gps = rate2gps(rn, rid, 0.5)
            cand = CandidatePoint(gps.lat, gps.lng, rid, 0, 0.5 * rn.edgeDis[rid], 0.5)
            poi_list.append(cand)
            poi_point.append([cand.lat, cand.lng])
        else:
            for cand in poi_rn[rid]:
                if cand.eid in rn.valid_edge:
                    poi_list.append(cand)
                    poi_point.append([cand.lat, cand.lng])

    tree = KDTree(poi_point)
    print(f'Missing RN count: {missing}')
    return poi_list, tree


def get_poi_graph(rn, poi_list):
    g = dgl.DGLGraph()
    g.add_nodes(len(poi_list))
    rn_poi = {}
    for rid in rn.valid_edge:
        rn_poi[rid] = []
    for i, cand in enumerate(poi_list):
        assert cand.eid in rn.valid_edge
        rn_poi[cand.eid].append((i, cand.rate, cand))
    src, dst = [], []
    for rid in rn.valid_edge:
        rn_poi[rid] = sorted(rn_poi[rid], key=lambda x: x[1])

    for rid in rn.valid_edge:
        for (st, en) in zip(rn_poi[rid][:-1], rn_poi[rid][1:]):
            src.append(st[0])
            dst.append(en[0])
        for nrid in rn.edgeDict[rid]:
            if nrid in rn.valid_edge:
                src.append(rn_poi[rid][-1][0])
                dst.append(rn_poi[nrid][0][0])
    g.add_edges(src, dst)
    g = dgl.add_self_loop(g)

    dg = dgl.DGLGraph()
    dg.add_nodes(len(poi_list) + rn.valid_edge_cnt_one)

    src, dst = [], []
    for (i, cand) in enumerate(poi_list):
        eid = rn.valid_edge_one[cand.eid]
        src.append(eid)
        dst.append(rn.valid_edge_cnt_one + i)
    dg.add_edges(src, dst)
    dg.add_edges(dst, src)
    dg = dgl.add_self_loop(dg)
    return g, dg