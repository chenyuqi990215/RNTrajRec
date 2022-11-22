import dgl
import torch
from queue import Queue
from sklearn.neighbors import KDTree
from utils.spatial_func import rate2gps
from utils.candidate_point import CandidatePoint
import numpy as np


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
        g.ndata['gt'] = torch.tensor([[1]]).float()
    return g


def gps2grid(pt, mbr, grid_size):
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


def get_rn_grid(mbr, rn, grid_size):
    rn_grid = []
    rn_grid.append(torch.tensor([[0, 0]]))
    for i in range(1, rn.valid_edge_cnt_one):
        rid = rn.valid_to_origin_one[i]
        cur_grid = []
        for rate in range(1000):
            r = rate / 1000
            gps = rate2gps(rn, rid, r)
            grid_x, grid_y = gps2grid(gps, mbr, grid_size)
            if len(cur_grid) == 0 or [grid_x, grid_y] != cur_grid[-1]:
                cur_grid.append([grid_x, grid_y])
        rn_grid.append(torch.tensor(cur_grid))
    return rn_grid