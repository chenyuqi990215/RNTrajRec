import math
from queue import PriorityQueue, Queue
import numpy as np
import sys

sys.setrecursionlimit(100000)
from tqdm import tqdm

from multiprocessing import Pool
from utils.spatial_func import *
from utils.mbr import MBR
from math import *
from rtree import Rtree
from utils.candidate_point import CandidatePoint

class RoadNetworkMap:
    def __init__(self, dir, zone_range, unit_length):
        edgeFile = open(dir + '/edgeOSM.txt')
        self.edgeDis = []
        self.edgeNode = []
        self.edgeCord = []
        self.edgeOffset = []
        self.nodeSet = set()
        self.nodeDict = {}
        self.edgeDict = {}
        self.edgeRevDict = {}
        self.nodeEdgeDict = {}
        self.nodeEdgeRevDict = {}
        self.zone_range = zone_range
        self.unit_length = unit_length
        self.minLat = 1e18
        self.maxLat = -1e18
        self.minLon = 1e18
        self.maxLon = -1e18
        self.edgeNum = 0
        self.nodeNum = 0
        self.valid_edge = {}
        self.valid_to_origin = {}
        self.valid_edge_cnt = 0

        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}

        for line in edgeFile.readlines():
            item_list = line.strip().split()
            a = int(item_list[1])
            b = int(item_list[2])
            self.edgeNode.append((a, b))
            self.nodeDict[a] = b
            if a not in self.nodeEdgeDict:
                self.nodeEdgeDict[a] = []
            if b not in self.nodeEdgeRevDict:
                self.nodeEdgeRevDict[b] = []
            self.nodeEdgeDict[a].append(self.edgeNum)
            self.nodeEdgeRevDict[b].append(self.edgeNum)
            self.nodeSet.add(a)
            self.nodeSet.add(b)
            num = int(item_list[3])
            dist = 0
            # print (item_list)
            self.edgeCord.append(list(map(float, item_list[4:])))
            inzone_flag = True
            for i in range(num):
                tmplat = float(item_list[4 + i * 2])
                tmplon = float(item_list[5 + i * 2])
                self.minLat = min(self.minLat, tmplat)
                self.maxLat = max(self.maxLat, tmplat)
                self.minLon = min(self.minLon, tmplon)
                self.maxLon = max(self.maxLon, tmplon)
                inzone_flag = inzone_flag and self.inside_zone(tmplat, tmplon)

            if inzone_flag:
                self.valid_edge[self.edgeNum] = self.valid_edge_cnt
                self.valid_to_origin[self.valid_edge_cnt] = self.edgeNum
                self.valid_edge_cnt += 1

            offset = []
            for i in range(num - 1):
                dist += self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2]))
                offset.append(self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2])))
            self.edgeDis.append(dist)
            for i in range(len(offset) - 1, 0, -1):
                offset[i - 1] = offset[i - 1] + offset[i]
            offset.append(0)

            self.edgeOffset.append(offset)
            self.edgeNum += 1

        self.valid_edge_one = {}
        for (key, value) in self.valid_edge.items():
            self.valid_edge_one[key] = value + 1
        self.valid_to_origin_one = {}
        for (key, value) in self.valid_to_origin.items():
            self.valid_to_origin_one[key + 1] = value
        self.valid_edge_cnt_one = self.valid_edge_cnt + 1

        mid_point = SPoint((self.zone_range[0] + self.zone_range[2]) / 2,
                            (self.zone_range[1] + self.zone_range[3]) / 2)
        min_dist = 1e18
        best_cand = -1
        for rid in self.valid_edge.keys():
            gps, rate, dist = project_pt_to_road(self, mid_point, rid)
            if dist < min_dist:
                min_dist = dist
                best_cand = rid
        self.valid_to_origin_one[0] = best_cand  # sos reflect to the middle of the road network.

        self.nodeNum = len(self.nodeSet)
        self.mbr = MBR(*zone_range)

        for eid in range(self.edgeNum):
            self.edgeRevDict[eid] = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            self.edgeDict[eid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    self.edgeDict[eid].append(nid)
                    self.edgeRevDict[nid].append(eid)

        # self.igraph = igraph.Graph(directed=True)
        #  self.igraph.add_vertices(self.nodeNum)
        edge_list = []
        edge_weight_list = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            if (a == b):
                continue
            edge_list.append((a, b))
            edge_weight_list.append(self.edgeDis[eid])

        # self.igraph.add_edges(edge_list)
        #   self.igraph.es['dis'] = edge_weight_list

        print('edge Num: ', self.edgeNum)
        print('node Num: ', self.nodeNum)
        print('valid edge Num: ', self.valid_edge_cnt)

        self.wayType = {}

        wayFile = open(dir + '/wayTypeOSM.txt')
        for line in wayFile.readlines():
            item_list = line.strip().split()
            roadId = int(item_list[0])
            wayId = int(item_list[-1])
            self.wayType[roadId] = wayId

        self.long_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0],
                                                self.zone_range[3]) + unit_length - 1) / unit_length)

        self.width_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2],
                                                       self.zone_range[1]) + unit_length - 1) / unit_length)

        self.cnn_graph = np.zeros((self.long_num, self.width_num)).astype(int)
        self.cnn_to_edge = np.ones((self.long_num, self.width_num)).astype(int) * self.valid_edge_cnt
     #   self.cnn_to_edge_set = [[set() for j in range(self.width_num)] for i in range(self.long_num)]

        print('long length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0], self.zone_range[3]),
              self.long_num)
        print('width length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2], self.zone_range[1]),
              self.width_num)

        print ('----- construct cnn graph ------')
    #    self.confict_num = 0
        self.construct_cnn_graph()

    def center_geolocation(self, cluster):
        x = y = z = 0
        coord_num = len(cluster)
        for coord in cluster:
            lat, lon = radians(coord[0]), radians(coord[1])
            x += cos(lat) * cos(lon)
            y += cos(lat) * sin(lon)
            z += sin(lat)
        x /= coord_num
        y /= coord_num
        z /= coord_num
        return (degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x)))

    def DDALine(self, x1, y1, x2, y2, valid_edge_id, way_id):
        if self.cnn_graph[x1,y1] < way_id:
            self.cnn_graph[x1,y1] = way_id
            self.cnn_to_edge[x1,y1] = valid_edge_id

        if (x1 == x2 and y1 == y2):
            return

        dx = x2 - x1
        dy = y2 - y1
        steps = 0
        # 斜率判断
        if abs(dx) > abs(dy):
            steps = abs(dx)
        else:
            steps = abs(dy)
        # 必有一个等于1，一个小于1
        delta_x = float(dx / steps)
        delta_y = float(dy / steps)
        # 四舍五入，保证x和y的增量小于等于1，让生成的直线尽量均匀
        x = x1 + 0.5
        y = y1 + 0.5
        for i in range(0, int(steps + 1)):
            # 绘制像素点
            if self.cnn_graph[int(x),int(y)] < way_id:
                self.cnn_graph[int(x),int(y)] = way_id
                self.cnn_to_edge[int(x),int(y)] = valid_edge_id
            x += delta_x
            y += delta_y

    def get_cnn_id(self, lat, lon):
        x = int ( (lon - self.zone_range[1]) / ((self.zone_range[3] - self.zone_range[1]) / self.long_num))
        y = int ( (lat - self.zone_range[0]) / ((self.zone_range[2] - self.zone_range[0]) / self.width_num))
        return x,y

    def draw_cnn_line(self, stlat, stlon, edlat, edlon, valid_edge_id, way_id):
        stx, sty = self.get_cnn_id(stlat, stlon)
        edx, edy = self.get_cnn_id(edlat, edlon)

        self.DDALine(stx, sty, edx, edy, valid_edge_id, way_id)


    def construct_cnn_graph(self):
        for i in range(self.edgeNum):
            if i in self.valid_edge:
                cord_len = len(self.edgeCord[i])
                for j in range(cord_len // 2 - 2):
                    stlat = self.edgeCord[i][j * 2]
                    stlon = self.edgeCord[i][j * 2 + 1]
                    edlat = self.edgeCord[i][j * 2 + 2]
                    edlon = self.edgeCord[i][j * 2 + 3]
                    self.draw_cnn_line(stlat, stlon, edlat, edlon, self.valid_edge[i], self.wayType[i])



    def inside_zone(self, lat, lon):
        return self.zone_range[0] <= lat and lat <= self.zone_range[2] and self.zone_range[1] <= lon and lon <= \
               self.zone_range[3]

    def calSpatialDistance(self, x1, y1, x2, y2):
        lat1 = (math.pi / 180.0) * x1
        lat2 = (math.pi / 180.0) * x2
        lon1 = (math.pi / 180.0) * y1
        lon2 = (math.pi / 180.0) * y2
        R = 6378.137
        t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        if t > 1.0:
            t = 1.0
        d = math.acos(t) * R * 1000
        return d

    def edgeDistance(self, edgeId):
        return self.edgeDis[edgeId]

    def getEdgeNode(self, edgeId):
        return self.edgeNode[edgeId]

    def shortestPathAll(self, start, end=-1, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if dis > max_len:
                if end != -1:
                    return 1e18, []
                else:
                    return dis, pred
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestAStarPath(self, start, end, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        st = self.edgeCord[start][-2:]
        en = self.edgeCord[end][-2:]
        pq.put((self.calSpatialDistance(*st, *en), 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while pq.qsize():
            h, dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if h > max_len:
                return 1e18, []
            for nid in self.edgeDict[id]:
                if nid in self.valid_edge:
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        st = self.edgeCord[nid][-2:]
                        pq.put((dist[nid] + self.calSpatialDistance(*st, *en), dist[nid], nid))
        if not with_route:
            pred = []
        return dist[end], pred

    def dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    def cal_angle(self, eid, nid):
        ex, ey = self.edgeCord[eid][-2] - self.edgeCord[eid][0], self.edgeCord[eid][-1] - self.edgeCord[eid][1]
        nx, ny = self.edgeCord[nid][-2] - self.edgeCord[nid][0], self.edgeCord[nid][-1] - self.edgeCord[nid][1]
        v1 = (ex, ey)
        v2 = (nx, ny)
        if (self.length(v1) < 1e-5 or self.length(v2) < 1e-5):
            return 0
        else:
            return math.acos(round(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2))))

    def shortestRankPathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        dist2 = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        dist2[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, dis2, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_rank = self.wayType[id] > self.wayType[nid]
                    if (dist[nid] > dist[id] + en_rank) or (
                            dist[nid] == dist[id] + en_rank and dist2[nid] > dist2[id] + self.edgeDis[nid]):
                        dist[nid] = dist[id] + en_rank
                        dist2[nid] = dist2[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], dist2[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestAnglePathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_angle = self.cal_angle(id, nid)
                    if dist[nid] > dist[id] + en_angle:
                        dist[nid] = dist[id] + en_angle
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestPath(self, start, end, stype='slen', with_route=True, max_len=1e18):
        # start = self.valid_to_origin[start]
        #  end = self.valid_to_origin[end]
        if stype == 'slen':
            if end != -1:
                dis, pred = self.shortestAStarPath(start, end, with_route, max_len)
            else:
                dis, pred = self.shortestPathAll(start, end, with_route, max_len)
        elif stype == 'rlen':
            dis, pred = self.shortestRankPathAll(start, end, with_route)
        elif stype == 'alen':
            dis, pred = self.shortestAnglePathAll(start, end, with_route)

        if end == -1:
            return dis, pred
        if dis > max_len:
            return dis, []

        if with_route:
            id = end
            # print ('route: ')
            arr = [id]
            while pred[id] >= 0 and pred[id] < 1e18:
                #  print (pred[id],end=',')
                id = pred[id]
                arr.append(id)
            arr = list(reversed(arr))
            #    arr = [self.valid_edge[item] for item in arr]
            return dis, arr
        else:
            return dis

    def output_dataset_part(self, data_file, start, end, w):

        f = open(data_file + '_%d_%d' % (start, end), 'w')

        for eid in tqdm(range(start, end)):
            if eid in self.valid_edge:
                connect = [eid]
                sdis, spred = self.shortestPath(start=eid, end=-1, stype='slen')
                rdis, rpred = self.shortestPath(start=eid, end=-1, stype='rlen')
                adis, apred = self.shortestPath(start=eid, end=-1, stype='alen')
                for nid in range(self.edgeNum):
                    if (nid in self.valid_edge) and (eid != nid) and sdis[nid] < 1e18 and rdis[nid] < 1e18 and adis[
                        nid] < 1e18:
                        connect.append(nid)
                if len(connect) > 1:
                    des_list = np.random.choice(connect[1:], size=min(len(connect[1:]), w), replace=False)
                    for des in des_list:
                        for ipred in [spred, rpred, apred]:
                            id = des
                            # print ('route: ')
                            arr = [id]
                            while (ipred[id] >= 0 and ipred[id] < 1e18):
                                #  print (pred[id],end=',')
                                id = ipred[id]
                                arr.append(id)
                            arr = list(reversed(arr))
                            f.write(' '.join(list(map(str, arr))) + '\n')

    def output_dataset(self, data_size, data_num, data_file):
        w = data_size // self.valid_edge_cnt
        p = Pool(data_num)
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            if start < end:
                p.apply_async(self.output_dataset_part, args=(data_file, start, end, w))

        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

    def output_train_dataset(self, output_data_file, data_num, data_file):
        g = open(output_data_file, 'w')
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            f = open(data_file + '_%d_%d' % (start, end), 'r')
            for line in tqdm(f.readlines()):
                g.write(line)


class RoadNetworkMapFull:
    def __init__(self, dir, zone_range, unit_length):
        edgeFile = open(dir + '/edgeOSM.txt')
        self.edgeDis = []
        self.edgeNode = []
        self.edgeCord = []
        self.edgeOffset = []
        self.nodeSet = set()
        self.nodeDict = {}
        self.edgeDict = {}
        self.edgeRevDict = {}
        self.nodeEdgeDict = {}
        self.nodeEdgeRevDict = {}
        self.zone_range = zone_range
        self.unit_length = unit_length
        self.minLat = 1e18
        self.maxLat = -1e18
        self.minLon = 1e18
        self.maxLon = -1e18
        self.edgeNum = 0
        self.nodeNum = 0
        self.valid_edge = {}
        self.valid_to_origin = {}
        self.valid_edge_cnt = 0

        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}

        for line in edgeFile.readlines():
            item_list = line.strip().split()
            a = int(item_list[1])
            b = int(item_list[2])
            self.edgeNode.append((a, b))
            self.nodeDict[a] = b
            if a not in self.nodeEdgeDict:
                self.nodeEdgeDict[a] = []
            if b not in self.nodeEdgeRevDict:
                self.nodeEdgeRevDict[b] = []
            self.nodeEdgeDict[a].append(self.edgeNum)
            self.nodeEdgeRevDict[b].append(self.edgeNum)
            self.nodeSet.add(a)
            self.nodeSet.add(b)
            num = int(item_list[3])
            dist = 0
            # print (item_list)
            self.edgeCord.append(list(map(float, item_list[4:])))
            inzone_flag = True
            for i in range(num):
                tmplat = float(item_list[4 + i * 2])
                tmplon = float(item_list[5 + i * 2])
                self.minLat = min(self.minLat, tmplat)
                self.maxLat = max(self.maxLat, tmplat)
                self.minLon = min(self.minLon, tmplon)
                self.maxLon = max(self.maxLon, tmplon)
                inzone_flag = inzone_flag and self.inside_zone(tmplat, tmplon)

            if inzone_flag:
                self.valid_edge[self.edgeNum] = self.valid_edge_cnt
                self.valid_to_origin[self.valid_edge_cnt] = self.edgeNum
                self.valid_edge_cnt += 1

            offset = []
            for i in range(num - 1):
                dist += self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2]))
                offset.append(self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2])))
            self.edgeDis.append(dist)
            for i in range(len(offset) - 1, 0, -1):
                offset[i - 1] = offset[i - 1] + offset[i]
            offset.append(0)
            self.edgeOffset.append(offset)
            self.edgeNum += 1

        self.valid_edge_one = {}
        for (key, value) in self.valid_edge.items():
            self.valid_edge_one[key] = value + 1
        self.valid_to_origin_one = {}
        for (key, value) in self.valid_to_origin.items():
            self.valid_to_origin_one[key + 1] = value
        self.valid_edge_cnt_one = self.valid_edge_cnt + 1

        self.spatial_index = Rtree()
        for rid in self.valid_edge.keys():
            edge_cords = self.edgeCord[rid]
            cords = []
            for i in range(len(edge_cords) // 2):
                cords.append(SPoint(edge_cords[2 * i], edge_cords[2 * i + 1]))
            mbr = MBR.cal_mbr(cords)
            self.spatial_index.insert(rid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))

        mid_point = SPoint((self.zone_range[0] + self.zone_range[2]) / 2,
                            (self.zone_range[1] + self.zone_range[3]) / 2)
        min_dist = 1e18
        best_cand = -1
        for rid in self.valid_edge.keys():
            gps, rate, dist = project_pt_to_road(self, mid_point, rid)
            if dist < min_dist:
                min_dist = dist
                best_cand = rid
        self.valid_to_origin_one[0] = best_cand  # sos reflect to the middle of the road network.

        self.nodeNum = len(self.nodeSet)
        self.mbr = MBR(*zone_range)

        for eid in range(self.edgeNum):
            self.edgeRevDict[eid] = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            self.edgeDict[eid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    self.edgeDict[eid].append(nid)
                    self.edgeRevDict[nid].append(eid)

        # self.igraph = igraph.Graph(directed=True)
        #  self.igraph.add_vertices(self.nodeNum)
        edge_list = []
        edge_weight_list = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            if (a == b):
                continue
            edge_list.append((a, b))
            edge_weight_list.append(self.edgeDis[eid])

        # self.igraph.add_edges(edge_list)
        #   self.igraph.es['dis'] = edge_weight_list

        print('edge Num: ', self.edgeNum)
        print('node Num: ', self.nodeNum)
        print('valid edge Num: ', self.valid_edge_cnt)

        self.wayType = {}

        wayFile = open(dir + '/wayTypeOSM.txt')
        for line in wayFile.readlines():
            item_list = line.strip().split()
            roadId = int(item_list[0])
            wayId = int(item_list[-1])
            self.wayType[roadId] = wayId

        self.long_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0],
                                                self.zone_range[3]) + unit_length - 1) / unit_length)

        self.width_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2],
                                                       self.zone_range[1]) + unit_length - 1) / unit_length)

        self.cnn_graph = np.zeros((self.long_num, self.width_num)).astype(int)
        self.cnn_to_edge = np.ones((self.long_num, self.width_num)).astype(int) * self.valid_edge_cnt
     #   self.cnn_to_edge_set = [[set() for j in range(self.width_num)] for i in range(self.long_num)]

        print('long length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0], self.zone_range[3]),
              self.long_num)
        print('width length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2], self.zone_range[1]),
              self.width_num)

        print ('----- construct cnn graph ------')
    #    self.confict_num = 0
        self.construct_cnn_graph()

    def range_query(self, mbr: MBR) -> list:
        eids = self.spatial_index.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return list(eids)

    def nearest_query(self, gps: SPoint, return_type='spoint') -> SPoint:
        search_dist = 50
        while True:
            mbr = MBR(gps.lat - search_dist * LAT_PER_METER,
                      gps.lng - search_dist * LNG_PER_METER,
                      gps.lat + search_dist * LAT_PER_METER,
                      gps.lng + search_dist * LNG_PER_METER)
            candis = self.get_candidates(gps, mbr)
            if len(candis) > 0:
                best_cand = None
                min_err = 1e9
                for cand in candis:
                    if cand.error < min_err:
                        min_err = cand.error
                        if return_type == 'spoint':
                            best_cand = SPoint(cand.lat, cand.lng)
                        else:
                            best_cand = cand
                return best_cand
            else:
                search_dist = search_dist * 2

    def point_in_mbr(self, x: SPoint, mbr: MBR) -> bool:
        return mbr.min_lat <= x.lat <= mbr.max_lat and mbr.min_lng <= x.lng <= mbr.max_lng

    def get_candidates(self, x: SPoint, mbr: MBR) -> list:
        candi = self.range_query(mbr)
        refined_candi = []
        for eid in candi:
            projected, rate, dist = project_pt_to_road(self, x, eid)
            if self.point_in_mbr(projected, mbr):
                candidate = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.edgeDis[eid], rate)
                refined_candi.append(candidate)
        return refined_candi

    def DDALine(self, x1, y1, x2, y2, valid_edge_id, way_id):
        if self.cnn_graph[x1,y1] < way_id:
            self.cnn_graph[x1,y1] = way_id
            self.cnn_to_edge[x1,y1] = valid_edge_id

        if (x1 == x2 and y1 == y2):
            return

        dx = x2 - x1
        dy = y2 - y1
        steps = 0
        # 斜率判断
        if abs(dx) > abs(dy):
            steps = abs(dx)
        else:
            steps = abs(dy)
        # 必有一个等于1，一个小于1
        delta_x = float(dx / steps)
        delta_y = float(dy / steps)
        # 四舍五入，保证x和y的增量小于等于1，让生成的直线尽量均匀
        x = x1 + 0.5
        y = y1 + 0.5
        for i in range(0, int(steps + 1)):
            # 绘制像素点
            if self.cnn_graph[int(x),int(y)] < way_id:
                self.cnn_graph[int(x),int(y)] = way_id
                self.cnn_to_edge[int(x),int(y)] = valid_edge_id
            x += delta_x
            y += delta_y

    def get_cnn_id(self, lat, lon):
        x = int ( (lon - self.zone_range[1]) / ((self.zone_range[3] - self.zone_range[1]) / self.long_num))
        y = int ( (lat - self.zone_range[0]) / ((self.zone_range[2] - self.zone_range[0]) / self.width_num))
        return x,y

    def draw_cnn_line(self, stlat, stlon, edlat, edlon, valid_edge_id, way_id):
        stx, sty = self.get_cnn_id(stlat, stlon)
        edx, edy = self.get_cnn_id(edlat, edlon)

        self.DDALine(stx, sty, edx, edy, valid_edge_id, way_id)


    def construct_cnn_graph(self):
        for i in range(self.edgeNum):
            if i in self.valid_edge:
                cord_len = len(self.edgeCord[i])
                for j in range(cord_len // 2 - 2):
                    stlat = self.edgeCord[i][j * 2]
                    stlon = self.edgeCord[i][j * 2 + 1]
                    edlat = self.edgeCord[i][j * 2 + 2]
                    edlon = self.edgeCord[i][j * 2 + 3]
                    self.draw_cnn_line(stlat, stlon, edlat, edlon, self.valid_edge[i], self.wayType[i])



    def inside_zone(self, lat, lon):
        return self.zone_range[0] <= lat and lat <= self.zone_range[2] and self.zone_range[1] <= lon and lon <= \
               self.zone_range[3]

    def calSpatialDistance(self, x1, y1, x2, y2):
        lat1 = (math.pi / 180.0) * x1
        lat2 = (math.pi / 180.0) * x2
        lon1 = (math.pi / 180.0) * y1
        lon2 = (math.pi / 180.0) * y2
        R = 6378.137
        t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        if t > 1.0:
            t = 1.0
        d = math.acos(t) * R * 1000
        return d

    def edgeDistance(self, edgeId):
        return self.edgeDis[edgeId]

    def getEdgeNode(self, edgeId):
        return self.edgeNode[edgeId]

    def shortestPathAll(self, start, end=-1, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if dis > max_len:
                if end != -1:
                    return 1e18, []
                else:
                    return dis, pred
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestAStarPath(self, start, end, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        st = self.edgeCord[start][-2:]
        en = self.edgeCord[end][-2:]
        pq.put((self.calSpatialDistance(*st, *en), 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while pq.qsize():
            h, dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if h > max_len:
                return 1e18, []
            for nid in self.edgeDict[id]:
                if nid in self.valid_edge:
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        st = self.edgeCord[nid][-2:]
                        pq.put((dist[nid] + self.calSpatialDistance(*st, *en), dist[nid], nid))
        if not with_route:
            pred = []
        return dist[end], pred

    def dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    def cal_angle(self, eid, nid):
        ex, ey = self.edgeCord[eid][-2] - self.edgeCord[eid][0], self.edgeCord[eid][-1] - self.edgeCord[eid][1]
        nx, ny = self.edgeCord[nid][-2] - self.edgeCord[nid][0], self.edgeCord[nid][-1] - self.edgeCord[nid][1]
        v1 = (ex, ey)
        v2 = (nx, ny)
        if (self.length(v1) < 1e-5 or self.length(v2) < 1e-5):
            return 0
        else:
            return math.acos(round(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2))))

    def shortestRankPathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        dist2 = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        dist2[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, dis2, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_rank = self.wayType[id] > self.wayType[nid]
                    if (dist[nid] > dist[id] + en_rank) or (
                            dist[nid] == dist[id] + en_rank and dist2[nid] > dist2[id] + self.edgeDis[nid]):
                        dist[nid] = dist[id] + en_rank
                        dist2[nid] = dist2[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], dist2[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestAnglePathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_angle = self.cal_angle(id, nid)
                    if dist[nid] > dist[id] + en_angle:
                        dist[nid] = dist[id] + en_angle
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestPath(self, start, end, stype='slen', with_route=True, max_len=1e18):
        # start = self.valid_to_origin[start]
        #  end = self.valid_to_origin[end]
        if stype == 'slen':
            if end != -1:
                dis, pred = self.shortestAStarPath(start, end, with_route, max_len)
            else:
                dis, pred = self.shortestPathAll(start, end, with_route, max_len)
        elif stype == 'rlen':
            dis, pred = self.shortestRankPathAll(start, end, with_route)
        elif stype == 'alen':
            dis, pred = self.shortestAnglePathAll(start, end, with_route)

        if end == -1:
            return dis, pred
        if dis > max_len:
            return dis, []

        if with_route:
            id = end
            # print ('route: ')
            arr = [id]
            while pred[id] >= 0 and pred[id] < 1e18:
                #  print (pred[id],end=',')
                id = pred[id]
                arr.append(id)
            arr = list(reversed(arr))
            #    arr = [self.valid_edge[item] for item in arr]
            return dis, arr
        else:
            return dis

    def output_dataset_part(self, data_file, start, end, w):

        f = open(data_file + '_%d_%d' % (start, end), 'w')

        for eid in tqdm(range(start, end)):
            if eid in self.valid_edge:
                connect = [eid]
                sdis, spred = self.shortestPath(start=eid, end=-1, stype='slen')
                rdis, rpred = self.shortestPath(start=eid, end=-1, stype='rlen')
                adis, apred = self.shortestPath(start=eid, end=-1, stype='alen')
                for nid in range(self.edgeNum):
                    if (nid in self.valid_edge) and (eid != nid) and sdis[nid] < 1e18 and rdis[nid] < 1e18 and adis[
                        nid] < 1e18:
                        connect.append(nid)
                if len(connect) > 1:
                    des_list = np.random.choice(connect[1:], size=min(len(connect[1:]), w), replace=False)
                    for des in des_list:
                        for ipred in [spred, rpred, apred]:
                            id = des
                            # print ('route: ')
                            arr = [id]
                            while (ipred[id] >= 0 and ipred[id] < 1e18):
                                #  print (pred[id],end=',')
                                id = ipred[id]
                                arr.append(id)
                            arr = list(reversed(arr))
                            f.write(' '.join(list(map(str, arr))) + '\n')

    def output_dataset(self, data_size, data_num, data_file):
        w = data_size // self.valid_edge_cnt
        p = Pool(data_num)
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            if start < end:
                p.apply_async(self.output_dataset_part, args=(data_file, start, end, w))

        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

    def output_train_dataset(self, output_data_file, data_num, data_file):
        g = open(output_data_file, 'w')
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            f = open(data_file + '_%d_%d' % (start, end), 'r')
            for line in tqdm(f.readlines()):
                g.write(line)
