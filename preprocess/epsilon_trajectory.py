city = "Porto"
map_root = f"path/to/road_network"
epsilon = 10

import sys
sys.path.append('../')

from map import RoadNetworkMap

if city == "Porto":
    road_map = RoadNetworkMap(map_root, zone_range=[41.0776277, -8.8056464, 41.2203117, -8.4585492], unit_length=50)

import math
def Euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

from math import *
def haversine(lat1, lon1, lat2, lon2):
    lat1 = (math.pi / 180.0) * lat1
    lat2 = (math.pi / 180.0) * lat2
    lon1 = (math.pi / 180.0) * lon1
    lon2 = (math.pi / 180.0) * lon2
    R = 6378.137
    t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    if t > 1.0:
        t = 1.0
    d = math.acos(t) * R * 1000
    return d

def center_geolocation(cluster):
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

import numpy as np
def projectToSegment(x, line):
    l = line[0]
    r = line[1]
    cnt = 0
    while cnt < 30:
        # m1 = l + (r - l) / 3
        m1 = center_geolocation([list(l), list(l), list(r)])
        # m2 = r - (r - l) / 3
        m2 = center_geolocation([list(l), list(r), list(r)])
        if haversine(*x, *m1) > haversine(*x, *m2):
            l = m1
        else:
            r = m2
        cnt += 1
    return l

import re
def getTrajs(file_path):
    ret_records = []
    buffer_records = []
    data = open(file_path, 'r')
    for item in data:
        line = item.strip()
        line = re.split(' |,', line)
        if line[0][0] == '-':
            ret_records.append(buffer_records)
            buffer_records = []
        else:
            buffer_records.append(line)
    return ret_records

import numpy as np
def findSegment(gps, rid):
    x, y = gps
    if rid not in road_map.valid_edge.keys():
        return [-1, -1]
    edge_cords = np.array(road_map.edgeCord[rid]).reshape(-1, 2)
    dist = []
    gps = []
    for i in range(edge_cords.shape[0] - 1):
        gps.append(projectToSegment([x, y], [edge_cords[i], edge_cords[i + 1]]))
        dist.append(haversine(x, y, *gps[-1]))
    idx = np.argmin(dist)
    return idx

def findGPS(line, dist):
    st = line[0]
    l = line[0]
    r = line[1]
    if haversine(*l, *r) <= dist:
        return r
    cnt = 0
    while cnt < 30:
        mid = center_geolocation([list(l), list(r)])
        if haversine(*st, *mid) > dist:
            r = mid
        else:
            l = mid
        cnt += 1
    return l

def linearInterpolate(pre, post, curtime):
    pre_time = pre[0]
    post_time = post[0]
    if pre[-1] < 0 or post[-1] < 0:
        return [-1, -1, -1, -1]
    travel_path = road_map.shortestPath(pre[-1], post[-1], with_route=True)[1]
    if travel_path == []:
        return [-1, -1, -1, -1]
    travel_gps = []
    if pre[-1] != post[-1]:
        travel_gps.append([pre[1], pre[2], pre[-1]])
        idx = findSegment([pre[1], pre[2]], pre[-1])
        edge_cords = np.array(road_map.edgeCord[pre[-1]]).reshape(-1, 2).tolist()
        for i in range(idx + 1, len(edge_cords)):
            travel_gps.append([*edge_cords[i], pre[-1]])
        for i in range(1, len(travel_path) - 1, 1):
            edge_cords = np.array(road_map.edgeCord[travel_path[i]]).reshape(-1, 2).tolist()
            for gps in edge_cords:
                travel_gps.append([*gps, travel_path[i]])
        idx = findSegment([post[1], post[2]], post[-1])
        edge_cords = np.array(road_map.edgeCord[post[-1]]).reshape(-1, 2).tolist()
        for i in range(idx + 1):
            travel_gps.append([*edge_cords[i], post[-1]])
        travel_gps.append([post[1], post[2], post[-1]])
    else:
        idx1 = findSegment([pre[1], pre[2]], pre[-1])
        idx2 = findSegment([post[1], post[2]], post[-1])
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            pre, post = post, pre
        edge_cords = np.array(road_map.edgeCord[pre[-1]]).reshape(-1, 2).tolist()
        travel_gps.append([pre[1], pre[2], pre[-1]])
        for i in range(idx1 + 1, idx2 + 1, 1):
            travel_gps.append([*edge_cords[i], pre[-1]])
        travel_gps.append([post[1], post[2], post[-1]])
    travel_dist = 0
    for i in range(len(travel_gps) - 1):
        travel_dist += haversine(travel_gps[i][0], travel_gps[i][1], travel_gps[i + 1][0], travel_gps[i + 1][1])
    pass_dist = travel_dist / (post_time - pre_time) * (curtime - pre_time)
    total = 0
    for i in range(len(travel_gps) - 1):
        total += haversine(travel_gps[i][0], travel_gps[i][1], travel_gps[i + 1][0], travel_gps[i + 1][1])
        if total >= pass_dist:
            total -= haversine(travel_gps[i][0], travel_gps[i][1], travel_gps[i + 1][0], travel_gps[i + 1][1])
            curGPS = findGPS([travel_gps[i][:2], travel_gps[i + 1][:2]], pass_dist - total)
            return [curtime, *curGPS, travel_gps[i + 1][-1]]
    return [curtime, post[1], post[2], post[3]]

def processTraj(record):
    if len(record) == 0:
        return []
    timestamp = {}
    count = [0 for _ in range(epsilon)]
    for item in record:
        timestamp[int(item[0])] = [int(item[0]), float(item[1]), float(item[2]), int(item[3])]
        count[int(item[0]) % epsilon] += 1
    m = np.argmax(count)
    st = int(record[0][0])
    while st % epsilon != m:
        st += 1
    en = int(record[-1][0])
    processed_traj = []
    while st <= en:
        if st in timestamp:
            processed_traj.append(timestamp[st])
        else:
            pre = st
            while pre not in timestamp:
                pre -= 1
            post = st
            while post not in timestamp:
                post += 1
            gps = linearInterpolate(timestamp[pre], timestamp[post], st)
            if gps[0] == -1:
                processed_traj.append([st, timestamp[pre][1], timestamp[pre][2], -997])
            else:
                processed_traj.append(gps)
        st += epsilon
    return processed_traj

import os
input_root = f"path/to/input"
output_root = f"path/to/output"
if not os.path.exists(output_root):
    os.makedirs(output_root)
use_ray = True
resume = True

# use ray
if use_ray:
    import ray
    ray.init(num_cpus=8)

    @ray.remote
    def process_file(data_root, output_root, file):
        processed_file = os.path.join(data_root, file)
        output_file = os.path.join(output_root, file)
        records = getTrajs(processed_file)
        if os.path.exists(output_file) and resume:
            f = open(output_file, 'a+')
            os.system(f'tail -n 1 {output_file} > ./{file}_cmd.txt')
            try:
                cnt = int(open(f'./{file}_cmd.txt', 'r').readlines()[0][1:-1])
                records = records[cnt:]
                cnt += 1
            except:
                cnt = 1
        else:
            f = open(output_file, 'w+')
            cnt = 1
        for traj in records:
            processed_traj = processTraj(traj)
            for item in processed_traj:
                f.write(f'{item[0]} {item[1]} {item[2]} {item[3]}\n')
            f.write(f'-{cnt}\n')
            cnt += 1
        f.close()


    file_list = os.listdir(input_root)
    file_list = np.sort(file_list).tolist()
    ray.get([process_file.remote(input_root, output_root, file) for file in file_list])

    # import ray
    # ray.shutdown()

# without ray
if not use_ray:
    import tqdm

    def process_file(data_root, output_root, file):
        processed_file = os.path.join(data_root, file)
        output_file = os.path.join(output_root, file)
        records = getTrajs(processed_file)
        if os.path.exists(output_file) and resume:
            f = open(output_file, 'a+')
            os.system(f'tail -n 1 {output_file} > cmd.txt')
            cnt = int(open('./cmd.txt', 'r').readlines()[0][1:-1])
            records = records[cnt:]
            cnt += 1
        else:
            f = open(output_file, 'w+')
            cnt = 1
        for traj in tqdm.tqdm(records):
            processed_traj = processTraj(traj)
            for item in processed_traj:
                f.write(f'{item[0]} {item[1]} {item[2]} {item[3]}\n')
            f.write(f'-{cnt}\n')
            cnt += 1
        f.close()


    file_list = os.listdir(input_root)
    file_list = np.sort(file_list).tolist()
    [process_file(input_root, output_root, file) for file in file_list]