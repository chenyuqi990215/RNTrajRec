city = "Porto"
map_root = f"path/to/road_network/"
import sys
sys.path.append('../')
sys.path.append('../../')

from map import RoadNetworkMap

if city == "Shanghai":
    road_map = RoadNetworkMap(map_root,zone_range=[31.17491, 121.439492, 31.305073, 121.507001], unit_length=50)
elif city == "Chengdu":
    road_map = RoadNetworkMap(map_root,zone_range=[30.655347, 104.039711, 30.730157, 104.127151], unit_length=50)
elif city == "Porto":
    road_map = RoadNetworkMap(map_root,zone_range=[41.111975, -8.667057, 41.177462, -8.585305], unit_length=50)

if city == "Shanghai":
    zone_range = [31.17491, 121.439492, 31.305073, 121.507001]
    epsilon = 10
elif city == "Chengdu":
    zone_range = [30.655347, 104.039711, 30.730157, 104.127151]
    epsilon = 12
elif city == "Porto":
    zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
    epsilon = 15

edgeOSM = f"path/to/road_network/edgeOSM.txt"
nodeOSM = f"path/to/road_network/nodeOSM.txt"
wayOSM = f"path/to/road_network/wayTypeOSM.txt"


output_root = f"path/to/output/file"
import os
if not os.path.exists(output_root):
    os.makedirs(output_root)

sample_rate = [8]
edge = open(edgeOSM, 'r').readlines()
node = open(nodeOSM, 'r').readlines()
way = open(wayOSM, 'r').readlines()

from math import *
import math
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

use_ray = False
k = len(node)
n = road_map.valid_edge_cnt

if use_ray:
    import ray
    ray.init(num_cpus=2)

    @ray.remote
    def process_file(output_root, sample_rate):
        output_path = os.path.join(output_root, f'test_input_{sample_rate}.txt')
        output = open(output_path, 'w+')
        trajOSM = f"path/to/raw/trajectory/test_output_{sample_rate}.txt"
        record = getTrajs(trajOSM)

        m = len(record)

        output.write(f'{zone_range[0]} {zone_range[1]} {zone_range[2]} {zone_range[3]}\n')
        output.write(f'{n} {k}\n')

        for rid in road_map.valid_edge.keys():
            edges = edge[rid].strip().split()
            ways = way[rid].strip().split()
            assert edges[0] == ways[0]
            output.write(f'{road_map.valid_edge[rid]} {edges[1]} {edges[2]} ')
            output.write(f'{ways[1]} {ways[2]}')
            for item in edges[3:]:
                output.write(f' {item}')
            output.write('\n')

        output.write(f'{m}\n')
        cnt = 1
        for traj in record:
            for item in traj:
                output.write(f'{item[0]} {item[1]} {item[2]}\n')
            output.write(f'{cnt}\n')
            cnt += 1
        output.close()

    ray.get([process_file.remote(output_root, rate) for rate in sample_rate])

    @ray.remote
    def run_hmm(output_root, sample_rate):
        input_path = os.path.join(output_root, f'test_input_{sample_rate}.txt')
        output_path = os.path.join(output_root, f'test_output_{sample_rate}.txt')
        os.system(f'./hmm < {input_path} > {output_path}')
    ray.get([run_hmm.remote(output_root, rate) for rate in sample_rate])

if not use_ray:
    def process_file(output_root, sample_rate):
        output_path = os.path.join(output_root, f'test_input_{sample_rate}.txt')
        output = open(output_path, 'w+')
        trajOSM = f"path/to/raw/trajectory/test_output_{sample_rate}.txt"
        record = getTrajs(trajOSM)

        n = road_map.valid_edge_cnt
        m = len(record)
        k = len(node)

        output.write(f'{zone_range[0]} {zone_range[1]} {zone_range[2]} {zone_range[3]}\n')
        output.write(f'{n} {k}\n')

        for rid in road_map.valid_edge.keys():
            edges = edge[rid].strip().split()
            ways = way[rid].strip().split()
            assert edges[0] == ways[0]
            output.write(f'{road_map.valid_edge[rid]} {edges[1]} {edges[2]} ')
            output.write(f'{ways[1]} {ways[2]}')
            for item in edges[3:]:
                output.write(f' {item}')
            output.write('\n')

        output.write(f'{m}\n')
        cnt = 1
        for traj in record:
            for item in traj:
                output.write(f'{item[0]} {item[1]} {item[2]}\n')
            output.write(f'{cnt}\n')
            cnt += 1
        output.close()

    [process_file(output_root, rate) for rate in sample_rate]

    def run_hmm(output_root, sample_rate):
        input_path = os.path.join(output_root, f'test_input_{sample_rate}.txt')
        output_path = os.path.join(output_root, f'test_output_{sample_rate}.txt')
        os.system(f'./hmm < {input_path} > {output_path}')
    [run_hmm(output_root, rate) for rate in sample_rate]