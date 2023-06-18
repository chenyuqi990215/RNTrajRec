city = "Porto"
data_root = "path/to/trajectory_data"
map_root = "path/to/road_network"

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

def projectToRoadmap(x, y, rid):
    if rid not in road_map.valid_edge:
        return [-1, -1]
    edge_cords = np.array(road_map.edgeCord[rid]).reshape(-1, 2)
    dist = []
    gps = []
    for i in range(edge_cords.shape[0] - 1):
        gps.append(projectToSegment([x, y], [edge_cords[i], edge_cords[i + 1]]))
        dist.append(haversine(x, y, *gps[-1]))
    idx = np.argmin(dist)
    return gps[idx]

import os
output_root = f"/path/to/output/directory"
if not os.path.exists(output_root):
    os.makedirs(output_root)
use_ray = True

# use ray
if use_ray:
    import ray
    ray.init(num_cpus=10)

    @ray.remote
    def process_file(data_root, output_root, file):
        processed_file = os.path.join(data_root, file)
        output_file = os.path.join(output_root, file)
        f = open(output_file, 'w+')
        r = open(processed_file, 'r').readlines()
        for line in r:
            item = line.strip()
            if item[0] == '-':   # end of trajectory
                f.write(f'{item}\n')
            elif item.split(' ')[-1] == '-999':   # unmatched trajectory
                f.write(f'{item}\n')
            else:
                item = item.split(' ')
                assert len(item) == 4
                projected_point = projectToRoadmap(float(item[1]), float(item[2]), int(item[3]))
                if projected_point[0] == -1:   # not in zone range
                    f.write(f'{item[0]} {item[1]} {item[2]} -998\n')
                else:
                    f.write(f'{item[0]} {projected_point[0]} {projected_point[1]} {item[3]}\n')
        f.close()


    file_list = os.listdir(data_root)
    file_list = np.sort(file_list).tolist()[:10]
    ray.get([process_file.remote(data_root, output_root, file) for file in file_list])

# without ray
if not use_ray:
    import tqdm

    def process_file(data_root, output_root, file):
        processed_file = os.path.join(data_root, file)
        output_file = os.path.join(output_root, file)
        f = open(output_file, 'w+')
        r = open(processed_file, 'r').readlines()
        for line in tqdm.tqdm(r):
            item = line.strip()
            if item[0] == '-':   # end of trajectory
                f.write(f'{item}\n')
            elif item.split(' ')[-1] == '-999':   # unmatched trajectory
                f.write(f'{item}\n')
            else:
                item = item.split(' ')
                assert len(item) == 4
                projected_point = projectToRoadmap(float(item[1]), float(item[2]), int(item[3]))
                if projected_point[0] == -1:   # not in zone range
                    f.write(f'{item[0]} {item[1]} {item[2]} -998\n')
                else:
                    f.write(f'{item[0]} {projected_point[0]} {projected_point[1]} {item[3]}\n')
        f.close()


    file_list = os.listdir(data_root)
    file_list = np.sort(file_list).tolist()[:10]
    [process_file(data_root, output_root, file) for file in file_list]
