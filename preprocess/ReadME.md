# Brief Description of Data Preprocessing

## Trajectory Datasets

* Porto is a public dataset downloaded from [Kaggle Link](https://www.kaggle.com/datasets/crailtap/taxi-trajectory).
* For other public datasets, please refer to [tptk](https://github.com/sjruan/tptk) and [MTrajRec](https://github.com/huiminren/MTrajRec).

### Data Formats

* Each trajectory is stored in the following format:

```
timestamp_1, latitude_1, longitude_1, road_id_1
timestamp_2, latitude_2, longitude_2, road_id_2
...
timestamp_N, latitude_N, longitude_N, road_id_N
<count>
```

## Road Network

* The latest OSM map for China can be downloaded in [OSM_China](https://download.geofabrik.de/asia/china-latest.osm.pbf).
* Please clip the map using the tool [osm2rn](https://github.com/sjruan/osm2rn) and transfer it to `edgeOSM.txt`, `nodeOSM.txt` and `wayTypeOSM.txt`.

### Data Format of `edgeOSM.txt`: Information of Road Segments

```
edgeID_1, start_nodeID_1, end_nodeID_1, #point_1, [latitudes_1, longitudes_1] * #point_1
...
edgeID_M, start_nodeID_M, end_nodeID_M, #point_M, [latitudes_M, longitudes_M] * #point_M
```

* edgeID: id of the road segment
* start_nodeID: id of the intersection with which the road segment starts.
* end_nodeID: id of the intersection with which the road segment ends.
* #point: number of interval GPS points that make up the road segment. (#point >= 2, indicates the start GPS point and end GPS point.)

### Data Format of `nodeOSM.txt`: Information of Intersections

```
nodeID_1, latitudes_1, longitudes_1
...
nodeID_P, latitudes_P, longitudes_P
```

### Data Format of `wayTypeOSM.txt`: Information of Type for Each Road Segments

```
edgeID_1, wayTypeName_1, wayTypeNum_1
...
edgeID_M, wayTypeName_M, wayTypeNum_M
```

## Map Matching

* `hmm.py` first merges the road network information from `edgeOSM.txt`, `nodeOSM.txt` and `wayTypeOSM.txt`, together with the trajectories into a single file, and invoke `hmm.cpp` to perform map matching.
* `hmm.cpp` is a re-implementation of [paper](https://dl.acm.org/doi/abs/10.1145/2424321.2424428). If you find the code useful, please consider citing our paper.

To perform map matching, please modify the directories in `hmm.py`, then run `python hmm.py`. 

## Generating Training data

* `interpolate_trajectory.py` performs linear interpolation on the raw trajectories to obtain trajectories with fixed time intervals.
* `project_trajectory.py` projects each GPS point onto the corresponding road segment to obtain map-matched trajectories.
* `epilson_trajectory.py` performs linear interpolation on map-matched trajectories to obtain map-matched trajectories with fixed time intervals.
* For more detailed information on the data preprocessing procedure, please refer to [Issue #6](https://github.com/chenyuqi990215/RNTrajRec/issues/6).

## Citations

```
@inproceedings{song2012quick,
  title={Quick map matching using multi-core cpus},
  author={Song, Renchu and Lu, Wei and Sun, Weiwei and Huang, Yan and Chen, Chunan},
  booktitle={Proceedings of the 20th International Conference on Advances in Geographic Information Systems},
  pages={605--608},
  year={2012}
}
```
