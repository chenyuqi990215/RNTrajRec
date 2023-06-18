# Brief Description of Data Preprocessing

## Trajectory Dataset

* Porto is a public dataset downloaded from [Kaggle Link](https://www.kaggle.com/datasets/crailtap/taxi-trajectory).
* For other public datasets, please refer to [tptk](https://github.com/sjruan/tptk) and [MTrajRec](https://github.com/huiminren/MTrajRec).

## Map Matching

* `hmm.py` first merges the road network information from `edgeOSM.txt`, `nodeOSM.txt` and `wayTypeOSM.txt`, together with the trajectories into a single file, and invoke `hmm.cpp` to perform map matching.
* `hmm.cpp` is a re-implementation of [paper](https://dl.acm.org/doi/abs/10.1145/2424321.2424428). If you find the code useful, please consider citing our paper.

## Generating Training data

* `noise_trajectory.py` performs linear interpolation on the raw trajectories to obtain trajectories with fix time intervals.
* `project_trajectory.py` projects each GPS point onto the corresponding road segment to obtain map-matched trajectories.

## Citation

```
@inproceedings{song2012quick,
  title={Quick map matching using multi-core cpus},
  author={Song, Renchu and Lu, Wei and Sun, Weiwei and Huang, Yan and Chen, Chunan},
  booktitle={Proceedings of the 20th International Conference on Advances in Geographic Information Systems},
  pages={605--608},
  year={2012}
}
```
