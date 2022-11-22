## Code for `RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer`

### Data format

#### OSM map format

Map from OSM that contains: `edgeOSM.txt nodeOSM.txt wayTypeOSM.txt`. Other map format is preferred and `map.py` need to be modified.

#### Train data format

Train data has the following format: 

```
--- ./data
  |____ train
    |____ train_input.txt
    |____ train_output.txt
  |____ valid
    |____ valid_input.txt
    |____ valid_output.txt
  |____ test
    |____ test_input.txt
    |____ test_output.txt
```

Note that:
* `{train_valid_test}_input.txt` contains raw GPS trajectory, `{train_valid_test}_output.txt` contains map-matched trajectory.
* The sample rate of input and output file for train and valid dataset in both raw GPS trajectory and map-matched trajectory need to be the same, as the downsampling process in done while obtaining training item.
* The sample rate of test input and output file is different, i.e. `test_input.txt` contain low-sample raw GPS trajectories and `test_output.txt` contain high-sample map-matched trajectories.

#### Training and Testing

```
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --dis_prob_mask_flag --pro_features_flag \
      --tandem_fea_flag --decay_flag   > chengdu_8.txt &
```

#### File information

* `gps_transformer_layer.py`: implement of Graph Refinement Layer (GRL).
* `graph_norm.py`: implement of Graph Normalization.
* `graph_func.py`: implement of graph functions.
* `map.py`: implement of map functions, i.e. calculating shortest path and r-tree indexing.
* `model.py`: implement of RNTrajRec.
