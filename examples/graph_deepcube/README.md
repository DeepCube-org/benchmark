## DyResGNN (DeepCube)
```
export IMAGE=nvcr.io/nvidia/pytorch:23.03-py3
```
**General Instructions**
```
cd examples/graph_deepcube
pip install torch_geometric
```
### Inference performance: NVIDIA A10G (aws g5.xlarge)

num_nodes: 4000, num_edges: 80000, channels: 96

#### layer_name: DyResLayer, n_layers: 1/2/3
```
python metrics.py --layer_name DyResLayer --n_layers 1 --num_nodes 4000 --num_edges 80000
python metrics.py --layer_name DyResLayer --n_layers 2 --num_nodes 4000 --num_edges 80000
python metrics.py --layer_name DyResLayer --n_layers 3 --num_nodes 4000 --num_edges 80000
```

| **Num of Layers** | **Batch Size** | **Latency Avg** |
| :--------------: |:--------------:|:---------------: |
|       1          |       1        |    1.8571  ms    | <!-- (std: 0.0089) -->
|       2          |       1        |    3.6073  ms    | <!-- (std: 0.0090) -->
|       3          |       1        |    5.3608  ms     | <!-- (std: 0.0092) -->

| **Num of Layers** | **Batch Size** | **Throughput Avg** |
| :--------------: |:--------------:|:---------------: |
|       1          |       32       |    577.1216 graph/s | 
|       2          |       32       |    288.7843 graph/s | 
|       3          |       32       |    192.6183 graph/s |