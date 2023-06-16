
## General Instructions
```
docker pull $IMAGE
docker run --rm --shm-size=1g --ulimit memlock=-1 --gpus all -it $IMAGE /bin/bash
git clone https://github.com/DeepCube-org/benchmark.git
cd benchmark
pip install -e .
```
## TorchGeometric
```
export IMAGE=nvcr.io/nvidia/pytorch:23.03-py3
```
General Instructions
```
cd examples/torchgeometric
pip install torch_geometric
```
### Inference performance: NVIDIA A10G (aws g5.xlarge)

#### Full Batch

num_nodes: 10000, num_edges: 200000, channels: 96

```
./fullbatch.sh
./fullbatch_compile.sh
```
<!-- All the results have been correcly computed using a repetition equal to 100 --> 
| **Model Name**   | **Number of Parameters** | **Batch Size**    | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |
| :--------------: |:---------------:         |:--------------:   |:--------------:   |:---------------:     |:---------------:              | 
|       GIN        |   18624                  |      1            |       1           |    0.6663  ms        |    0.2906  ms                 | <!-- (std: 0.0067), (std: 0.0085)--> 
|       GraphSAGE  |   18528                  |      1            |       1           |    0.6736  ms        |    0.3264  ms                 | <!-- (std: 0.0066), (std: 0.0100)--> 
|       GCN        |   9312                   |      1            |       1           |    1.3214  ms        |    0.6722  ms                 | <!-- (std: 0.0207), (std: 0.0489)--> 
|       GAT        |   9504                   |      1            |       1           |    1.5185 ms         |    0.6699  ms                 | <!-- (std: 0.0243), (std: 0.0374)--> 
|       GATv2      |   18816                  |      1            |       1           |    2.9112  ms        |    0.8025  ms                 | <!-- (std: 0.0165), (std: 0.0586)--> 
| **Model Name**   | **Number of Parameters** | **Batch Size**    | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |
|       GIN        |   37248                  |      1            |       2           |    1.2875  ms        |    0.5038  ms                 | <!-- (std: 0.0059), (std: 0.0112)--> 
|       GraphSAGE  |   37056                  |      1            |       2           |    1.3108  ms        |    0.5869  ms                 | <!-- (std: 0.0116), (std: 0.0085)--> 
|       GCN        |   18624                  |      1            |       2           |    2.5526  ms        |    1.2051  ms                 | <!-- (std: 0.0262), (std: 0.0628)--> 
|       GAT        |   19008                  |      1            |       2           |    2.9507  ms        |    1.1969  ms                 | <!-- (std: 0.0317), (std: 0.0687)--> 
|       GATv2      |   37632                  |      1            |       2           |    5.7337  ms        |    1.4747  ms                 | <!-- (std: 0.0227), (std: 0.1046)--> 
| **Model Name**   | **Number of Parameters** | **Batch Size**    | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |
|       GIN        |   55872                  |      1            |       3           |    1.9078  ms        |    0.7189  ms                 | <!-- (std: 0.0050), (std: 0.0108)--> 
|       GraphSAGE  |   55584                  |      1            |       3           |    1.9319  ms        |    0.8535  ms                 | <!-- (std: 0.0047), (std: 0.0166)--> 
|       GCN        |   27936                  |      1            |       3           |    3.7931  ms        |    1.7481  ms                 | <!-- (std: 0.0326), (std: 0.0898)--> 
|       GAT        |   28512                  |      1            |       3           |    4.3658  ms        |    1.7416  ms                 | <!-- (std: 0.0411), (std: 0.1066)--> 
|       GATv2      |   56448                  |      1            |       3           |    8.5612  ms        |    2.1651  ms                 | <!-- (std: 0.0239), (std: 0.1654)--> 

<!-- All the results have been correcly computed using a repetition equal to 100 --> 
| **Model Name**   | **Number of Parameters** | **Batch Size**    | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |  
| :--------------: |:---------------:         | :--------------:  |:--------------:   |:---------------:     |:---------------:              |  
|       GIN        |   18624                  |       32          |       1           |    19.8790   ms      |    7.6549  ms                 |   <!-- (std: 0.0147) -->   <!-- (std: 0.0163) --> 
|       GraphSAGE  |   18528                  |       32          |       1           |    19.6060   ms      |    7.7938  ms                 |   <!-- (std: 0.0215) -->   <!-- (std: 0.0121) --> 
|       GCN        |   9312                   |       32          |       1           |    32.6232   ms      |    9.2600  ms                 |   <!-- (std: 0.0749) -->   <!-- (std: 0.0307) --> 
|       GAT        |   9504                   |       32          |       1           |    36.0897   ms      |    Out of Memory              |   <!-- (std: 0.0733) -->   <!-- (std: X) --> 
|       GATv2      |   18816                  |       32          |       1           |    85.5409   ms      |    14.7598  ms                |   <!-- (std: 0.1623) -->   <!-- (std: 0.1294) --> 
| **Model Name**   | **Number of Parameters** |   **Batch Size**  | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |                            
|       GIN        |   37248                  |       32          |       2           |    40.2147   ms      |    15.2431  ms                |   <!-- (std: 0.0212) -->   <!-- (std: 0.0188) --> 
|       GraphSAGE  |   37056                  |       32          |       2           |    39.6611   ms      |    16.0212  ms                |   <!-- (std: 0.0287) -->   <!-- (std: 0.0210) --> 
|       GCN        |   18624                  |       32          |       2           |    65.7081   ms      |    18.9430  ms                |   <!-- (std: 0.1466) -->   <!-- (std: 0.0475) --> 
|       GAT        |   19008                  |       32          |       2           |    72.1441   ms      |    Out of Memory              |   <!-- (std: 0.1453) -->   <!-- (std: X) --> 
|       GATv2      |   37632                  |       32          |       2           |    171.0824  ms      |    29.9095  ms                |   <!-- (std: 0.3164) -->   <!-- (std: 0.3169) -->
| **Model Name**   | **Number of Parameters** |   **Batch Size**  | **Num of Layers** | **Latency Avg**      | **Latency Avg (compiled)**    |                            
|       GIN        |   55872                  |       32          |       3           |    60.5363   ms      |    22.8345  ms                |   <!-- (std: 0.0278) -->   <!-- (std: 0.0263) --> 
|       GraphSAGE  |   55584                  |       32          |       3           |    59.7112   ms      |    24.2440  ms                |   <!-- (std: 0.0319) -->   <!-- (std: 0.0245) --> 
|       GCN        |   27936                  |       32          |       3           |    98.8118   ms      |    30.2752  ms                |   <!-- (std: 0.2040) -->   <!-- (std: 28.5666) -->
|       GAT        |   28512                  |       32          |       3           |    108.1666  ms      |    Out of Memory              |   <!-- (std: 0.2078) -->   <!-- (std: X) -->  
|       GATv2      |   56448                  |       32          |       3           |    256.4948  ms      |    45.2049  ms                |   <!-- (std: 0.4617) -->   <!-- (std: 0.6197) -->

<!-- All the results have been correcly computed using a repetition equal to 100 --> 
| **Model Name**   | **Number of Parameters** |   **Batch Size**  | **Num of Layers** | **Throughput Avg**   | **Throughput Avg (compiled)** |
| :--------------: |:---------------:         | :--------------:  |:--------------:   |:---------------:     |:---------------:              |        
|       GIN        |   18624                  |       32          |       1           |    1609.7412 graph/s |    4182.7457 graph/s          |  
|       GraphSAGE  |   18528                  |       32          |       1           |    1632.7169 graph/s |    4104.6874 graph/s          |  
|       GCN        |   9312                   |       32          |       1           |    981.0884 graph/s  |    3454.1460 graph/s          |  
|       GAT        |   9504                   |       32          |       1           |    886.5153 graph/s  |    Out of Memory              | 
|       GATv2      |   18816                  |       32          |       1           |    373.9294 graph/s  |    2172.7055 graph/s          | 
| **Model Name**   | **Number of Parameters** |   **Batch Size**  | **Num of Layers** | **Throughput Avg**   | **Throughput Avg (compiled)** |
|       GIN        |   37248                  |       32          |       2           |    795.5087 graph/s  |    2099.0966 graph/s          |  
|       GraphSAGE  |   37056                  |       32          |       2           |    806.7792 graph/s  |    1996.9840 graph/s          |  
|       GCN        |   18624                  |       32          |       2           |    486.7943 graph/s  |    1238.7911 graph/s          |  
|       GAT        |   19008                  |       32          |       2           |    443.3660 graph/s  |    Out of Memory              | 
|       GATv2      |   37632                  |       32          |       2           |    187.1179 graph/s  |    1069.7311 graph/s          | 
| **Model Name**   | **Number of Parameters** |   **Batch Size**  | **Num of Layers** | **Throughput Avg**   | **Throughput Avg (compiled)** |
|       GIN        |   55872                  |       32          |       3           |    528.6044 graph/s  |    1401.0162 graph/s          |   
|       GraphSAGE  |   55584                  |       32          |       3           |    535.8115 graph/s  |    1320.0867 graph/s          |  
|       GCN        |   27936                  |       32          |       3           |    323.8777 graph/s  |    1117.8085 graph/s          |   
|       GAT        |   28512                  |       32          |       3           |    295.8685 graph/s  |    Out of Memory              | 
|       GATv2      |   56448                  |       32          |       3           |    124.7942 graph/s  |    709.6241 graph/s           | 


#### NeighborSampling

num_nodes: 10000, num_edges: 200000, channels: 96, full neighborhood (num_neighbors=[-1]*n_layers), neighbor sampling batch size: 32

In order to run this examples optional PyTorch Geometric libraries are required. 
At the time of these benchmarks, the wheels of these libraries were not available for CUDA 12.1, so they were compiled from source:
```
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install --verbose torch_spline_conv
```

```
./neighborsampling.sh
```


<!-- All the results have been correcly computed using a repetition equal to 100 --> 
| **Model Name**   | **Number of Parameters** |  **Batch Size**  | **Num of Layers** | **Latency Avg** |
| :--------------: |:---------------:         |:--------------:  |:--------------:|:---------------:   |
|       GIN        |   18624                  |      1           |       1        |    97.0898  ms     | <!-- (std: 0.8772) --> 
|       GraphSAGE  |   18528                  |      1           |       1        |    96.4141  ms     | <!-- (std: 0.5672) --> 
|       GCN        |   9312                   |      1           |       1        |    171.3949  ms    | <!-- (std: 1.9348) --> 
|       GAT        |   9504                   |      1           |       1        |    233.1329  ms    | <!-- (std: 2.3914) --> 
|       GATv2      |   18816                  |      1           |       1        |    224.6168  ms    | <!-- (std: 1.3029) --> 
| **Model Name**   | **Number of Parameters** |  **Batch Size**  | **Num of Layers** | **Latency Avg** |
|       GIN        |   37248                  |      1           |       2        |    187.3966  ms    | <!-- (std: 1.7678) --> 
|       GraphSAGE  |   37056                  |      1           |       2        |    187.6341  ms    | <!-- (std: 1.8938) --> 
|       GCN        |   18624                  |      1           |       2        |    341.0543  ms    | <!-- (std: 2.9032) --> 
|       GAT        |   19008                  |      1           |       2        |    460.0741  ms    | <!-- (std: 3.7628) --> 
|       GATv2      |   37632                  |      1           |       2        |    451.7971  ms    | <!-- (std: 4.8045) --> 
| **Model Name**   | **Number of Parameters** |  **Batch Size**  | **Num of Layers** | **Latency Avg** |
|       GIN        |   55872                  |      1           |       3        |    427.3370  ms    | <!-- (std: 4.8659) --> 
|       GraphSAGE  |   55584                  |      1           |       3        |    433.6125  ms    | <!-- (std: 8.6000) --> 
|       GCN        |   27936                  |      1           |       3        |    984.7043  ms    | <!-- (std: 4.2324) --> 
|       GAT        |   28512                  |      1           |       3        |    1185.8528  ms   | <!-- (std: 7.9140) --> 
|       GATv2      |   56448                  |      1           |       3        |    2056.7208  ms   | <!-- (std: 4.6069) --> 


## DyResGNN (DeepCube)
```
export IMAGE=nvcr.io/nvidia/pytorch:23.03-py3
```
General Instructions
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

## ConvNext
```
export IMAGE=nvcr.io/nvidia/pytorch:23.03-py3
```
General Instructions
```
cd examples/convnext
```
### Inference performance: NVIDIA A10G (aws g5.xlarge)

#### ConvNext, resolution: 224x224, channels: 3
```
python metrics.py --version small --resolution 224 --channels 3 --compile 0
python metrics.py --version tiny --resolution 224 --channels 3 --compile 0
python metrics.py --version xxs --resolution 224 --channels 3 --compile 0
```

| **Version** | **Batch Size** | **Latency Avg** |
|:--------------:|:--------------:|:---------------:|
|       small    |       1        |     9.4740 ms     | <!-- (std: 0.0867 ) -->
|       tiny     |       1        |     5.2547 ms    | <!-- (std: 0.0638 ) -->
|       xxs      |       1        |     3.3172 ms     | <!-- (std: 0.0490 ) -->

| **Version** | **Batch Size** | **Throughput Avg** |
|:--------------:|:--------------:|:------------------:|
|       small    |       32       |      515.8591 img/s      |
|       tiny     |       32       |      843.0127 img/s      |
|       xxs      |       32       |      9305.7458 img/s      |


#### ConvNext XXS, resolution: 80x80, channels: 11
```
python metrics.py --version xxs --resolution 80 --channels 11 --compile 0
```

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    3.4235  ms     | <!-- (std: 0.3164) -->

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      9601.0154 img/s      |


## DynamicWorld
```
export IMAGE=nvcr.io/nvidia/tensorflow:23.04-tf2-py3
```
General Instructions
```
cd examples/dynamicworld
```
### Inference performance: NVIDIA A10G (aws g5.xlarge)

#### FP32, resolution: 224x224, channels: 9
```
python tf2rt.py --precision 32 --path forward_trt/
python tf2rt.py --precision 16 --path forward_trt_16/

python metrics.py --path forward/
python metrics.py --path forward_trt/
python metrics.py --path forward_trt_16/
```

| **Version** | **Batch Size** | **Latency Avg** |
|:--------------:|:--------------:|:---------------:|
|     FP32       |       1        |    18.8594  ms     | <!-- (std: 0.3340) -->
|     TensorRT+FP32       |       1        |    15.2464 ms     | <!-- (std: 0.3939) -->
|     TensorRT+FP16       |       1        |    14.4222 ms     | <!-- (std: 0.2938) -->

| **Version** | **Batch Size** | **Throughput Avg** |
|:--------------:|:--------------:|:------------------:|
|     FP32       |       32        |      58.3326 img/s      |
|     TensorRT+FP32       |       32        |      66.8489 img/s      |
|      TensorRT+FP16       |       32        |      70.3552 img/s      |

## Notes

If SageMaker notebooks are used during the benchmarks, it could be necessary to change the directory where docker images are saved (to use the EBS storage).
```
sudo systemctl stop docker
sudo mv /var/lib/docker/ /home/ec2-user/SageMaker/docker/
sudo ln -s /home/ec2-user/SageMaker/docker/ /var/lib/docker
sudo systemctl start docker
```
