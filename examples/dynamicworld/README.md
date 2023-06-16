
## DynamicWorld
```
export IMAGE=nvcr.io/nvidia/tensorflow:23.04-tf2-py3
```
**General Instructions**
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
