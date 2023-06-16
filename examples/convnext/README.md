## ConvNext
```
export IMAGE=nvcr.io/nvidia/pytorch:23.03-py3
```
**Follow the General Instructions contained in the "main" README.md**
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

