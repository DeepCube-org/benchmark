
## General Instructions
```
docker pull $IMAGE
docker run --rm --shm-size=1g --ulimit memlock=-1 --gpus all -it $IMAGE /bin/bash
git clone https://github.com/DeepCube-org/benchmark.git
cd benchmark
pip install -e .
```

For each example there is a README.md, in order to run an example, the starting directory has to be the one where this file is located.

## Notes

If SageMaker notebooks are used during the benchmarks, it could be necessary to change the directory where docker images are saved (to use the EBS storage).
```
sudo systemctl stop docker
sudo mv /var/lib/docker/ /home/ec2-user/SageMaker/docker/
sudo ln -s /home/ec2-user/SageMaker/docker/ /var/lib/docker
sudo systemctl start docker
```
