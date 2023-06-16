from benchmark.pytorch import PyTorchBenchmark
import torch
import torchvision.models

class TorchVisionExample(PyTorchBenchmark):

    def load_model(self, path):
        self.device = torch.device("cuda")
        self.model = torchvision.models.convnext_base(weights=None).eval()
        self.model = self.model.to(self.device)
        
def test_pytorch():
    benchmark = TorchVisionExample(path = None, shape=[224, 224, 3])
    benchmark.metrics(latency_batch_size = 1, throughput_batch_size = 1)