import torch
from .base import Benchmark


class PyTorchBenchmark(Benchmark):
    """
    Abstract Class, for Computer Vision problems
    shape: [resolution, resolution, channels]
    """
    
    def get_dummy(self, shape):
        x = torch.randn(shape[0], shape[3], shape[2], shape[1], dtype=torch.float, device=self.device)
        return(x)

    @torch.no_grad()
    def time_model(self, model, dummy_input):
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        return(curr_time)
