from abc import ABC, abstractmethod
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from .base import Benchmark

class TorchGeometricBenchmark(Benchmark):
    """
    Abstract Class, for Graph problems
    shape: [num_nodes, channels, num_edges]
    """

    def get_dummy(self, shape): #(batch_size, num_nodes, feature_size, num_edges)
        batch_list = []
        for i in range(shape[0]):
            x = torch.randn(shape[1], shape[2], dtype=torch.float, device=self.device)
            edge_index = torch.randint(shape[1], (2, shape[3]), device=self.device)
            batch_list.append(Data(x = x, edge_index = edge_index).to(self.device))
        return Batch.from_data_list(batch_list)

    @torch.no_grad()
    def time_model(self, model, dummy_input):
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input.x, dummy_input.edge_index)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        return curr_time
    
    #def get_optimal_batch_size(self, model):
    #    self.warm_up(model)
    #
    #    optimal_batch_size = 1
    #    for batch_size in [32, 64, 128, 256, 512, 1024]:
    #        dummy_input = self.get_dummy((batch_size, self.num_nodes, self.channels, self.num_edges))
    #        try:
    #            _ = model(dummy_input)
    #            optimal_batch_size = batch_size
    #        except RuntimeError as e:
    #            print(e)
    #            break
    #    return(optimal_batch_size)