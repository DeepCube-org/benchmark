from argparse import ArgumentParser
from torch import device
from torch_geometric.nn import GraphSAGE

from benchmark.torchgeometric import TorchGeometricBenchmark
from graph import GraphModel

class GraphDeepCubeBenchmark(TorchGeometricBenchmark):

    def __init__(self, shape, layer, n_layers):
        self.layer = layer
        self.n_layers = n_layers
        super().__init__(path = None, shape = shape)

    def load_model(self, path):
        self.device = device("cuda")
        self.model = GraphModel(
            layer      = self.layer,
            n_layers   = self.n_layers,
            hidden_dim = self.shape[1], 
            dropout    = 0.5,
            heads      = 1
        ).eval()
        self.model = self.model.to(self.device)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=False, help='number of nodes', default=10000)
    parser.add_argument('--num_edges', type=int, required=False, help='number of edges', default=200000)
    parser.add_argument('--channels', type=int, required=False, help='number of channels', default=96)
    parser.add_argument('--layer_name', type=str, required=True, help='Layer type', choices=list(GraphModel.LAYER_TYPES.keys()))
    parser.add_argument('--n_layers',   type=int, required=True, help='Number of Layers')
    args = parser.parse_args()


    benchmark = GraphDeepCubeBenchmark(
        shape = [args.num_nodes, args.channels, args.num_edges],
        layer     = GraphModel.LAYER_TYPES[args.layer_name],
        n_layers  = args.n_layers
    )
    benchmark.metrics(
        latency_batch_size = 1, 
        throughput_batch_size = 32
    )