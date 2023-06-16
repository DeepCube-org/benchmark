from argparse import ArgumentParser
from torch import device
import torch_geometric.nn
from torch_geometric.profile import count_parameters

from benchmark.torchgeometric import TorchGeometricBenchmark


LAYER_TYPES = {
    'GCN'      : torch_geometric.nn.GCN,
    'GraphSAGE': torch_geometric.nn.GraphSAGE,
    'GIN'      : torch_geometric.nn.GIN,
    'GAT'      : torch_geometric.nn.GAT,
    'GATv2'    : lambda **kwargs: torch_geometric.nn.GAT(**kwargs, v2=True)
}

class GraphBenchmark(TorchGeometricBenchmark):

    def __init__(self, shape, layer, n_layers, compile = False):
        self.layer = layer
        self.n_layers = n_layers
        self.compile = compile
        super().__init__(path = None, shape = shape)

    def load_model(self, path):
        self.device = device("cuda")
        self.model = self.layer(
            in_channels     = self.shape[1], 
            hidden_channels = self.shape[1], 
            num_layers      = self.n_layers, 
            out_channels    = self.shape[1],
        ).eval()
        self.model = self.model.to(self.device)

        if self.compile:
            
            try:
                import torch._dynamo
                import torch._dynamo.config
            except:
                print('Compilation unsupported, check your version of torch')
                return
            
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.reset()

            self.model = torch_geometric.compile(self.model)

        print('Number of parameters:', count_parameters(self.model))



def main():

    parser = ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=False, help='number of nodes', default=10000)
    parser.add_argument('--num_edges', type=int, required=False, help='number of edges', default=200000)
    parser.add_argument('--channels', type=int, required=False, help='number of channels', default=96)
    parser.add_argument('--model_name', type=str, required=True, help='Layer type', choices=list(LAYER_TYPES.keys()))
    parser.add_argument('--n_layers',   type=int, required=True, help='Number of Layers')
    
    parser.add_argument('--latency_batch_size', type=int, required=False, help='latency batch size', default = None)
    parser.add_argument('--throughput_batch_size', type=int, required=False, help='throughput batch size', default = None)

    parser.add_argument('--compile', type=int, required=False, help='compile the model', default=0)



    parser.add_argument('--repetitions', type=int, required=False, help='Number of repetitions', default=2)
    parser.add_argument('--warm_up_repetitions', type=int, required=False, help='Number of warm up repetitions', default=50)
    parser.add_argument('--latency_repetitions', type=int, required=False, help='Number of latency repetitions', default=300)
    parser.add_argument('--throughput_repetitions', type=int, required=False, help='Number of throughput repetitions', default=100)


    args = parser.parse_args()

    args.compile = bool(args.compile)

    if args.latency_batch_size == 0:
        args.latency_batch_size = None
    if args.throughput_batch_size == 0:
        args.throughput_batch_size = None   
    
    return(args)

if __name__ == '__main__':


    args = main()
    
    benchmark = GraphBenchmark(
        shape     = [args.num_nodes, args.channels, args.num_edges],
        layer     = LAYER_TYPES[args.model_name],
        n_layers  = args.n_layers,
        compile   = args.compile
    )
    
    benchmark.metrics(
        latency_batch_size = args.latency_batch_size, 
        throughput_batch_size = args.throughput_batch_size,

        warm_up_repetitions = args.warm_up_repetitions,
        latency_repetitions = args.latency_repetitions,
        throughput_repetitions = args.throughput_repetitions,
        repetitions=args.repetitions
    )
