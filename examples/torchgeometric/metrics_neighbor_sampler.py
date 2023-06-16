from torch_geometric.loader import NeighborLoader

from metrics import GraphBenchmark, LAYER_TYPES, main

class NLGraphBenchmark(GraphBenchmark):

    def time_model(self, model, dummy_input):

        loader = NeighborLoader(dummy_input, batch_size = 32, num_neighbors=[-1]*self.n_layers, input_nodes = None)
        curr_time = 0
        for batch in loader:
          curr_time = curr_time + super().time_model(model, dummy_input=batch)
        
        return curr_time


if __name__ == '__main__':

    args = main()

    benchmark = NLGraphBenchmark(
        shape = [args.num_nodes, args.channels, args.num_edges],
        layer     = LAYER_TYPES[args.model_name],
        n_layers  = args.n_layers,
        compile   = args.compile
    )
    benchmark.metrics(
        latency_batch_size = args.latency_batch_size, 
        throughput_batch_size = args.throughput_batch_size
    )
