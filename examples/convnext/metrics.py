from argparse import ArgumentParser
from torch import device
from benchmark.pytorch import PyTorchBenchmark
import convnext

class TorchVisionConvNext(PyTorchBenchmark):

    def __init__(
        self,
        shape,
        version = 'large',
        compile = False
    ):
        self.version = version
        self.compile = compile
        super().__init__(path = None, shape = shape)


    def load_model(self, path):

        self.device = device("cuda")

        model_map = {
            'tiny':  convnext.convnext_tiny,
            'small': convnext.convnext_small,
            'base':  convnext.convnext_base,
            'large': convnext.convnext_large,
            'xxs':   convnext.convnext_ultratiny # Used in the deepcube project
        }
        self.model = model_map[self.version](num_classes = None, in_chans = self.shape[-1]).eval()
        self.model = self.model.to(self.device)

        if self.compile:

            try:
                import torch._dynamo
            except:
                print('Compilation unsupported, check your version of torch')
                return

            torch._dynamo.reset()
            self.model = torch.compile(self.model, mode = 'reduce-overhead')


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='type of convnext')
    parser.add_argument('--compile', type=int, required=False, help='compile the model', default=0)
    parser.add_argument('--channels', type=int, required=False, help='number of channels', default=3)
    parser.add_argument('--resolution', type=int, required=False, help='resolution', default=256)
    args = parser.parse_args()

    args.compile = bool(args.compile)


    benchmark = TorchVisionConvNext(
        shape = [args.resolution, args.resolution, args.channels],
        version = args.version, 
        compile = args.compile
    )

    benchmark.metrics(latency_batch_size = 1, throughput_batch_size = 32)