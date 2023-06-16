#!docker pull nvcr.io/nvidia/pytorch:20.12-py3
#!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from benchmark.tensorflow import TensorFlowBenchmark


class TensorFlowExample(TensorFlowBenchmark):

    def load_model(self, path):
        
        physical_devices = tf.config.list_physical_devices('GPU')
        assert len(physical_devices) > 0, 'No GPUs available'
        tf.config.set_visible_devices(physical_devices[0], 'GPU') #Only the first GPU will be considered

        saved_model_loaded = tf.saved_model.load(path, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.model = graph_func
        self.device = '/GPU:0'
        
        print('TensorFlow Version:', tf.__version__)
        print('Device:', self.device)


if __name__ == '__main__':


    os.system('python --version')
    
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="path to the model", default = 'forward/')
    parser.add_argument("--latency_batch_size", type=int, required=False, help="Batch size used for latency", default = 1)
    parser.add_argument("--throughput_batch_size", type=int, required=False, help="Batch size used for latency", default = 32)
    args = parser.parse_args()
    
    print('Model used:', args.path)
    

    benchmark = TensorFlowExample(
        path = args.path, 
        shape = [224, 224, 9]
    )
        
    benchmark.metrics(
        latency_batch_size = args.latency_batch_size,
        throughput_batch_size = args.throughput_batch_size
    )