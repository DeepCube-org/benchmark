
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

def test_pytorch():
    benchmark = TensorFlowExample(path = './tests/tensorflow_test_model/', shape = [224, 224, 9])
    benchmark.metrics(latency_batch_size = 1, throughput_batch_size = 1)