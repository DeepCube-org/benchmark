
import time
import tensorflow as tf

try:
    from tensorflow.test.experimental import sync_devices
    print('TensorFlow version supporting sync_devices')
except ModuleNotFoundError:
    from .experimental import sync_devices
    print('TensorFlow version not supporting sync_devices')

from .base import Benchmark

class TensorFlowBenchmark(Benchmark):
    """
    Abstract Class,
    shape: [resolution, resolution, channels]
    """
    
    def get_dummy(self, shape):
        x = tf.random.uniform(shape)
        return(x)

    
    def time_model(self, model, dummy_input):
        with tf.device(self.device):
            sync_devices()
            starter = time.time()*1000.0
            _ = model(dummy_input)
            sync_devices()
            ender = time.time()*1000.0
            curr_time = ender-starter
        return(curr_time)