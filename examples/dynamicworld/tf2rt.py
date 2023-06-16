"""
tf2rt.py
====================================
Utility function to convert the original Dynamic World model to an optimized TensorRT version of it.
The TensorRT execution engine should be built on a GPU of the same device type as the one on which inference will be executed 
as the building process is GPU specific.
"""

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
 
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path to the model")
    parser.add_argument("--precision", type=int, required=True, help="precision to be used (16 or 32)")
    args = parser.parse_args()
    
    saved_model_dir = 'forward/'
    output_saved_model_dir = args.path
    
    if args.precision == 32:
        precision_mode = trt.TrtPrecisionMode.FP32
    elif args.precision == 16:
        precision_mode = trt.TrtPrecisionMode.FP16
    else:
        raise Exception('Unsupported precision')
    
    # Instantiate the TF-TRT converter
    converter = trt.TrtGraphConverterV2(
       input_saved_model_dir=saved_model_dir,
       precision_mode=precision_mode,
       use_calibration=False,
       use_dynamic_shape=True, # Enable dynamic shape for all the dimensions
       dynamic_shape_profile_strategy='Optimal', # Limited by the inputs provided during the build but the best performing one
       maximum_cached_engines = 16, 
       allow_build_at_runtime = True
    )

    # Convert the model into TRT compatible segments
    trt_func = converter.convert()
    converter.summary()


    def input_fn():
        # max batch size expected to be used in inference, 
        #If we try to infer the model with larger batch size, then TF-TRT will build another engine to do so.
        xs = [
            #tf.ones((64, 256, 256, 9), tf.float32),
            #tf.ones((32, 256, 256, 9), tf.float32),
            #tf.ones((1,  256, 256, 9), tf.float32),


            #tf.ones((128, 224, 224, 9), tf.float32),
            tf.ones((64, 224, 224, 9), tf.float32),
            tf.ones((32, 224, 224, 9), tf.float32),
            tf.ones((1,  224, 224, 9), tf.float32),

            #tf.ones((128, 128, 128, 9), tf.float32),
            tf.ones((64, 128, 128, 9), tf.float32),
            tf.ones((32, 128, 128, 9), tf.float32),
            tf.ones((1,  128, 128, 9), tf.float32)
        ]
        for x in xs:
            yield [x]

    converter.build(input_fn=input_fn)
    converter.save(output_saved_model_dir=output_saved_model_dir)
