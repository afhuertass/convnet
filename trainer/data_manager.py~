


import numpy as np

import multiprocessing

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


SIZE_SCAN = 150*150*150 
def parse_examples(examples):

    feature_map = {
        'labels' : tf.FixedLenFeature (
            shape = [] , dtype = tf.int64 , default_value = [-1 ] ) ,
        
        'images' : tf.FixedLenFeature (
            shape = [SIZE_SCAN] , dtype = tf.float32 ) 

    }

    features = tf.parse_example( examples , features=feature_map)

    return features['images'] , features['labels']

def make_input_fn( files , example_parser  , batch_size  , num_epochs= 10 ):


    def _input_fn():
        print( files )
        thread_count = multiprocessing.cpu_count()

        min_after_dequeue = 1000

        queue_size_multiplier = thread_count + 3

        filename_queue = tf.train.string_input_producer( files , num_epochs = num_epochs)

        example_id , encoded_examples = tf.TFRecordReader(
            options = tf.python_io.TFRecordOptions  (
                compression_type= TFRecordCompressionType.GZIP
            )
        ).read_up_to( filename_queue , batch_size)

        

        features, targets = example_parser(encoded_examples )
        capacity = min_after_dequeue + queue_size_multiplier

        return tf.train.shuffle_batch(
            [ features , targets ]  ,
            batch_size ,
            capacity ,
            min_after_dequeue ,
            enqueue_many = True ,
            num_threads = thread_count
        )

    return _input_fn

