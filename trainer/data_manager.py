


import numpy as np

import multiprocessing

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


SIZE_SCAN = 150*150*150 
def parse_examples(examples):
    print("parse samples ")
    feature_map = {
        'label' : tf.FixedLenFeature (
            shape = () , dtype = tf.int64 , default_value = [-1 ] ) ,
        
        'images' : tf.FixedLenFeature (
            shape = [SIZE_SCAN] , dtype = tf.float32 ) 

    }

    features = tf.parse_example( examples , features=feature_map)
    print("examplesss parseddd ")
    print(features['images'].shape )
    
    return features['images'] , features['label']

def make_input_fn( files , example_parser  , batch_size  , num_epochs= 10 ):


    def _input_fn():
        print( files )
        thread_count = multiprocessing.cpu_count()
        print( thread_count )
        min_after_dequeue = 1000

        queue_size_multiplier = thread_count + 3

        #filename_queue = tf.train.string_input_producer( files , num_epochs = num_epochs)
        print( num_epochs )
        filename_queue = tf.train.string_input_producer( files , num_epochs = num_epochs   )
        
        _ , encoded_examples = tf.TFRecordReader(
            options = tf.python_io.TFRecordOptions  (
                compression_type= TFRecordCompressionType.GZIP
            )
        ).read_up_to( filename_queue , batch_size)

        

        features, targets = example_parser( encoded_examples )
        capacity = batch_size*2
        result = tf.train.shuffle_batch(
            [features , targets ] , batch_size , num_threads = 2 ,
            capacity = capacity , min_after_dequeue = batch_size ,
            enqueue_many = True 
        )
        print("ress")
        result[1] = tf.reshape( result[1] , [-1 , 1]   )
        
        print(result[1].shape)
        print( type(  result ) )
        return result
        
        
        """
        return tf.train.shuffle_batch(
            [ features , targets ]  ,
            batch_size ,
            capacity ,
            min_after_dequeue ,
            enqueue_many = True ,
            num_threads = thread_count
        )
        """
    return _input_fn

