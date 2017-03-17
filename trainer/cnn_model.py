from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



#from tensorflow.contrib import learn

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

#from tensorflow.contrib import learn

#from tensorflow.contrib.learn.ModelFnOps import model_fn as model_fn_lib 
from tensorflow.contrib.metrics.python.ops import metric_ops

from tensorflow.contrib.learn.python.learn import metric_spec
import tensorflow as tf
"""
 for building the cnn model. 

"""

METRICS = {
    'loss' : metric_spec.MetricSpec (
        metric_fn = metric_ops.streaming_mean , 
        prediction_key = 'loss'
    )
    
}

def make_model(learning_rate):

    def cnn_model( features , labels , mode  ):

    # imagenes 400x400x400 , un canal de color
        feat_size = 150
        batch_size = 100
    
        # input shape = [ batch_size , 150,150,150,1]
        input_layer = tf.reshape( features , [-1 , feat_size, feat_size , feat_size , 1 ]  )
        labels = tf.reshape( labels , [ -1 , 1] ) 
        print("shape cnn")
        print( labels.shape )
        print( input_layer.shape ) 

        # inputs = [ batch_size , 150 , 150 , 150 , 1 ]
        conv1 = tf.layers.conv3d(
            inputs = input_layer , 
            filters = 5 ,
            kernel_size = [ 5 , 5 , 5] ,
            padding = "same" ,
            activation = tf.nn.relu
            
        )
        # conv1 = [ batch_size , 150,150,150, 32]
        
        pool1 = tf.layers.max_pooling3d(
            inputs = conv1 ,
            pool_size = [3,3,3],
            strides = 3
        )
        # pool1 = [batch_size , 75,75,75 , 32]
        print("shape layer1")
        print(conv1.shape)
        print( pool1.shape )
        #compute 64 features
        conv2 = tf.layers.conv3d(
            inputs = pool1 ,
            filters = 12,
            kernel_size = [ 5 , 5, 5] , 
            padding = "same" , 
            activation  = tf.nn.relu
        )
        # conv2 = [ batch_size , 75,75,75,64]
    
        pool2 = tf.layers.max_pooling3d(
            inputs = conv2 ,
            pool_size = [ 10  ,10, 10  ] ,
            strides = 10
        )
        # pool2 = [ batch_size , 15,15,15, 64]
    
        print("shape layer2")
        print(conv2.shape)
        print( pool2.shape )

        """
        conv3 = tf.layers.conv3d(
            inputs = pool2 ,
            filters = 128 ,
            kernel_size = [15,15,15] ,
            padding = "same" ,
            activation = tf.nn.relu 
        )
        #conv3 = [batch_size , 15,15,15,128]
    
        pool3 = tf.layers.max_pooling3d(
            inputs= conv3 ,
            pool_size = [2  , 2 ,2 ],
            strides = 2 
        )
        # pool3 = [batch_size , 5, 5, 5, 128]
       
        print("shape layer3")
        print(conv3.shape)
        print( pool3.shape )
        """
        # check dimensiones
        # 7*7*64
        # 2239488
        #size capa final = 5*5*5*128
        size = 5*5*5*12
        pool_flat = tf.reshape( pool2 , [ -1 , size ] )
        
        print("shape pool_flat")
        print(pool_flat.shape)
        
        dropout = tf.layers.dropout(
            inputs = pool_flat , rate = 0.4 , training = mode == model_fn_lib.ModeKeys.TRAIN 
        )
        
        dense1 = tf.layers.dense(
            inputs = dropout,
            units = 1000 , 
            activation = tf.nn.relu
        )

        result = tf.layers.dense(
            inputs = dense1, 
            units  = 1
        )
        # 
        #
        print( "shape result") 
        print( result.shape ) 
        loss = None
        
        train_op = None
        
        #infer and test mode
        if mode != model_fn_lib.ModeKeys.INFER:

            #onehot_labels = tf.one_hot( indices = tf.cast(labels, tf.int32 ) , depth = 1    )
            loss = tf.losses.log_loss (
                predictions = result , labels = labels 
        )

            

        if mode == model_fn_lib.ModeKeys.TRAIN:

            train_op = tf.contrib.layers.optimize_loss(
                loss = loss,
                global_step = tf.contrib.framework.get_global_step(),
                learning_rate = learning_rate ,
                optimizer = "SGD"
                
            )
            tf.summary.scalar( 'log_loss' , loss )
            

        predictions = {
            "classes" : tf.argmax(
                input = result ,
                axis = 1 
            ) ,
            
            "probabilities": tf.nn.softmax(
                result , name = "softmax_tensor"
            ) , 
            
            "dense_layer" : dense1 ,
            
            "loss" : loss 
        }
        
    
        return model_fn_lib.ModelFnOps( mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    return cnn_model
