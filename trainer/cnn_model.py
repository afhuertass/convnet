from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



#from tensorflow.contrib import learn

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.tfprof import model_analyzer  as model_analyzer
#from tensorflow.contrib import learn

#from tensorflow.contrib.learn.ModelFnOps import model_fn as model_fn_lib 
from tensorflow.contrib.metrics.python.ops import metric_ops


from tensorflow.contrib.learn.python.learn import metric_spec
import tensorflow as tf
import sys 
"""
 for building the cnn model. 

"""


def make_model(learning_rate):

    def cnn_model( features , labels , mode  ):

    
        feat_size = 150
        batch_size = 100
    
       
        input_layer = tf.reshape( features , [-1 , feat_size, feat_size , feat_size , 1 ]  )
        labels = tf.reshape( labels , [ -1 , 1] ) 
        print("shape cnn")
        print( labels.shape )
        print( input_layer.shape ) 

        # inputs = [ batch_size , 150 , 150 , 150 , 1 ]
        conv1 = tf.layers.conv3d(
            inputs = input_layer , 
            filters = 4 ,
            kernel_size = [ 5 , 5 , 5] ,
            padding = "same" ,
            activation = tf.nn.sigmoid 
            
        )
              
        pool1 = tf.layers.max_pooling3d(
            inputs = conv1 ,
            pool_size = [10,10,10],
            strides = 10
        )
        print("shape layer1")
        print( conv1.shape)
        print(pool1.shape)
        # pool1 = [batch_size , 30, 30, 30 , 32 ]
      

        """
        conv2 = tf.layers.conv3d(
            inputs = pool1 ,
            filters = 16 ,
            kernel_size = [ 5 , 5, 5 ] ,
            padding = "same" ,
            activation = tf.nn.sigmoid 
        )
        #conv2 = [ batch_size , 30, 30 , 30 , 64]

        pool2 = tf.layers.max_pooling3d(
            inputs = conv2 ,
            pool_size = [ 5 , 5 , 5 ] , 
            strides = 5 
        )
       
        
        # pool = [ batch_size , 6 , 6 , 6 , 64]
        #size capa final = 5*5*5*128
        #size = 15*15*15*2  # cambiar 5 por 2
        print("shapes second layer")
        print( conv2.shape )
        print(pool2.shape)

         """
        #size = 15*15*15*8
        size = 15*15*15*4
        pool_flat = tf.reshape( pool1 , [ -1 , size ] )
        # 43 MB 
        print("shape pool_flat")
        print(pool_flat.shape)
        
        dropout = tf.layers.dropout(
            inputs = pool_flat , rate = 0.4 , training = mode == model_fn_lib.ModeKeys.TRAIN 
        )
        
        result = tf.layers.dense(
            inputs = dropout,
            units = 1 , 
            activation  = tf.nn.sigmoid 
        )
        # activation = tf.nn.relu
        """
        result = tf.layers.dense(
            inputs = dense1, 
            units  = 1 ,
            activation =  tf.nn.relu 
        )
        """
        # 
        #
        print( "shape result") 
        print( result.shape ) 
        loss = None
        
        train_op = None
        

        param_stats = model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

        print("STAAAAAP")
        #infer and test mode
        if mode != model_fn_lib.ModeKeys.INFER:

            #onehot_labels = tf.one_hot( indices = tf.cast(labels, tf.int32 ) , depth = 1    )
            loss = tf.losses.log_loss (
                predictions = result , labels = labels 
            )
            

        print (loss.shape )
            

        if mode == model_fn_lib.ModeKeys.TRAIN:
            print("TRAININIGNIGNI")
            print( input_layer.shape )
            """
            train_op = tf.contrib.layers.optimize_loss(
                loss = loss,
                global_step = tf.contrib.framework.get_global_step(),
                learning_rate = learning_rate ,
                optimizer = "Adam"
                
            )
            """
            global_step = tf.contrib.framework.get_global_step()
            #optimizer = tf.train.GradientDescentOptimizer( learning_rate )
            
            optimizer = tf.train.AdamOptimizer( learning_rate , epsilon = 0.0001 )
            train_op = optimizer.minimize( loss , global_step = global_step )

            tf.summary.scalar( "log_loss", loss )
            
        else:
            train_op = None
            
            print("Training not build ")

        predictions = {
                      
            "probabilities": tf.nn.softmax(
                result , name = "softmax_tensor"
            ) , 
            
            "loss" : loss 
        }
        return predictions , loss , train_op
    
       # return model_fn_lib.ModelFnOps( mode=mode, predictions=predictions, loss=loss, train_op=train_op)
      
    return cnn_model


def metrics_wr( values , omit  ):
    weights = tf.ones( shape =() )
    
    return metric_ops.streaming_mean(values  , weights )


METRICS = {
    'loss' : metric_spec.MetricSpec (
        metric_fn = metrics_wr , 
        prediction_key = 'loss' ,
        weight_key = None ,
        label_key = None 
    )
    
}


        
