



import cnn_model as cnn_maker
import data_manager as dm



from tensorflow.contrib import learn
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.contrib.learn.python.learn import learn_runner
import argparse

import os
import json 
import logging
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def make_experiment_fn( args ):

    train_input_fn = dm.make_input_fn(
        args.train_data_path  ,
        dm.parse_examples ,
        args.batch_size ,
        num_epochs = args.num_epochs
    )
    eval_input_fn = dm.make_input_fn(
        args.eval_data_path ,
        dm.parse_examples ,
        args.batch_size ,
        num_epochs = args.num_epochs
    )

    def _experiment_fn( output_dir ):
        runconfig = learn.RunConfig(
            gpu_memory_fraction = 0.6 
        )
        estimator =  learn.Estimator(
            model_fn = cnn_maker.make_model( args.learning_rate) , 
            model_dir = output_dir ,
            config = runconfig
        )
        
        return learn.Experiment(
            estimator
            ,
            train_input_fn = train_input_fn ,
            eval_input_fn = eval_input_fn ,
            train_steps = args.max_steps,
            eval_metrics = cnn_maker.METRICS,  # AGREGAR METRICAS
            continuous_eval_throttle_secs = args.min_eval_seconds ,
            min_eval_frequency = args.min_train_eval_rate ,
            
        )
    return _experiment_fn
    

def metric_log_fn ( predictions , labels , weights = 1.0) :
    # funcion para calcular la log_loss
    loss = tf.losses.log_loss (
            predictions = result , labels = labels
        )
    return loss 
def run_evaluate( args ):
    # todo : implement this
    return ""

def run_train( args ):
    
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task', {'type': 'master', 'index': 0})
    logging.info('Original job data: %s', env.get('job', {}))
    trial = task_data.get('trial')
    if trial is not None:
        args.output_path = os.path.join( args.output_path , trial )
    learn_runner.run( make_experiment_fn(args) , args.output_path   )
    

def model_arguments(parser):

    group = parser.add_argument_group(
        title = "Model arguments" ,
        description = " model arguments, in this case the learning rate, which will be used by the hyperparameter tunnning engine by google ml. pass it a --learning-rate xxx"
    )

    group.add_argument( '--learning-rate' , type=float , default=0.01)

    return group

def path_arguments(parser):

    group = parser.add_argument_group(title= "Data paths for training")

    group.add_argument(
        '--train-data-path' ,
        type = str ,
        required = True,
        nargs = '+',
        help = 'Path to training data, local or GCS'
    )

    group.add_argument(
        '--output-path' ,
        type = str,
        required = True ,
        help = "path to save the output of the model, checkpoints and others. could be local or GCS"
    )
    group.add_argument(
        '--eval-data-path' ,
        type = str,
        required = True ,
        nargs='+' ,
        help= "Path to the eval data, local or GCS"
    )
    
    return group

def end_arguments(parser):

    group = parser.add_mutually_exclusive_group(required= True)

    group.add_argument(
        '--num-epochs' ,
        type = int ,
        
        help = "Number of epochs to run the model "
    )

    group.add_argument(
        '--max-steps' ,
        type = int ,
        help = " maximun numbert of steps to train the model"
    )
    
    return group 

def training_arguments(parser):

    group = parser.add_argument_group(title= "Training arguments")

    group.add_argument(
        '--batch-size' ,
        type = int,
        default = 64 ,
        help = "Number units  of processed per min batch "
    )
    
    group.add_argument(
        '--job-dir' ,
        type=str,
        help = "not used at the moment but required because im an idiot"
    )
    group.add_argument(
        '--min-eval-seconds' ,
        type = float ,
        default = 5 , 
        help = "Min interval between calculating eval metrics, and sav eval summaries "
    )
    group.add_argument(
        '--min-train-eval-rate' ,
        type = int ,
        default = 20 ,
        help="Minimal train/eval ratio on master "
    )
    
    return group
if __name__ == "__main__":
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.45

    
    parser = argparse.ArgumentParser()

    model_arguments( parser )
    path_arguments( parser )
    end_arguments( parser ) 
    training_arguments(parser )
    
    run_train(  parser.parse_args() )
    
    print("train")
