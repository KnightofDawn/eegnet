from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

import json
import os


from eegnet.eegnet_v2 import eegnet_v2 as network
from eegnet.read_preproc_dataset import read_dataset

##
# Directories
##

tf.app.flags.DEFINE_string('dataset_dir', '/shared/dataset/train/*.tfr', 
    'Where dataset TFReaders files are loaded from.')

tf.app.flags.DEFINE_string('checkpoint_dir', None,
    'Where checkpoints are loaded from.')

tf.app.flags.DEFINE_string('log_dir', '/shared/logs',
    'Where checkpoints and event logs are written to.')

##
# Train configuration
##

tf.app.flags.DEFINE_boolean('is_training', True,
                            'Determines shuffling, dropout/batch_norm behaviour and drop-out removal.')

tf.app.flags.DEFINE_integer('num_splits', 1,
                            'Splits to perform on each TFRecord file.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'Training batch size.')

tf.app.flags.DEFINE_integer('num_iters', 5000,
                            'Number of training iterations.')

FLAGS = tf.app.flags.FLAGS


def get_init_fn():
    if FLAGS.checkpoint_dir is None:
        tf.logging.info('Skipping checkpoint load.')
        return None
    
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    
    if checkpoint_path is None:
        raise ValueError('No checkpoint found in %s. Supply a valid --checkpoint_dir' % FLAGS.checkpoint_dir)
    
    tf.logging.info('Loading model from %s' % checkpoint_path)
    
    return slim.assign_from_checkpoint_fn(
        model_path=checkpoint_path, 
        var_list=slim.get_model_variables(), 
        ignore_missing_vars=True)


def parameter_server_fn(cluster, task):
    tf.logging.info('Starting parameter server %d', task.index)
    
    # Start Server
    if not task.type:
        raise ValueError('--task_type must be specified.')
    if task.index is None:
        raise ValueError('--task_index must be specified.')
        
    server = tf.train.Server(tf.train.ClusterSpec(cluster),
                             protocol='grpc',
                             job_name=task.type,
                             task_index=task.index)
    server.join()

    
def worker_ps_fn(cluster, task):
    # Between-graph replication (https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html#replicated-training) 
    is_master = task.type != 'worker'
    if is_master and task.index > 0:
        raise StandardError('Only one replica of master expected')

    # Distributed
    if cluster:
        tf.logging.info('Starting %s/%d', task.type, task.index)

        # Start Server
        if not task.type:
            raise ValueError('--task_type must be specified.')
        if task.index is None:
            raise ValueError('--task_index must be specified.')
    
        server = tf.train.Server(tf.train.ClusterSpec(cluster),
                                 protocol='grpc',
                                 job_name=task.type,
                                 task_index=task.index)
            
        # Target where 'session' is going to run
        target = server.target

        # Device where to assign the replica
        worker_device = '/job:%s/task:%d' % (task.type, task.index)
        device_fn = tf.train.replica_device_setter(
            ps_device='/job:ps',
            worker_device=worker_device,
            cluster=cluster)
        # device_filter limits communication of this job to ps servers only, 
        # i.e., no comm with other workers, which would cause reliability problems.
        config = tf.ConfigProto(device_filters= ['/job:ps', worker_device])

    # Single
    else:
        target = ''
        device_fn = ''
        config = None

    with tf.Graph().as_default():
        with tf.device(device_fn):
            # Input pipeline
            data, labels = read_dataset(tf.gfile.Glob(FLAGS.dataset_dir),
                                        num_splits=FLAGS.num_splits,
                                        batch_size=FLAGS.batch_size,
                                        is_training=FLAGS.is_training)
            shape = data.get_shape().as_list()
            tf.logging.info('Batch size/num_points: %d/%d' % (shape[0], shape[2]))

            # Create model   
            logits, predictions = network(data, is_training=FLAGS.is_training)
            tf.logging.info('Network model created.')

            # Specify loss
            slim.losses.softmax_cross_entropy(logits, labels, scope='loss')
            total_loss = slim.losses.get_total_loss()
            # Summarize loss
            tf.summary.scalar('losses/total_loss', total_loss)

            # Optimizer and training op
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-2)
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            # Add histograms for trainable variables.
            for var in slim.get_model_variables():
                tf.summary.histogram(var.op.name, var)

            # Batch accuracy
            train_predictions = tf.one_hot(tf.argmax(predictions, 1), 2, dtype=tf.int32)
            train_accuracy = slim.metrics.accuracy(train_predictions, labels, 100.0)
            tf.summary.scalar('batch_stats/accuracy', train_accuracy)

            # Batch mixture: true labels / total labels
            mix = tf.div(tf.to_float(tf.reduce_sum(labels, 0)[1]), FLAGS.batch_size)
            tf.summary.scalar('batch_stats/labels_ratio', mix)

        # Run the training
        final_loss = slim.learning.train(train_op,
                                         logdir=FLAGS.log_dir,
                                         master=target,
                                         is_chief=is_master,
                                         number_of_steps=FLAGS.num_iters,
                                         init_fn=get_init_fn(),
                                         session_config=config,
                                         log_every_n_steps=5,
                                         save_summaries_secs=15,
                                         save_interval_secs=15*60)


def main(_):
    # Enable INFO level logging
    tf.logging.set_verbosity(tf.logging.INFO)
    
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Print the job data as provided by the service.
    tf.logging.info('Original job data: %s', env.get('job', {}))

    # First find out if there's a task value on the environmtent variable.
    # If there is none or it is empty define a default one.
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task = type('TaskSpec', (object,), task_data)
    trial = task_data.get('trial')
    
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    
    # Start Job/task
    if not cluster or not task or task.type == 'master' or task.type == 'worker':
        worker_ps_fn(cluster, task)
    elif task.type == 'ps':
        parameter_server_fn(cluster, task)
    else:
        raise ValueError('invalid task_type %s' % (task.type,))     
            
            
if __name__ == '__main__':
    tf.app.run()