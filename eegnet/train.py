from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


from eegnet_v1 import eegnet_v1 as network
from read_preproc_dataset import read_dataset

##
# Directories
##

tf.app.flags.DEFINE_string('dataset_dir', '/shared/dataset/train_small2/*.tfr', 
    'Where dataset TFReaders files are loaded from.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/shared/logs/train24/',
    'Where checkpoints are loaded from.')

tf.app.flags.DEFINE_string('log_dir', '/shared/logs/',
    'Where checkpoints and event logs are written to.')

##
# TFReaders configuration
##

tf.app.flags.DEFINE_integer('file_num_splits', 600,
                            'Splits to perform on each TFReader file.')

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Number of splits/files in each batch to the network.')

##
# Network configuration
##

tf.app.flags.DEFINE_boolean('is_training', True,
                            'Determines shuffling, dropout and batch norm behaviour and dropout removal.')

FLAGS = tf.app.flags.FLAGS


def get_init_fn():
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    
    if checkpoint_path is None:
        tf.logging.info('No checkpoint found in %s' % FLAGS.checkpoint_dir)
        return None
    
    tf.logging.info('Loading model from %s' % checkpoint_path)
    
    variables_to_restore = slim.get_model_variables()
    
    ## Create dictionary between old names and new ones
    #name_in_checkpoint = lambda var: var.op.name.replace("eegnet_v1", "eegnet_network")    
    #variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
    
    return slim.assign_from_checkpoint_fn(
        checkpoint_path, 
        variables_to_restore, 
        ignore_missing_vars=True,
    )


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Input pipeline
        data, labels = read_dataset(tf.gfile.Glob(FLAGS.dataset_dir), 
                                    num_splits=FLAGS.file_num_splits,
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
        tf.scalar_summary('losses/Total loss', total_loss)
        
        # Optimizer and training op
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-4)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        
        # Add histograms for trainable variables.
        for var in slim.get_model_variables():
            tf.histogram_summary(var.op.name, var)

        # Batch accuracy
        train_predictions = tf.one_hot(tf.argmax(predictions, 1), 2, dtype=tf.int32)
        train_accuracy = slim.metrics.accuracy(train_predictions, labels, 100.0)
        tf.scalar_summary('batch_stats/accuracy', train_accuracy)
        
        # Batch mixture: true labels / total labels
        mix = tf.div(tf.to_float(tf.reduce_sum(labels, 0)[1]), FLAGS.batch_size)
        tf.scalar_summary('batch_stats/labels ratio', mix)

        # Run the training
        final_loss = slim.learning.train(train_op,
                                         logdir=FLAGS.log_dir, 
                                         log_every_n_steps=10, 
                                         is_chief=True, 
                                         number_of_steps=125001, 
                                         init_fn=get_init_fn(), 
                                         save_summaries_secs=5, 
                                         save_interval_secs=15*60, 
                                         trace_every_n_steps=None, 
                                        )
    
    
if __name__ == '__main__':
    tf.app.run()