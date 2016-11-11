from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


from eegnet_v2 import eegnet_v2 as network
import read_preproc_dataset as read

##
# Directories
#

tf.app.flags.DEFINE_string('dataset_dir', '/shared/dataset/train/*.tfr', 
    'Where dataset TFReaders files are loaded from.')

tf.app.flags.DEFINE_string('log_dir', '/shared/logs/',
    'Where checkpoints and event logs are written to.')

##
# TFReaders configuration
##

tf.app.flags.DEFINE_integer('file_num_points', 240000,
                            'Data points in each TFReader file.')

tf.app.flags.DEFINE_integer('file_num_channels', 16,
                            'Sensor channels in each TFReader file.')

tf.app.flags.DEFINE_integer('file_num_splits', 1200,
                            'Splits to perform on each TFReader file.')

tf.app.flags.DEFINE_boolean('file_remove_dropouts', True,
                            'Remove or Not dropouts from input data.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Number of splits/files in each batch to the network.')

tf.app.flags.DEFINE_boolean('shuffle', True,
                            'Shuffle input data or not.')

##
# Network configuration
##

tf.app.flags.DEFINE_integer('num_labels', 2,
                            'Labels/classes being classified. 0 - Interictal | 1 - Preictal')

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                            'Convolutional filter width.')

FLAGS = tf.app.flags.FLAGS


def get_init_fn():
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.log_dir)
    
    if checkpoint_path is None:
        tf.logging.info('No checkpoint found in %s' % FLAGS.log_dir)
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
        train_data, train_labels = read.read_dataset(tf.gfile.Glob(FLAGS.dataset_dir), 
                                                     num_points=FLAGS.file_num_points,
                                                     num_channels=FLAGS.file_num_channels,
                                                     num_labels=FLAGS.num_labels,
                                                     num_splits=FLAGS.file_num_splits,
                                                     rem_dropouts=FLAGS.file_remove_dropouts,
                                                     batch_size=FLAGS.batch_size,
                                                     shuffle=FLAGS.shuffle)
        shape = train_data.get_shape().as_list()
        tf.logging.info('Batch size/num_points: %d/%d' % (shape[0], shape[2]))
        
        # Batch mixture: true labels / total labels
        mix = tf.div(tf.to_float(tf.reduce_sum(train_labels, 0)[1]), FLAGS.batch_size)
        tf.scalar_summary('batch_stats/Train batch mixture', mix)
        
        # Create model   
        logits, predictions = network(train_data,
                                      num_labels=FLAGS.num_labels,
                                      weight_decay=FLAGS.weight_decay,
                                      is_training=True)
        tf.logging.info('Network model created.')

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add summaries for activations: NOT WORKING YET. TF ERROR.
        #slim.summarize_activations()

        # Specify loss
        slim.losses.softmax_cross_entropy(logits, train_labels, scope='loss')
        total_loss = slim.losses.get_total_loss()
        # Summarize loss
        tf.scalar_summary('losses/Total loss', total_loss)

        # Optimizer and training op
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-4)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Train accuracy
        train_predictions = tf.one_hot(tf.argmax(predictions, 1), FLAGS.num_labels, dtype=tf.int32)
        train_accuracy = slim.metrics.accuracy(train_predictions, train_labels, 100.0)
        # Summarize train accuracy
        tf.scalar_summary('accuracy/Train batch accuracy', train_accuracy)

        # Run the training
        final_loss = slim.learning.train(train_op,
                                         logdir=FLAGS.log_dir, 
                                         log_every_n_steps=10, 
                                         is_chief=True, 
                                         number_of_steps=15001, 
                                         init_fn=get_init_fn(), 
                                         save_summaries_secs=5, 
                                         save_interval_secs=15*60, 
                                         trace_every_n_steps=None, 
                                        )
    
    
if __name__ == '__main__':
    tf.app.run()