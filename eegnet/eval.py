from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


import eegnet_v1 as network
import read_preproc_dataset as read

##
# Directories
#

tf.app.flags.DEFINE_string('dataset_dir', '/shared/dataset/train_small/*.tfr', 
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

tf.app.flags.DEFINE_integer('file_num_splits', 1,
                            'Splits to perform on each TFReader file.')

tf.app.flags.DEFINE_boolean('file_remove_dropouts', False,
                            'Remove or Not dropouts from input data.')

tf.app.flags.DEFINE_integer('batch_size', 2,
                            'Number of splits/files in each batch to the network.')

tf.app.flags.DEFINE_boolean('shuffle', False,
                            'Shuffle input data or not.')

##
# Network configuration
##

tf.app.flags.DEFINE_integer('num_labels', 2,
                            'Labels/classes being classified. 0 - Interictal | 1 - Preictal')

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                            'Convolutional filter width.')


FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Input pipeline
        filenames = tf.gfile.Glob(FLAGS.dataset_dir)
        data, labels = read.read_dataset(filenames, 
                                         num_points=FLAGS.file_num_points,
                                         num_channels=FLAGS.file_num_channels,
                                         num_labels=FLAGS.num_labels,
                                         num_splits=FLAGS.file_num_splits,
                                         rem_dropouts=FLAGS.file_remove_dropouts,
                                         batch_size=FLAGS.batch_size,
                                         shuffle=FLAGS.shuffle)
        shape = data.get_shape().as_list()
        tf.logging.info('Batch size/num_points: %d/%d' % (shape[0], shape[2]))
        
        # Create model   
        logits, predictions = network.eegnet_v1(data,
                                                num_labels=FLAGS.num_labels,
                                                weight_decay=FLAGS.weight_decay,
                                                is_training=True)
        tf.logging.info('Network model created.')
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'Accuracy': slim.metrics.streaming_accuracy(predictions, labels, weights=100.0),
                'Recall': slim.metrics.streaming_recall(logits, labels, weights=100.0),
            })
        
        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
            summary_name = 'eval/%s' % name
            op = tf.scalar_summary(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        
        # This ensures that we make a single pass over all of the data.
        num_batches = tf.ceil(len(filenames)*FLAGS.file_num_channels / float(FLAGS.batch_size))
        
        # Get checkpoint path
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.log_dir)
        if checkpoint_path is None:
            tf.logging.info('No checkpoint found in %s' % FLAGS.log_dir)
            tf.logging.info('Finished evaluation.')
            return
        else:
            tf.logging.info('Loading model from %s' % checkpoint_path)
        
        tf.logging.info('Evaluating %s' % checkpoint_path)

        #slim.evaluation.evaluate_once(
        #    master='',
        #    checkpoint_path=checkpoint_path,
        #    logdir=FLAGS.log_dir,
        #    num_evals=10,
        #    eval_op=names_to_updates.values(),
        #    variables_to_restore=slim.get_variables_to_restore())
        
        initial_op = tf.group(
            tf.initialize_all_variables(),
            tf.initialize_local_variables())
        
        with tf.Session() as sess:
            metric_values = slim.evaluation(
                sess,
                num_evals=1,
                initial_op=initial_op,
                eval_op=names_to_updates.values(),
                final_op=names_to_values.values())
            
            for metric, value in zip(names_to_values.keys(), metric_values):
                logging.info('Metric %s has value: %f', metric, value)
    
if __name__ == '__main__':
    tf.app.run()