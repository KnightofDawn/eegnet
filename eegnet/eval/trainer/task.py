from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


from eegnet_v1 import eegnet_v1 as network
from read_preproc_dataset import read_dataset

##
# Directories
#

tf.app.flags.DEFINE_string('dataset_dir', '/content/dataset/train_gbucket/*.tfr', 
    'Where dataset TFReaders files are loaded from.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/content/logs/gcloud_test/',
    'Where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string('log_dir', '/content/logs/eval_test/',
    'Where checkpoints and event logs are written to.')

##
# TFReaders configuration
##

tf.app.flags.DEFINE_integer('file_num_splits', 50,
                            'Splits to perform on each TFReader file.')

tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Number of splits/files in each batch to the network.')

##
# Network configuration
##

tf.app.flags.DEFINE_boolean('is_training', False,
                            'Determines shuffling, dropout and batch norm behaviour and dropout removal.')


FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        # Input pipeline
        filenames = tf.gfile.Glob(FLAGS.dataset_dir)
        data, labels = read_dataset(filenames, 
                                    num_splits=FLAGS.file_num_splits,
                                    batch_size=FLAGS.batch_size,
                                    is_training=FLAGS.is_training)
        shape = data.get_shape().as_list()
        tf.logging.info('Batch size/num_points: %d/%d' % (shape[0], shape[2]))
        
        # Create model   
        logits, predictions = network(data, is_training=FLAGS.is_training)
        tf.logging.info('Network model created.')
        
        # Loss
        slim.losses.softmax_cross_entropy(logits, labels, scope='loss')
        
        # Accuracy
        predictions_onehot = tf.one_hot(tf.argmax(predictions, 1), 2, dtype=tf.int32)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'Stream Accuracy': slim.metrics.streaming_accuracy(predictions_onehot, labels, weights=100.0),
                'Stream Recall': slim.metrics.streaming_recall(predictions_onehot, labels, weights=100.0),
                'Stream Loss': slim.metrics.streaming_mean(slim.losses.get_total_loss(), weights=100.0),
            })
        
        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
            summary_name = 'eval/%s' % name
            op = tf.scalar_summary(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        
        # This ensures that we make a single pass over all of the data.
        num_batches = len(filenames)*FLAGS.file_num_splits//float(FLAGS.batch_size)
        
        #
        # Evaluate
        #
        # Get checkpoint path
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        
        tf.logging.info('Evaluating %s' % checkpoint_path)
        
        sv = tf.train.Supervisor(graph=graph,
                                 logdir=FLAGS.log_dir,
                                 summary_op=tf.merge_all_summaries(),
                                 summary_writer=tf.train.SummaryWriter(FLAGS.log_dir),
                                 global_step=slim.get_or_create_global_step(),
                                 saver=tf.train.Saver(slim.get_variables_to_restore()))
        with sv.managed_session(master='', start_standard_services=False) as sess:
            
            tf.logging.info('Starting evaluation.')
            # Load model from checkpoint
            sv.saver.restore(sess, checkpoint_path)
            # Start queues for TFRecords reading
            sv.start_queue_runners(sess)
            
            for i in range(int(num_batches)):
                tf.logging.info('Executing eval_op %d/%d', i + 1, num_batches)
                metric_values = sess.run(names_to_updates.values())
                
                output = dict(zip(names_to_values.keys(), metric_values))
                for name in output:
                    print('%s: %f' % (name, output[name]))

    
if __name__ == '__main__':
    tf.app.run()