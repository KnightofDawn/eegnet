"""
The main runtime file
"""

from __future__ import print_function
import tensorflow as tf
import tf.contrib.slim as slim
from trainer.eegnet_v1 import eegnet_v1 as network
from trainer.read_preproc_dataset import read_dataset

##
# Directories
##
tf.app.flags.DEFINE_string('dataset_dir', '/shared/dataset/eval/*.tfr',
                           'Where dataset TFReaders files are loaded from.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/shared/checkpoints',
                           'Where checkpoints are loaded from.')

tf.app.flags.DEFINE_string('log_dir', '/shared/logs',
                           'Where checkpoints and event logs are written to.')

##
# Train configuration
##
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Determines shuffling, dropout/batch_norm behaviour and removal.')

tf.app.flags.DEFINE_integer('num_splits', 1,
                            'Splits to perform on each TFRecord file.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'Training batch size.')

FLAGS = tf.app.flags.FLAGS


def get_init_fn():
    """Loads the NN"""
    if FLAGS.checkpoint_dir is None:
        raise ValueError('None supplied. Supply a valid checkpoint directory with --checkpoint_dir')

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if checkpoint_path is None:
        raise ValueError('No checkpoint found in %s. Supply a valid --checkpoint_dir' %
                         FLAGS.checkpoint_dir)

    tf.logging.info('Loading model from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        model_path=checkpoint_path,
        var_list=slim.get_model_variables(),
        ignore_missing_vars=True)


def main(_):
    """Generates the TF graphs and loads the NN"""
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        # Input pipeline
        filenames = tf.gfile.Glob(FLAGS.dataset_dir)
        data, labels = read_dataset(filenames,
                                    num_splits=FLAGS.num_splits,
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
            'stream_accuracy': slim.metrics.streaming_accuracy(predictions_onehot, labels),
            'stream_recall': slim.metrics.streaming_recall(predictions_onehot, labels),
            'stream_loss': slim.metrics.streaming_mean(slim.losses.get_total_loss()),
            })

        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
            summary_name = 'eval/%s' % name
            tf.summary.scalar(summary_name, value)

        # This ensures that we make a single pass over all of the data.
        num_batches = len(filenames)*FLAGS.num_splits//float(FLAGS.batch_size)

        #
        # Evaluate
        #
        supervi = tf.train.Supervisor(graph=graph,
                                      logdir=FLAGS.log_dir,
                                      summary_op=tf.merge_all_summaries(),
                                      summary_writer=tf.train.SummaryWriter(FLAGS.log_dir),
                                      save_summaries_secs=5,
                                      global_step=slim.get_or_create_global_step(),
                                      init_fn=get_init_fn()) # restores checkpoint

        with supervi.managed_session(master='', start_standard_services=False) as sess:
            tf.logging.info('Starting evaluation.')
            # Start queues for TFRecords reading
            supervi.start_queue_runners(sess)

            for i in range(int(num_batches)):
                tf.logging.info('Executing eval_op %d/%d', i + 1, num_batches)
                metric_values = sess.run(names_to_updates.values())

                output = dict(zip(names_to_values.keys(), metric_values))
                for name in output:
                    tf.logging.info('%s: %f' % (name, output[name]))


if __name__ == '__main__':
    tf.app.run()
