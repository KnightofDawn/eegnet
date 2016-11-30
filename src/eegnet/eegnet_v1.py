"""
The NN basic methods
"""

from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS


def get_init_fn(continue_oncheck=False):
    """Loads the NN"""
    if FLAGS.checkpoint_dir is None:
        if continue_oncheck:
            return None
        else:
            raise ValueError('No checkpoint provided, using --checkpoint_dir')

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if checkpoint_path is None:
        raise ValueError('No checkpoint found in %s. Supply a valid --checkpoint_dir' %
                         FLAGS.checkpoint_dir)

    tf.logging.info('Loading model from %s', checkpoint_path)

    return slim.assign_from_checkpoint_fn(model_path=checkpoint_path,
                                          var_list=slim.get_model_variables(),
                                          ignore_missing_vars=True)



def dilated_block(hidden, rate, scope):
    """Base Wavenet NN Architecture"""
    with tf.variable_scope(scope):
        # Residual
        layer_input = hidden
        # Compress layer input features
        hidden = slim.conv2d(hidden, 16, 1, scope='1x1compress')
        # Dilated convolution. NO nonlinear and batch norm!
        hidden = slim.conv2d(hidden, 8, [1, 3], rate=rate, normalizer_fn=None,
                             activation_fn=None, scope='dilconv')
        # Split features (dim=3) in half
        filtr, gate = tf.split(3, 2, hidden)
        # Apply nonlinear functions and batch_norm
        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
        hidden = slim.batch_norm(hidden, scope='norm_filterXgate')
        # Output to add with residual. NO nonlinear and batch norm!
        hidden = slim.conv2d(hidden, layer_input.get_shape()[3], 1,
                             normalizer_fn=None, activation_fn=None, scope='1x1toRes')
        # Add output and residual -> input for next layer
        return tf.add(hidden, layer_input)



def eegnet_v1(inputs,
              weight_decay=0.00004,
              reuse=None,
              is_training=True,
              scope='eegnet_v1'):
    """The Total NN"""
    with tf.variable_scope(scope, 'eegnet_v1', reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_regularizer=slim.l2_regularizer(weight_decay),
                                normalizer_fn=slim.batch_norm):

                with tf.variable_scope('input_layer'):
                    hidden = slim.conv2d(inputs, 48, [1, 3], scope='conv1')

                with tf.variable_scope('hidden'):
                    hidden = dilated_block(hidden, 2, 'layer1')
                    hidden = dilated_block(hidden, 4, 'layer2')
                    hidden = dilated_block(hidden, 8, 'layer3')
                    hidden = dilated_block(hidden, 16, 'layer4')
                    hidden = dilated_block(hidden, 2, 'layer5')
                    hidden = dilated_block(hidden, 4, 'layer6')

                with tf.variable_scope('logits'):
                    batch_num_points = hidden.get_shape().as_list()[2]

                    hidden = slim.avg_pool2d(hidden,
                                             [1, batch_num_points*2//2400],
                                             [1, batch_num_points//2400])

                    # 1 x 2400 x 48
                    hidden = slim.conv2d(hidden, 16, 1, scope='1x1compress')
                    # 1 x 2400 x 16
                    hidden = slim.conv2d(hidden, 8, [1, 5], stride=3, scope='1x3reduce1')
                    # 1 x 800 x 8
                    hidden = slim.conv2d(hidden, 2, [1, 7], stride=5, scope='1x3reduce2')
                    # 1 x 160 x 2

                    hidden = slim.flatten(hidden)

                    hidden = slim.dropout(hidden, 0.8)

                    logits = slim.fully_connected(hidden, 2, normalizer_fn=None,
                                                  activation_fn=None, scope='logits')
                    predictions = tf.nn.softmax(logits)

    return logits, predictions
