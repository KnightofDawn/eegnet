from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


def eegnet_v1(batch_data,
              num_labels=2,
              filter_width=3,
              residual_channels=3*16,
              pool_size=2400,              
              reuse=None, 
              is_training=True,
              scope='eegnet_network'):
    
    with tf.variable_scope(scope, 'eegnet_network', reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], 
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                weights_initializer=slim.xavier_initializer(), 
                                normalizer_fn=slim.batch_norm):
                with tf.variable_scope('input_layer'):
                    hidden = slim.conv2d(batch_data, residual_channels, [1, filter_width], stride=1, rate=1, 
                                         activation_fn=None, scope='conv1')

                with tf.variable_scope('hidden'):
                    with tf.variable_scope('layer1'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=2, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = hidden # skip conn
                        hidden = tf.add(hidden, layer_input) # residual conn
                    with tf.variable_scope('layer2'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=4, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = tf.add(skip, hidden) # skip conn
                        hidden = tf.add(hidden, layer_input) # residual conn
                    with tf.variable_scope('layer3'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=8, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = tf.add(skip, hidden) # skip conn
                        hidden = tf.add(hidden, layer_input) # residual conn
                    with tf.variable_scope('layer4'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=16, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = tf.add(skip, hidden) # skip conn   
                        hidden = tf.add(hidden, layer_input) # residual conn
                    with tf.variable_scope('layer5'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=32, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = tf.add(skip, hidden) # skip conn  
                        hidden = tf.add(hidden, layer_input) # residual conn
                    with tf.variable_scope('layer6'):
                        layer_input = hidden
                        hidden = slim.conv2d(hidden, 2*residual_channels, [1, filter_width], stride=1, rate=64, 
                                             activation_fn=None, scope='dilconv')
                        filtr, gate = tf.split(3, 2, hidden) # split features in half
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.conv2d(hidden, residual_channels, 1, activation_fn=None, scope='1x1skip')
                        skip = tf.add(skip, hidden) # skip conn    
                        hidden = tf.add(hidden, layer_input) # residual conn
                    

                with tf.variable_scope('skip_processing'):
                    hidden = tf.nn.relu(skip)
                    batch_num_points = hidden.get_shape().as_list()[2]
                    hidden = slim.avg_pool2d(hidden, [1, batch_num_points*2//pool_size], 
                                             [1, batch_num_points//pool_size])
                    # 1 x 2400 x residual_channels
                    hidden = slim.conv2d(hidden, 32, 1, activation_fn=tf.nn.relu, scope='1x1compress1')
                    hidden = slim.conv2d(hidden, 16, [1, 8], stride=4, activation_fn=tf.nn.relu, scope='1x5reduce1')
                    # 1 x 600 x 16
                    hidden = slim.conv2d(hidden, 8, 1, activation_fn=tf.nn.relu, scope='1x1compress2')
                    hidden = slim.conv2d(hidden, 4, [1, 8], stride=4, activation_fn=tf.nn.relu, scope='1x5reduce2')
                    # 1 x 150 x 4
                    hidden = slim.conv2d(hidden, 2, 1, activation_fn=tf.nn.relu, scope='1x1compress3')
                    hidden = slim.conv2d(hidden, 2, [1, 6], stride=3, activation_fn=tf.nn.relu, scope='1x5reduce3')
                    # 1 x 75 x 2

                with tf.variable_scope('logits'):
                    hidden = slim.dropout(hidden, 0.7)
                    hidden = slim.flatten(hidden)
                    logits = slim.fully_connected(hidden, num_labels, activation_fn=None, 
                                                  normalizer_fn=None, scope='fc1')
    return logits
