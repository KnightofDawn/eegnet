from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


def eegnet_v1(inputs,
              num_labels=2,
              weight_decay=0.00004,
              reuse=None, 
              is_training=True,
              scope='eegnet_v1'):
    
    with tf.variable_scope(scope, 'eegnet_v1', reuse=reuse):
        
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                activation_fn=tf.nn.relu, 
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_regularizer=slim.l2_regularizer(weight_decay),
                                normalizer_fn=slim.batch_norm):
                
                with tf.variable_scope('input_layer'):
                    hidden = slim.conv2d(inputs, 32, [1, 3], scope='conv1')
                
                with tf.variable_scope('hidden'):
                    with tf.variable_scope('layer1'):
                        layer_input = hidden
                        # Compress layer input features
                        hidden = slim.conv2d(hidden, 16, 1, scope='1x1compress')
                        # Dilated convolution. NO nonlinear and batch norm!
                        hidden = slim.conv2d(hidden, 8, [1, 3], rate=2, normalizer_fn=None, 
                                             activation_fn=None, scope='dilconv')
                        # Split features in half
                        filtr, gate = tf.split(3, 2, hidden)
                        # Apply nonlinear functions and batch_norm
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.batch_norm(hidden, scope='norm_filterXgate')
                        # 1x1conv for residual extraction
                        hidden = slim.conv2d(hidden, layer_input.get_shape()[3], 1, 
                                             normalizer_fn=None, activation_fn=None, scope='1x1toRes')
                        # Add to skip connections
                        skip = hidden
                        # Add layer input and residual -> input for next layer
                        hidden = tf.add(hidden, layer_input)
                        
                    with tf.variable_scope('layer2'):
                        layer_input = hidden
                        # Compress layer input features
                        hidden = slim.conv2d(hidden, 16, 1, scope='1x1compress')
                        # Dilated convolution. NO nonlinear and batch norm!
                        hidden = slim.conv2d(hidden, 8, [1, 3], rate=4, normalizer_fn=None, 
                                             activation_fn=None, scope='dilconv')
                        # Split features in half
                        filtr, gate = tf.split(3, 2, hidden)
                        # Apply nonlinear functions and batch_norm
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.batch_norm(hidden, scope='norm_filterXgate')
                        # 1x1 Residual extraction. NO nonlinear and batch norm!
                        hidden = slim.conv2d(hidden, layer_input.get_shape()[3], 1, 
                                             normalizer_fn=None, activation_fn=None, scope='1x1toRes')
                        # Add to skip connections
                        skip = tf.add(skip, hidden)
                        # Add layer input and residual -> input for next layer
                        hidden = tf.add(hidden, layer_input)
                        
                    with tf.variable_scope('layer3'):
                        layer_input = hidden
                        # Compress layer input features
                        hidden = slim.conv2d(hidden, 16, 1, scope='1x1compress')
                        # Dilated convolution. NO nonlinear and batch norm!
                        hidden = slim.conv2d(hidden, 8, [1, 3], rate=8, normalizer_fn=None, 
                                             activation_fn=None, scope='dilconv')
                        # Split features in half
                        filtr, gate = tf.split(3, 2, hidden)
                        # Apply nonlinear functions and batch_norm
                        hidden = tf.mul(tf.tanh(filtr), tf.sigmoid(gate), name='filterXgate')
                        hidden = slim.batch_norm(hidden, scope='norm_filterXgate')
                        # 1x1 Residual extraction. NO nonlinear and batch norm!
                        hidden = slim.conv2d(hidden, layer_input.get_shape()[3], 1, 
                                             normalizer_fn=None, activation_fn=None, scope='1x1toRes')
                        # Add to skip connections
                        skip = tf.add(skip, hidden)
                        # Add layer input and residual -> input for next layer
                        hidden = tf.add(hidden, layer_input)
                        
                   

                with tf.variable_scope('skip_processing'):
                    hidden = tf.nn.relu(skip)
                    batch_num_points = hidden.get_shape().as_list()[2]
                    hidden = slim.avg_pool2d(hidden, [1, batch_num_points*2//2400], 
                                             [1, batch_num_points//2400])
                    # 1 x 2400 x 32
                    hidden = slim.conv2d(hidden, 4, 1, scope='1x1compress')
                    hidden = slim.conv2d(hidden, 2, [1, 5], stride=3, scope='1x5reduce')
                    # 1 x 800 x 2

                with tf.variable_scope('logits'):
                    hidden = slim.dropout(hidden, 0.8)
                    hidden = slim.flatten(hidden)
                    logits = slim.fully_connected(hidden, num_labels, normalizer_fn=None, 
                                                  activation_fn=None, scope='logits')
                    predictions = tf.nn.softmax(logits)
                        
    return logits, predictions
