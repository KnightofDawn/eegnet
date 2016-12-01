"""
Preprocessing of the dataset
"""

from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

def read_dataset(filenames,
                 num_splits=1,
                 batch_size=1,
                 is_training=True,
                 is_testing=False):
    """Reads the entire dataset"""
    tf.logging.info("Reading #%d files." % len(filenames))

    ## TFRecords description for Dataset reader
    keys_to_features = {
        'data': tf.FixedLenFeature([240000*16], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
        'filename': tf.FixedLenFeature([], tf.string),
    }
    items_to_handlers = {
        'data': slim.tfexample_decoder.Tensor('data'),
        'label': slim.tfexample_decoder.Tensor('label'),
        'filename': slim.tfexample_decoder.Tensor('filename'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    items_to_descriptions = {
        'data': '240000x16 channels sample points of iEEG.',
        'label': 'Label 0 indicates interictal and 1 preictal.',
        'filename': 'File name containing the data',
    }

    ## TFRecords files reading
    dataset = slim.dataset.Dataset(data_sources=filenames,
                                   reader=tf.TFRecordReader,
                                   decoder=decoder,
                                   num_samples=1,
                                   items_to_descriptions=items_to_descriptions)

    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                   shuffle=is_training,
                                                                   num_epochs=None,
                                                                   common_queue_capacity=10*batch_size,
                                                                   common_queue_min=5*batch_size)

    data, label = slim.utils.smart_cond(is_testing,
                                        lambda: data_provider.get(['data', 'filename']),
                                        lambda: data_provider.get(['data', 'label']))


    dim_pos = 1
    if is_testing:
        dim_pos = 0

    # testing | training -> dim_pos
    #    0          0    =    1   (eval)
    #    0          1    =    1   (training)
    #    1          0    =    0   (testing) 
    #    1          1    =    x

    # tf.logging.info("dim_pos=%d", dim_pos)
    # tf.logging.info("is_testing=%d", is_testing)
    # tf.logging.info("is_training=%d", is_training)

    ## Preprocess
    # Reshape data to original format: [width, channels]
    data = tf.reshape(data, shape=[240000, 16])

    if not is_testing:
        # Split data, if split = 1 only expands dim: [num_splits, width, channels]
        data = tf.pack(tf.split(0, num_splits, data), axis=0)

        # Detect dropout segments: indexes > sigma threshold
        _, var = tf.nn.moments(data, axes=[1, 2])
        # tf.where returns a 2D Tensor. reshape it to 1D
        idx_clean = tf.reshape(tf.where(tf.greater(var, 0.5)), shape=[-1])

        # Remove dropout segments based on training flag
        data = slim.utils.smart_cond(is_training,
                                     lambda: tf.gather(data, idx_clean),
                                     lambda: data)



    # Normalize: mean=0 and sigma=0.5
    data_mean = tf.expand_dims(tf.reduce_mean(data, reduction_indices=[dim_pos]), dim=dim_pos)
    data = tf.sub(data, data_mean)
    data_max = tf.expand_dims(tf.reduce_max(tf.abs(data), reduction_indices=[dim_pos]), dim=dim_pos)
    data = tf.div(data, tf.mul(2.0, data_max))

    # 3D tensor with height = 0: [height, width, channels]
    # 4D tensor with height = 1: [batch, height, width, channels]
    data = tf.expand_dims(data, dim=dim_pos)

    if not is_testing:
        # Create label array of segments
        num_segments = tf.shape(data)[0]
        label = tf.one_hot(label, 2, dtype=tf.int32)
        label = tf.reshape(tf.tile(label, [num_segments]), shape=[num_segments, 2])




    ## Batch 4D tensor: [batch, height, width, channels]
    shuffle_batch_fn = lambda: tf.train.shuffle_batch([data, label],
                                                      batch_size=batch_size,
                                                      capacity=5*num_splits*batch_size,
                                                      min_after_dequeue=3*num_splits*batch_size,
                                                      num_threads=1,
                                                      enqueue_many=True)

    batch_fn = lambda: tf.train.batch([data, label],
                                      batch_size=batch_size,
                                      capacity=5*num_splits*batch_size,
                                      num_threads=1,
                                      enqueue_many=not is_testing)

    data, label = slim.utils.smart_cond(is_training, shuffle_batch_fn, batch_fn)

    return data, label
