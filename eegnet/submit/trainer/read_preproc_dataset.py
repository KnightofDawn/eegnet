"""
Preprocessing of the dataset
"""

from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


def read_dataset(filenames,
                 num_splits=1,
                 batch_size=1):
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
                                                                   shuffle=False,
                                                                   num_epochs=None,
                                                                   common_queue_capacity=10*batch_size,
                                                                   common_queue_min=5*batch_size)
    data, filename = data_provider.get(['data', 'filename'])

    ## Preprocess
    # Reshape data to original format: [width, channels]
    data = tf.reshape(data, shape=[240000, 16])

    # Normalize: mean=0 and sigma=0.5
    data_mean = tf.expand_dims(tf.reduce_mean(data, reduction_indices=[0]), dim=0)
    data = tf.sub(data, data_mean)
    data_max = tf.expand_dims(tf.reduce_max(tf.abs(data), reduction_indices=[0]), dim=0)
    data = tf.div(data, tf.mul(2.0, data_max))

    # 3D tensor with height = 1: [height, width, channels]
    data = tf.expand_dims(data, dim=0)

    return tf.train.batch([data, filename],
                          batch_size=batch_size,
                          capacity=5*num_splits*batch_size,
                          num_threads=1)









