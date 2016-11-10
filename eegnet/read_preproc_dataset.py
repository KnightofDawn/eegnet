from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim


def preprocess_dataset(data, 
                       label,
                       num_points,
                       num_channels,
                       num_labels,
                       num_splits,
                       rem_dropouts):
    
    # Split data into smaller segments (speeds up trainning)
    data = tf.reshape(data, shape=[num_points, num_channels])
    data = tf.pack(tf.split(0, num_splits, data), axis=0)
    
    # Detect dropout segments (flow controlled by flag)
    _, var = tf.nn.moments(data, axes=[1, 2])
    # 'indexes > sigma threshold: tf.where' returns a 2D Tensor. reshape it to 1D.
    idx_clean = tf.reshape(tf.where(tf.greater(var, 0.5)), shape=[-1])
    
    # Remove dropout segments
    rem_dropouts_fn = lambda: tf.gather(data, idx_clean)
    id_fn = lambda: data
    # decide removal based on flag
    data = slim.utils.smart_cond(rem_dropouts, rem_dropouts_fn, id_fn)
    
    # Create label array of segments
    label = tf.one_hot(label, num_labels, dtype=tf.int32)
    num_segments = tf.shape(data)[0]
    label = tf.reshape(tf.tile(label, [num_segments]), shape=[num_segments, num_labels])
    
    # Normalize mean=0 and sigma=0.5
    data_mean = tf.expand_dims(tf.reduce_mean(data, reduction_indices=[1]), dim=1)
    data = tf.sub(data, data_mean)
    data_max = tf.expand_dims(tf.reduce_max(tf.abs(data), reduction_indices=[1]), dim=1)
    data = tf.div(data, tf.mul(2.0, data_max))
    
    # 4D tensor with height = 1: [batch, height, width, channels]
    data = tf.expand_dims(data, dim=1)
    
    return data, label
    

def read_dataset(filenames,
                 num_points=240000,
                 num_channels=16,
                 num_labels=2,
                 num_splits=10,
                 rem_dropouts=True,
                 batch_size=16, 
                 shuffle=True):
    
    tf.logging.info("Reading #%d files." % len(filenames))

    reader = tf.TFRecordReader

    keys_to_features = {
        'data': tf.FixedLenFeature([num_points*num_channels], tf.float32),
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
        'data': '240000 sample points of iEEG.',
        'label': 'Label 0 indicates interictal and 1 preictal.', 
        'filename': 'File name containing the data',
    }

    dataset = slim.dataset.Dataset(
        data_sources=filenames, 
        reader=reader, 
        decoder=decoder, 
        num_samples=1, 
        items_to_descriptions=items_to_descriptions)

    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                                   shuffle=shuffle, 
                                                                   num_epochs=None, 
                                                                   common_queue_capacity=40, 
                                                                   common_queue_min=20)

    data, label = data_provider.get(['data', 'label'])

    ## Preprocess
    data, label = preprocess_dataset(data, 
                                     label, 
                                     num_points, 
                                     num_channels, 
                                     num_labels, 
                                     num_splits, 
                                     rem_dropouts)

    ## Batch it up.
    shuffle_batch_fn = lambda: tf.train.shuffle_batch([data, label], 
                                                      batch_size=batch_size, 
                                                      capacity=4*num_splits*batch_size, 
                                                      min_after_dequeue=3*num_splits*batch_size, 
                                                      num_threads=1, 
                                                      enqueue_many=True)
    
    batch_fn = lambda: tf.train.batch([data, label], 
                                      batch_size=batch_size, 
                                      capacity=5*num_splits*batch_size, 
                                      num_threads=1, 
                                      enqueue_many=True)
    
    data, label = slim.utils.smart_cond(shuffle, shuffle_batch_fn, batch_fn)
    
    return data, label
