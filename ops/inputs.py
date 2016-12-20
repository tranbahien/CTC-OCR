"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from scipy import sparse


def parse_sequence_example(serialized, image_feature, word_feature):
    """Parses a tensorflow.SequenceExample into an image and caption.

    Args:
        serialized: A scalar string Tensor; a single serialized SequenceExample.
        image_feature: Name of SequenceExample context feature containing image
            data.
        word_feature: Name of SequenceExample feature list containing integer
            chars.

    Returns:
        encoded_image: A scalar string Tensor containing a JPEG encoded image.
        word: A 1-D uint64 Tensor with dynamically specified length.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            image_feature: tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            word_feature: tf.VarLenFeature(dtype=tf.int64),
        })

    encoded_image = context[image_feature]
    word = sequence[word_feature]

    return encoded_image, word


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """Prefetches string values from disk into an input queue.

    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
      reader: Instance of tf.ReaderBase.
      file_pattern: Comma-separated list of file patterns (e.g.
          /tmp/train_data-?????-of-00100).
      is_training: Boolean; whether prefetching for training or eval.
      batch_size: Model batch size used to determine queue capacity.
      values_per_shard: Approximate number of values per shard.
      input_queue_capacity_factor: Minimum number of values to keep in the queue
        in multiples of values_per_shard. See comments above.
      num_reader_threads: Number of reader threads to fill the queue.
      shard_queue_name: Name for the shards filename queue.
      value_queue_name: Name for the values input queue.

    Returns:
      A Queue containing prefetched string values.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue


def convert_to_sparse(x):
    """Convert a dense numpy array to sparse array.

    Args:
        x: A numpy array.

    Returns:
        A tuple of indices, values, shape of created sparse array.
    """
    sparse_x = sparse.coo_matrix(x)
    indices = np.array([[row, col] for row, col in
                        zip(sparse_x.row, sparse_x.col)], dtype=np.int64)

    values = sparse_x.data.astype(np.int32)
    shape = np.array(sparse_x.shape, dtype=np.int64)

    return indices, values, shape


def make_batch(images_and_words,
               batch_size,
               queue_capacity,
               vocab_size):
    """Batches input images and words.

    Args:
        images_and_words: A list of pairs[image, words].
        batch_size: Batch size.
        queue_capacity: Queue capacity.
        vocab_size: The size of vocabulary

    Returns:
        images: A Tensor of shape [batch_size, height, width, channels].
        targets:
            An int32 Tensor of shape[batch_size, padded_length].
    """
    # Create queue containing data samples
    enqueue_list = []
    for image, word in images_and_words:
        seq_length = word.shape[0]
        indicator = tf.ones([tf.to_int32(seq_length)], dtype=tf.int32)
        enqueue_list.append([image, word, seq_length, indicator])

    # Create batch of data from queue
    images, targets, seq_lengths, indicator = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch")

    # Reshape the target into the suitable format
    seq_lengths = tf.to_int32(seq_lengths, name='SeqLengthsToInt32')
    targets = tf.to_int32(targets, name='TargetToInt32')
    shape = targets.shape
    targets = tf.sparse_reshape(targets, [shape[0], shape[1]])

    return images, targets, seq_lengths
