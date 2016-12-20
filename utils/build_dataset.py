"""Converts synth90k data to TFRecord file format with SequenceExample protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import threading
import numpy as np
import tensorflow as tf

from collections import namedtuple
from datetime import datetime
from tqdm import tqdm

tf.flags.DEFINE_string("image_dir", "../data/synth90k/images",
                       "Image directory.")
tf.flags.DEFINE_string("imglist_file", "../data/synth90k/img_list.txt",
                       "Image list file.")
tf.flags.DEFINE_string("output_dir", "../data/synth90k/tf_records",
                       "Output data directory.")
tf.flags.DEFINE_string("output_info_file", "../data/synth90k/output_info.txt",
                       "Output dataset information file.")

tf.flags.DEFINE_integer("num_threads", 16,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_integer("train_shards", 128,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 32,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 32,
                        "Number of shards in testing TFRecord files.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "word"])


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to character_id.
        """
        print(vocab)
        self._vocab = vocab
        self.idx2char = dict((v, k) for k, v in self._vocab.iteritems())

    def char_to_id(self, char):
        """Returns the integer id of a character."""
        if char in self._vocab:
            return self._vocab[char]

    def id_to_char(self, idx):
        """Returns the character corresponding a char id"""
        if idx in self.idx2char:
            return str(self.idx2char[idx])

    def ids_to_word(self, ids):
        """Returns the word corresponding a list of char ids"""
        str_decoded = ''.join([self.id_to_char(x) for x in ids])
        return str_decoded


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._encoded_jpeg, channels=3
        )

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
    """Builds a SequenceExample proto for an image-word pair.

    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.

    Returns:
      A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image.filename, "r") as f:
        encoded_image = f.read()

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image.image_id),
        "image/data": _bytes_feature(encoded_image),
    })

    word = image.word
    word_ids = [vocab.char_to_id(char) for char in word]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/word": _bytes_feature_list(word),
        "image/word_ids": _int64_feature_list(word_ids),
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _create_vocab():
    """Creates the vocabulary of character to char_id.

    Returns:
      A Vocabulary object.
    """
    # Create vocabulary dictionary
    vocab_dict = {}

    # Blank token
    idx = 0
    vocab_dict['-'] = idx

    # 0-9
    for i in range(ord('9') - ord('0') + 1):
        idx += 1
        vocab_dict[chr(ord('0') + i)] = idx

    # a-z
    for i in range(ord('z') - ord('a') + 1):
        idx += 1
        vocab_dict[chr(ord('a') + i)] = idx

    # Create vocabulary object
    vocab = Vocabulary(vocab_dict)

    return vocab


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g.
        # 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-word pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-word pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]]
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images
    decoder = ImageDecoder()

    # Launch a thread for each batch
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all threads to terminate
    coord.join(threads)

    print("%s: Finished processing all %d image-word pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def _load_and_process_metadata(imglist_file, image_dir):
    """Loads image metadata from a file and processes the captions.

    Args:
      imlist_file:
      image_dir: Directory containing the image files.

    Returns:
      A list of ImageMetadata.
    """
    imglist_data = [line.rstrip('\n') for line in open(imglist_file)]

    # Extract data and combine the data into a list of ImageMetadata
    print("Processing raw data")
    image_metadata = []

    for line in imglist_data:
        try:
            image_id, base_filename = line.split(" ")
            filename = os.path.join(image_dir, base_filename)
            word = base_filename.split("_")[1].lower()
            image_metadata.append(ImageMetadata(int(image_id), filename, word))
        except Exception as e:
            print("File is not supported: %s" % filename)

    print("Finished processing %d images in %s" %
          (len(imglist_data), imglist_file))

    return image_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert FLAGS.image_dir, "--image_dir is required"
    assert FLAGS.imglist_file, "--imglist_file is required"
    assert FLAGS.output_dir, "--output_dir is required"
    assert FLAGS.output_info_file, "--output_info_file is required"

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    # Load image metadata
    dataset = _load_and_process_metadata(FLAGS.imglist_file, FLAGS.image_dir)

    # Distribute the dataset into:
    #   train_dataset = 90%
    #   val_dataset = 5%
    #   test_dataset = 5%
    train_cutoff = int(0.9 * len(dataset))
    val_cutoff = int(0.95 * len(dataset))
    train_dataset = dataset[0:train_cutoff]
    val_dataset = dataset[train_cutoff:val_cutoff]
    test_dataset = dataset[val_cutoff:]

    # Save dataset info to file
    with open(FLAGS.output_info_file, "w") as f:
        f.writelines("%s\n" % datetime.now())
        f.writelines("num_train_samples: %d\n" % len(train_dataset))
        f.writelines("num_val_samples: %d\n" % len(val_dataset))
        f.writelines("num_test_samples: %d\n" % len(test_dataset))
    tf.logging.info("Write dataset infomation into %s" %
                    FLAGS.output_info_file)

    # Create vocabulary
    vocab = _create_vocab()

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
