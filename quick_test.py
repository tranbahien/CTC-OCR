"""Quickly test the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import configuration
import ctc_ocr_model

from utils.build_dataset import _create_vocab

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",  "data/synth90k/tf_records/val-*",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoints/full_2_dropout",
                       "Path to the checkpoint directory.")
tf.flags.DEFINE_string("pause_time", 15,
                      "Pausing time for viewing result")
tf.flags.DEFINE_string("num_show_samples", 50,
                    "Number of showed examples for each time")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern

    checkpoint_dir = FLAGS.checkpoint_dir
    if not tf.gfile.IsDirectory(checkpoint_dir):
        tf.logging.info("Checkpoint directory does not exist: %s", checkpoint_dir)
        return

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = ctc_ocr_model.CtcOcrModel(model_config, mode='eval')
        model.build()

        # Set up the Saver for restoring model checkpoints.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Load the checkpoint
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, checkpoint)
            print("Model restored.")

            # Start queue runner
            tf.train.start_queue_runners(sess=sess)

            # Load the vocabulary for decoding
            vocab = _create_vocab()

            while True:
                # Fetching
                global_step, pred_chars, target_chars, total_loss = sess.run(
                    [model.global_step, model.pred_chars,
                     model.target_chars, model.total_loss])

                # Show result
                tf.logging.info("\tTotal loss: %.3f \n" % total_loss)

                for j in range(min(FLAGS.num_show_samples, model.config.batch_size)):
                    decoded_pred_str = vocab.ids_to_word(pred_chars[j, :])
                    decoded_target_str = vocab.ids_to_word(target_chars[j, :])
                    tf.logging.info("\ttarget: %s \t pred: %s" %
                                    (decoded_target_str, decoded_pred_str))

                time.sleep(FLAGS.pause_time)
                tf.logging.info("*" * 80)

if __name__ == "__main__":
    tf.app.run()
