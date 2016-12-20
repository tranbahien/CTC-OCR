"""Generate transcription for images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import cv2

import tensorflow as tf

from autocorrect import spell

import configuration
import inference_wrapper

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "checkpoints/full_2_dropout",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "test_images/*.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        infer = inference_wrapper.InferenceWrapper()
        restore_fn = infer.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)

    g.finalize()

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
              len(filenames), FLAGS.input_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Initialize the vocabulary lookup table
        infer.model.vocab_table.init.run(session=sess)

        filenames.sort()
        # Predict
        for filename in filenames:
            with tf.gfile.GFile(filename, "r") as f:
                # Predict transcription
                tic = time.time()
                image = f.read()
                pred_chars = infer.inference_step(sess, image)[0][0]
                pred_word = "".join([item for item in pred_chars])
                auto_correct_word = spell(pred_word)
                toc = time.time()

                # Print out the result
                print("Prediction for image %s in %.3f ms" %
                      (os.path.basename(filename), (toc - tic) * 1000))
                print("predicted word: %s" % pred_word)
                print("auto correct word: %s" % auto_correct_word)
                print("*" * 50)

                # Show image
                cv_img = cv2.imread(filename)
                cv2.imshow('image', cv_img)
                k = cv2.waitKey(0)
                if k == ord('n'):
                    cv2.destroyAllWindows()

if __name__ == "__main__":
  tf.app.run()
