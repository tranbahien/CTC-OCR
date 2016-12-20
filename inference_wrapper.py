"""Model wrapper class for performing inference with a CtcOcrModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

import ctc_ocr_model


class InferenceWrapper(object):
    """Model wapper class for performing inference with a CtcOcrModel."""

    def __init__(self):
        self.model = None

    def build_model(self, model_config):
        """Builds the model for inference.

        Args:
            model_config: Object containing configuration for building the model.

        Returns:
            model: The model object.
        """
        model = ctc_ocr_model.CtcOcrModel(model_config, mode="inference")
        model.build()
        self.model = model

        return model

    def inference_step(self, sess, encoded_image):
        """Runs one step of inference.

        Args:
            sess: TensorFlow Session object.
            encoded_image: An encoded image string.

        Returns:
            decoded_pred_chars: A numpy array of shape [batch_size, seq_length].
        """
        decoded_pred_chars = sess.run(
            fetches=["vocab_decoder/decoded_pred_chars:0"],
            feed_dict={
                "image_input/image_feed:0": encoded_image
            })

        return decoded_pred_chars

    def _create_restore_fn(self, checkpoint_path, saver):
        """Creates a function that restores a model from checkpoint.

        Args:
            checkpoint_path: Checkpoint file or a directory containing a
                checkpoint file.
            saver: Saver for restoring variables from the checkpoint file.

        Returns:
            restore_fn: A function such that restore_fn(sess) loads model
                variables from the checkpoint file.

        Raises:
            ValueError: If checkpoint_path does not refer to a checkpoint
                file or a directory containing a checkpoint file.
        """
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s"
                                 % checkpoint_path)

        def _restore_fn(sess):
            tf.logging.info(
                "Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info("Successfully loadded checkpoint: %s",
                            os.path.basename(checkpoint_path))

        return _restore_fn

    def build_graph_from_config(self, model_config, checkpoint_path):
        """Builds the inference graph from a configuration object.

        Args:
            model_config: Object containing configuration for building the model
            checkpoint_path: Checkpoint file or a directory containing a
                checkpoint file

        Returns:
            restore_fn: A function such that restore_fn(sess) loads model
                variables from the checkpoint file.
        """
        tf.logging.info("Building model.")
        self.build_model(model_config)
        saver = tf.train.Saver()

        return self._create_restore_fn(checkpoint_path, saver)

    def build_graph_from_proto(self, graph_def_file, saver_def_file,
                               checkpoint_path):
        """Builds the inference graph from serialized GraphDef and SaverDef protos.

        Args:
            graph_def_file: File containing a serialized GraphDef proto
            saver_def_file: File containing a serialized SaverDef proto
            checkpoint_path: Checkpoint file or a directory containing a
                checkpoint file

        Returns:
            restore_fn: A function such that restore_fn(sess) loads model
                variables from the checkpoint file.
        """
        # Load the Graph.
        tf.logging.info("Loading GraphDef from file: %s" % graph_def_file)
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_def_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        # Load the Saver.
        tf.logging.info("Loading SaverDef from file: %s", saver_def_file)
        saver_def = tf.train.SaverDef()
        with tf.gfile.FastGFile(saver_def_file, "rb") as f:
            saver_def.ParseFromString(f.read())
        saver = tf.train.Saver(saver_def=saver_def)

        return self._create_restore_fn(checkpoint_path, saver)
