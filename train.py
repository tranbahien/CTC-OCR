"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
import ctc_ocr_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "data/synth90k/tf_records/train-*",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("vgg_checkpoint_file",
                       "pretrained_models/vgg_16/vgg_16.ckpt",
                       "Path to a pretrained vgg model.")
tf.flags.DEFINE_string("ctc_ocr_checkpoint_file",
                       "checkpoints/full_2_dropout/model.ckpt-1377757",
                       "Path to a pretrained CTC OCR model.")
tf.flags.DEFINE_string("train_dir",
                       "checkpoints/full_22",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_vgg", True,
                        "Whether to train vgg submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 20000000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    if FLAGS.vgg_checkpoint_file != "":
        model_config.vgg_checkpoint_file = FLAGS.vgg_checkpoint_file
    if FLAGS.ctc_ocr_checkpoint_file != "":
        model_config.ctc_ocr_checkpoint_file = FLAGS.ctc_ocr_checkpoint_file

    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)
    else:
        tf.logging.info("Removing the old training directory: %s", train_dir)
        tf.gfile.DeleteRecursively(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = ctc_ocr_model.CtcOcrModel(model_config, mode='train')
        model.build()

        # Set up the learning rate.
        learning_rate_decay_fn = None

        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(
            max_to_keep=training_config.max_checkpoints_to_keep)

        # Create the restore the checkpoints
        init_fn = None
        if model.restore_fn:
            init_fn = model.restore_fn
        elif model.init_fn:
            init_fn = model.init_fn

        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            train_dir,
            log_every_n_steps=FLAGS.log_every_n_steps,
            graph=g,
            global_step=model.global_step,
            number_of_steps=FLAGS.number_of_steps,
            init_fn=init_fn,
            save_summaries_secs=20,
            saver=saver)

if __name__ == "__main__":
    tf.app.run()
