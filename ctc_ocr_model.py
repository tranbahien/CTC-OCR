from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from ops import image_extractor
from ops import image_processing
from ops import inputs as input_ops
from ops import layer_norm

slim = tf.contrib.slim


class CtcOcrModel(object):

    def __init__(self, config, mode, train_vgg=True):
        """Basic setup.

        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_vgg: Whether the vgg submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_vgg = train_vgg

        # Reader for the input data
        self.reader = tf.TFRecordReader()

        # Initialize all variable with  a random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # A int32 sparse Tensor with shape [batch_size, seq_length].
        self.targets = None

        # A int32 sparse Tensor with shape [batch_size].
        self.seq_lengths = None

        # A float32 Tensor with shape [batch_size, seq_length, feature_size].
        self.image_features = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size].
        self.ctc_losses = None

        # Collection of variables from the vgg submodel.
        self.vgg_pretrained_variables = []

        # Function to restore the vgg submodel from checkpoint.
        self.init_fn = None

        # Function to restore the model from checkpoint for training.
        self.restore_fn = None

        # Global step Tensor.
        self.global_step = None

        # The Levenshtein distance error
        self.edit_distance = None

        # A int32 dense Tensor; the predicted characters
        self.pred_chars = None

        # A int32 dense Tensor; the target characters
        self.target_chars = None

        # A int32 dense Tensor; the logits inputting into the CTC
        self.logits = None

        # A int32 dense Tensor with shape [batch_size, seq_length]; The decoded
        # predicted chars
        self.decoded_pred_chars = None

        # The hash table for decode chars from ids
        self.vocab_table = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.
        Args:
            encoded_image: A scalar string Tensor; the encoded image.
            thread_id: Preprocessing thread id used to select the ordering of
            color distortions.
        Returns:
            A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              thread_id=thread_id,
                                              mode=self.mode,
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

            Outputs:
                self.images
                self.input_seqs
                self.targets (training and eval only)
        """
        if self.mode == "inference":
            with tf.name_scope("image_input"):
                "In inference mode, images are fed via placeholders."
                image_feed = tf.placeholder(
                    dtype=tf.string, shape=[], name="image_feed")

                # Process image and insert batch dimensions
                images = tf.expand_dims(self.process_image(image_feed), 0)

                # No target word in inference mode.
                targets = None

                # No sequence length in inference mode.
                seq_lengths = None

        else:
            with tf.name_scope("load_tf_records"):
                # Prefetch serialized SequenceExample protos.
                input_queue = input_ops.prefetch_input_data(
                    self.reader,
                    self.config.input_file_pattern,
                    is_training=self.is_training(),
                    batch_size=self.config.batch_size,
                    values_per_shard=self.config.values_per_input_shard,
                    input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                    num_reader_threads=self.config.num_input_reader_threads)

                assert self.config.num_preprocess_threads % 2 == 0
                images_and_words = []
                for thread_id in range(self.config.num_preprocess_threads):
                    serialized_sequence_example = input_queue.dequeue()
                    encoded_image, word = input_ops.parse_sequence_example(
                        serialized_sequence_example,
                        image_feature=self.config.image_feature_name,
                        word_feature=self.config.word_feature_name)
                    image = self.process_image(
                        encoded_image, thread_id=thread_id)
                    images_and_words.append([image, word])

                # Batch inputs.
                queue_capacity = (2 * self.config.num_preprocess_threads *
                                  self.config.batch_size)
                images, targets, seq_lengths = input_ops.make_batch(
                    images_and_words,
                    batch_size=self.config.batch_size,
                    queue_capacity=queue_capacity,
                    vocab_size=self.config.vocab_size)

        self.images = images
        self.targets = targets
        self.seq_lengths = seq_lengths

    def build_image_features(self):
        """Builds the image model subgraph and generates image features.

        Inputs:
            self.images

        Outputs:
            self.image_features
        """
        # Get image features created by CNN model
        vgg_output = image_extractor.vgg(self.images,
                                         trainable=self.train_vgg,
                                         is_training=self.is_training(),
                                         use_batch_norm=self.config.use_batch_norm)
        self.vgg_pretrained_variables = slim.get_variables_to_restore(
            include=["vgg/conv1", "vgg/conv2"])

        # Split image features into sequence of features for feeding into RNN
        with tf.variable_scope("seq_image_features") as scope:
            image_features = tf.unpack(vgg_output, axis=1,
                                       name="split_image_features")

        self.image_features = image_features

    def build_model(self):
        """Builds the model

        Inputs:
            self.image_features
            self.targets (training and eval only)
        """
        # Define the cell of RNN model
        if self.config.use_layer_norm:
            lstm_cell = layer_norm.LayerNormalizedLSTMCell(
                num_units=self.config.num_lstm_units)
        else:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self.config.num_lstm_units, state_is_tuple=True)

        if self.mode == "train":
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,
                output_keep_prob=self.config.lstm_dropout_keep_prob)

        # Build the RNN model
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_sc:
            # Stack RNN cell if using multi-layer RNN
            with tf.name_scope("multilayer_rnn_cell"):
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    cells=[lstm_cell] * self.config.num_lstm_layers,
                    state_is_tuple=True)

            # Create the Bidirectional-RNN
            rnn_outputs, _, _ = tf.nn.bidirectional_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.image_features,
                dtype=tf.float32)

        # Get the batch size
        batch_size = tf.shape(self.images)[0]

        with tf.variable_scope("logits") as logits_scope:
            packed_rnn_outputs = tf.pack(rnn_outputs,
                                         axis=1,
                                         name="packed_rnn_outputs")

            rnn_outputs_flat = tf.reshape(packed_rnn_outputs,
                                          [-1, self.config.num_lstm_units * 2],
                                          name="rnn_outputs_flat")

            W_logits = tf.Variable(
                tf.truncated_normal([self.config.num_lstm_units * 2,
                                     self.config.vocab_size + 1],
                                    name="W_logits"))
            b_logits = tf.Variable(
                tf.constant(0., shape=[self.config.vocab_size + 1]),
                name="b_logits")

            # Doing the affine projection
            logits = tf.matmul(rnn_outputs_flat, W_logits) + b_logits

            # Reshaping back to the original space
            shape = tf.shape(logits, name="get_logits_shape")
            logits = tf.reshape(logits, [batch_size, -1,
                                         self.config.vocab_size + 1],
                                name="reshaped_logits")
            self.logits = logits

            # Time major
            logits = tf.transpose(logits, (1, 0, 2), name="logits_time_major")

        # Get the length of sequence
        seq_lengths = tf.fill([batch_size], tf.shape(logits)[0],
                              name="seq_lengths")

        with tf.name_scope("ctc_decoder"):
            # Decode predicted chars
            if self.config.use_beam_search:
                pred_chars, log_prob = tf.nn.ctc_beam_search_decoder(
                    logits, seq_lengths, merge_repeated=False)
            else:
                pred_chars, log_prob = tf.nn.ctc_greedy_decoder(
                    logits, seq_lengths, merge_repeated=False)

            pred_chars = tf.cast(pred_chars[0], tf.int32)

        if self.mode != "inference":
            with tf.name_scope("ctc_loss"):
                # Calculate the CTC loss
                losses = tf.nn.ctc_loss(logits, self.targets, seq_lengths)
                batch_loss = tf.reduce_mean(losses)
                tf.contrib.losses.add_loss(batch_loss)
                total_loss = tf.contrib.losses.get_total_loss()

            with tf.name_scope("levenshtein_distance"):
                # Calculate the Levenshtein distance between pred_chars and
                # target_chars
                self.edit_distance = tf.reduce_mean(
                    tf.edit_distance(pred_chars, self.targets))

        with tf.name_scope("sparse_to_dense"):
            # Convert sparse tensors to dense tensor for tracking
            pred_chars = tf.sparse_to_dense(
                pred_chars.indices, pred_chars.shape, pred_chars.values)
            self.pred_chars = pred_chars

            if self.mode != "inference":
                target_chars = tf.sparse_to_dense(
                    self.targets.indices, self.targets.shape, self.targets.values)
                self.target_chars = target_chars

        with tf.name_scope("vocab_decoder"):
            with tf.device("/cpu:0"):
                preds = tf.to_int64(self.pred_chars)
                self.vocab_table = tf.contrib.lookup.HashTable(
                    tf.contrib.lookup.TextFileInitializer("vocab.txt", tf.int64, 0,
                                                          tf.string, 1,
                                                          delimiter=" "), "-")
                self.decoded_pred_chars = self.vocab_table.lookup(
                    preds, name="decoded_pred_chars")

        # Add summaries
        if self.mode != "inference":
            tf.summary.scalar("batch_loss", batch_loss)
            tf.summary.scalar("total_loss", total_loss)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.total_loss = total_loss
            self.ctc_losses = losses

    def setup_vgg_initializer(self):
        """Sets up the function to restore vgg variables from checkpoint."""
        if self.mode != "inference":
            # Restore vgg variables only.
            saver = tf.train.Saver(self.vgg_pretrained_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring VGG variables from checkpoint file %s",
                                self.config.vgg_checkpoint_file)
                saver.restore(sess, self.config.vgg_checkpoint_file)

            self.init_fn = restore_fn

    def setup_checkpoint_loader(self):
        """Sets up the function to restore CTC OCR model from checkpoint."""
        if self.mode == "train":
            saver = tf.train.Saver()

            def restore_fn(sess):
                tf.logging.info(
                    "Loading model from checkpoint: %s",
                    self.config.ctc_ocr_checkpoint_file)
                saver.restore(sess, self.config.ctc_ocr_checkpoint_file)
                tf.logging.info("Successfully loadded checkpoint: %s",
                                os.path.basename(self.config.ctc_ocr_checkpoint_file))

            self.restore_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_features()
        self.build_model()
        if self.config.vgg_checkpoint_file:
            self.setup_vgg_initializer()
        if self.config.ctc_ocr_checkpoint_file:
            self.setup_checkpoint_loader()
        self.setup_global_step()
