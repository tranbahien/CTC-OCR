"""CTC OCR model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        # Image format ("jpeg" or "png").
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer words.
        self.word_feature_name = "image/word_ids"

        # Number of unique chars in the vocab
        self.vocab_size = 37

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Batch size.
        self.batch_size = 64

        # File containing an VGG checkpoint to initialize the variables
        # of the VGG model. Must be provided when starting training for the
        # first time.
        self.vgg_checkpoint_file = None

        # File containing an trained CTC OCR checkpoint to initialize the
        # variables of model.
        self.ctc_ocr_checkpoint_file = None

        # Dimensions of VGG input images.
        self.image_height = 32
        self.image_width = 100

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # Number of layers of LSTM
        self.num_lstm_layers = 2

        # Number of units of LSTM cell
        self.num_lstm_units = 256

        # Use batch normalization for CNN
        self.use_batch_norm = True

        # Use batch normalization for LSTM
        self.use_layer_norm = False

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        # Use beam search for decoding ctc
        self.use_beam_search = True


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 7760308

        # Optimizer for training the model.
        self.optimizer = "Adam"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.0004
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 0.8

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_vgg_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
