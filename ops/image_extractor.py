"""Image extractor ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim


def vgg(images,
        trainable=True,
        is_training=True,
        weight_decay=0.00004,
        stddev=0.1,
        dropout_keep_prob=0.5,
        use_batch_norm=True,
        batch_norm_params=None,
        add_summaries=True,
        scope="vgg"):
    """Builds an Oxford Net VGG subgraph for image feature extractor.

    Args:
      images: A float32 Tensor of shape [batch, height, width, channels].
      trainable: Whether the inception submodel should be trainable or not.
      is_training: Boolean indicating training mode or not.
      weight_decay: Coefficient for weight regularization.
      stddev: The standard deviation of the trunctated normal weight initializer.
      dropout_keep_prob: Dropout keep probability.
      use_batch_norm: Whether to use batch normalization.
      batch_norm_params: Parameters for batch normalization. See
        tf.contrib.layers.batch_norm for details.
      add_summaries: Whether to add activation summaries.
      scope: Optional Variable scope.

    Returns:
      end_points: A dictionary of activations from inception_v3 layers.
    """
    # Only consider the inception model to be in training mode if it's trainable.
    is_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    # TODO: Add regularizer
    # if trainable:
    #     weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    # else:
    #     weights_regularizer = None
    weights_regularizer = None


    with tf.variable_scope(scope, 'vgg', [images]) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer,
                            outputs_collections=end_points_collection,
                            trainable=trainable):

            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation_fn=tf.nn.relu):
                net = slim.conv2d(images, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2],  scope='pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 256, [3, 3],
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  scope='conv3')
                net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 1], stride=[2, 1],
                                      scope='pool3')
                net = slim.conv2d(net, 512, [3, 3],
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  scope='conv5')
                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.max_pool2d(net, [2, 1], stride=[2, 1],
                                      scope='pool4')
                net = tf.pad(net, paddings=[[0, 0], [0, 0], [1, 1], [0, 0]])
                net = slim.conv2d(net, 512, [2, 2],
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  padding="VALID",
                                  scope='conv7')
                with tf.variable_scope("feature_transpose"):
                    net = tf.squeeze(net, [1])

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

    # Add summaries.
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net
