"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def process_image(encoded_image,
                  thread_id,
                  is_training,
                  mode,
                  height=32, width=100,
                  image_format="jpeg"):
    """Decode an image, resize and apply random distortions.

    In training, images are distorted slightly differently depending on thread_id.

    Args:
        encoded_image: String Tensor containing the image.
        is_training: Boolean; whether preprocessing for training or eval.
        mode: "train", "eval" or "inference".
        height: Height of the output image.
        width: Width of the output image.
        thread_id: Preprocessing thread id used to select the ordering of color
          distortions. There should be a multiple of 2 preprocessing threads.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid.
    """

    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def image_summary(name, image):
        if not thread_id:
            tf.summary.image(name, tf.expand_dims(image, 0))

    # Decode image into a float32 tensor of shape [?, ?, 3] with values in [0, 1).
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == "jpeg":
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_summary("original_image", image)

    if mode == "inference":
        # Convert to grayscale
        gray_image = tf.image.rgb_to_grayscale(image, name="convert_gray_scale")
        image = tf.concat(2, [gray_image, gray_image, gray_image])

    # Resize image
    image = tf.image.resize_images(image,
                                   size=[height, width],
                                   method=tf.image.ResizeMethod.BILINEAR)

    image_summary("final_image", image)

    # Rescale to [-1, 1] instead of [0, 1]
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    return image
