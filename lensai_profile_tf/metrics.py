# lensai/metrics.py

import tensorflow as tf
import numpy as np

def calculate_brightness(image):
    grayscale = tf.image.rgb_to_grayscale(image)
    return tf.reduce_mean(grayscale)

def calculate_snr(image_tensor):
    if image_tensor.shape[-1] == 3:
        grayscale = tf.image.rgb_to_grayscale(image_tensor)
    elif image_tensor.shape[-1] == 4:
        grayscale = tf.image.rgb_to_grayscale(image_tensor[..., :3])
    else:
        grayscale = image_tensor

    mean, variance = tf.nn.moments(grayscale, axes=[0, 1])
    sigma = tf.sqrt(variance)

    signal = mean
    noise = sigma

    snr = tf.where(noise == 0, np.inf, 20 * tf.math.log(signal / noise) / tf.math.log(10.0))
    return snr

def calculate_channel_histogram(image):
    num_channels = image.shape[-1]
    channel_pixels = tf.reshape(image, [-1, num_channels])
    return channel_pixels


def calculate_sharpness_laplacian(image):
    kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    grayscale = tf.image.rgb_to_grayscale(image)
    grayscale = tf.expand_dims(grayscale, axis=0)
    sharpness = tf.nn.conv2d(grayscale, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.reduce_mean(tf.abs(sharpness))

def calculate_channel_mean(image):
    return tf.reduce_mean(image, axis=[0, 1])

def process_batch(images):
    brightness = tf.map_fn(calculate_brightness, images, dtype=tf.float32)
    sharpness = tf.map_fn(calculate_sharpness_laplacian, images, dtype=tf.float32)
    channel_mean = tf.map_fn(calculate_channel_mean, images, dtype=tf.float32)
    snr = tf.map_fn(calculate_snr, images, dtype=tf.float32)
    channel_pixels = tf.map_fn(calculate_channel_histogram, images, dtype=tf.float32)

    return brightness, sharpness, channel_mean, snr, channel_pixels
