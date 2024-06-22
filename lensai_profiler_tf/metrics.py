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

    snr = tf.where(noise == 0, np.inf, 20 * tf.math.log(signal / noise + 1e-7) / tf.math.log(10.0))
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

def calculate_percentiles(x, probabilities, lower_percentile=0.01, upper_percentile=0.99):
    """
    Calculates percentiles from a PMF (Probability Mass Function) represented as two separate lists.

    Args:
        x: List containing the x-values (possible values) in the distribution.
        probabilities: List containing the probabilities corresponding to the x-values.
        lower_percentile: Float between 0 and 1 (inclusive) specifying the lower percentile (default 0.01).
        upper_percentile: Float between 0 and 1 (inclusive) specifying the upper percentile (default 0.99).

    Returns:
        A tuple containing the lower and upper percentiles (x-values, float).
    """

    # Ensure lists have the same length
    if len(x) != len(probabilities):
        raise ValueError("x and probabilities lists must have the same length")

    # Ensure PMF is a valid probability distribution (sums to 1)
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("PMF must sum to 1")

    # Combine x-values and probabilities into a single list of tuples
    pmf = list(zip(x, probabilities))
    # Sort PMF based on x-values (ascending order)
    pmf.sort(key=lambda item: item[0])
    # Calculate cumulative sum of probabilities
    cdf = np.cumsum([p for _, p in pmf])
    # Calculate percentile indices with edge case handling
    lower_percentile_idx = np.searchsorted(cdf, lower_percentile, side='right')
    upper_percentile_idx = np.searchsorted(cdf, upper_percentile, side='right')
    # Access corresponding x-values from the sorted PMF
    lower_percentile_value = pmf[lower_percentile_idx][0] if lower_percentile_idx < len(pmf) else pmf[-1][0]
    upper_percentile_value = pmf[upper_percentile_idx][0] if upper_percentile_idx < len(pmf) else pmf[-1][0]
    return lower_percentile_value, upper_percentile_value


def get_histogram_sketch(sketch, num_splits=30):
    """
    Reads a binary file, deserializes the content, and extracts the PMF.

    Args:
        filename: Path to the binary file.
        num_splits: Number of splits for the PMF (default: 30).

    Returns:
        A tuple containing x-axis values and the PMF.
    """
    if sketch.is_empty():
        return None,None
    xmin = sketch.get_min_value()
    try:
        step = (sketch.get_max_value() - xmin) / num_splits
    except ZeroDivisionError:
        print(f"Error: num_splits should be non-zero for file {filename}")
        return None, None
    if step == 0:
        step = 0.01

    splits = [xmin + (i * step) for i in range(0, num_splits)]
    pmf = sketch.get_pmf(splits)
    x = splits + [sketch.get_max_value()]  # Append max value for x-axis

    return x, pmf
