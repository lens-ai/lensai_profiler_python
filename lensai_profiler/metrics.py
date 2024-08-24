# lensai/metrics.py

import numpy as np
import datasketches

class Metrics:
    def __init__(self, framework='tf'):
        """
        Initialize the Metrics class.

        Args:
            framework: Specify the framework ('tf' for TensorFlow, 'pt' for PyTorch).
        """
        self.framework = framework

        if self.framework == 'tf':
            global tf
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError("TensorFlow is not installed. Please install it to use TensorFlow metrics.")
        elif self.framework == 'pt':
            global torch
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch is not installed. Please install it to use PyTorch metrics.")
        else:
            raise ValueError("Unsupported framework. Use 'tf' for TensorFlow or 'pt' for PyTorch.")

    def calculate_brightness(self, image):
        """
        Calculate the brightness of an image.

        Args:
            image: A TensorFlow tensor representing an RGB image.

        Returns:
            A TensorFlow tensor containing the mean brightness of the image.
        """
        if self.framework == 'tf':
            return self._calculate_brightness_tf(image)
        elif self.framework == 'pt':
            return self._calculate_brightness_pt(image)

    def calculate_sharpness_laplacian(self, image):
        """
        Calculate the sharpness of an image using the Laplacian operator.

        Args:
            image: A TensorFlow tensor representing an RGB image.

        Returns:
            A TensorFlow tensor containing the sharpness of the image.
        """
        if self.framework == 'tf':
            return self._calculate_sharpness_laplacian_tf(image)
        elif self.framework == 'pt':
            return self._calculate_sharpness_laplacian_pt(image)

    def calculate_channel_mean(self, image):
        """
        Calculate the mean of each channel of an image.

        Args:
            image: A TensorFlow tensor representing an image.

        Returns:
            A TensorFlow tensor containing the mean of each channel.
        """
        if self.framework == 'tf':
            return self._calculate_channel_mean_tf(image)
        elif self.framework == 'pt':
            return self._calculate_channel_mean_pt(image)

    def calculate_snr(self, image):
        """
        Calculate the Signal-to-Noise Ratio (SNR) of an image.

        Args:
            image_tensor: A TensorFlow tensor representing an RGB or RGBA image.

        Returns:
            A TensorFlow tensor containing the SNR of the image.
        """
        if self.framework == 'tf':
            return self._calculate_snr_tf(image)
        elif self.framework == 'pt':
            return self._calculate_snr_pt(image)

    def calculate_channel_histogram(self, image):
        """
        Calculate the histogram of the channels of an image.

        Args:
            image: A TensorFlow tensor representing an image.

        Returns:
            A TensorFlow tensor containing the histogram of the image channels.
        """
        if self.framework == 'tf':
            return self._calculate_channel_histogram_tf(image)
        elif self.framework == 'pt':
            return self._calculate_channel_histogram_pt(image)

    def process_batch(self, images):
        """
        Process a batch of images and calculate various metrics.

        Args:
            images: A TensorFlow tensor representing a batch of images.

        Returns:
            A tuple containing the brightness, sharpness, channel mean, SNR, and channel histogram of the batch.
        """
        if self.framework == 'tf':
            return self._process_batch_tf(images)
        elif self.framework == 'pt':
            return self._process_batch_pt(images)

    # TensorFlow implementations
    def _calculate_brightness_tf(self, image):
        grayscale = tf.image.rgb_to_grayscale(image)
        return tf.reduce_mean(grayscale)

    def _calculate_sharpness_laplacian_tf(self, image):
        kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        grayscale = tf.image.rgb_to_grayscale(image)
        grayscale = tf.expand_dims(grayscale, axis=0)
        sharpness = tf.nn.conv2d(grayscale, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.reduce_mean(tf.abs(sharpness))

    def _calculate_channel_mean_tf(self, image):
        return tf.reduce_mean(image, axis=[0, 1])

    def _calculate_snr_tf(self, image_tensor):
        grayscale = tf.image.rgb_to_grayscale(image_tensor)
        mean, variance = tf.nn.moments(grayscale, axes=[0, 1])
        sigma = tf.sqrt(variance)
        snr = tf.where(sigma == 0, np.inf, 20 * tf.math.log(mean / sigma + 1e-7) / tf.math.log(10.0))
        return snr

    def _calculate_channel_histogram_tf(self, image, pixel_range=[0, 255], bins=256):
        num_channels = image.shape[-1]
        channel_pixels = tf.reshape(image, [-1, num_channels])
        histograms = []
        for i in range(num_channels):
            hist = tf.histogram_fixed_width(channel_pixels[:, i], value_range=pixel_range, nbins=bins)
            histograms.append(hist)
        return tf.stack(tf.cast(histograms, tf.float32), axis=0)

    def _calculate_channel_histogram_pt(self, image, pixel_range=[0, 255], bins=256):
        num_channels = image.shape[0]
        channel_pixels = image.view(num_channels, -1)
        histograms = []
        for i in range(num_channels):
            hist = torch.histc(channel_pixels[i], bins=bins, min=pixel_range[0], max=pixel_range[1])
            histograms.append(hist)
        return torch.stack(histograms, dim=0)

    def _process_batch_tf(self, images):
        brightness = tf.map_fn(self._calculate_brightness_tf, images, dtype=tf.float32)
        sharpness = tf.map_fn(self._calculate_sharpness_laplacian_tf, images, dtype=tf.float32)
        channel_mean = tf.map_fn(self._calculate_channel_mean_tf, images, dtype=tf.float32)
        snr = tf.map_fn(self._calculate_snr_tf, images, dtype=tf.float32)
        channel_pixels = tf.map_fn(self._calculate_channel_histogram_tf, images, dtype=tf.float32)
        return brightness, sharpness, channel_mean, snr, channel_pixels


    # PyTorch implementations
    def _calculate_brightness_pt(self, image):
        grayscale = torch.mean(image, dim=0, keepdim=True)
        return torch.mean(grayscale)
    
    def _calculate_sharpness_laplacian_pt(self, image):
        # Define the Laplacian kernel
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)  # Reshape to (out_channels, in_channels, height, width)
    
        # Convert the image to grayscale if it has more than one channel
        if image.size(1) != 1:  # Check if the image has more than one channel
            grayscale = torch.mean(image, dim=0, keepdim=True)  # Convert to grayscale
        else:
            grayscale = image  # Image is already grayscale
        
        grayscale = grayscale.unsqueeze(0)  # Add a batch dimension if necessary, shape becomes [1, 1, H, W]

        #Apply the Laplacian kernel using conv2d
        sharpness = torch.nn.functional.conv2d(grayscale, kernel, stride=1, padding=1)
        return torch.mean(torch.abs(sharpness))  # Calculate mean absolute sharpness value
                
    def _calculate_channel_mean_pt(self, image):
        return torch.mean(image, dim=[1, 2])

    def _calculate_snr_pt(self, image_tensor):
        grayscale = torch.mean(image_tensor, dim=0, keepdim=True)
        mean = torch.mean(grayscale)
        sigma = torch.std(grayscale)
        snr = torch.where(sigma == 0, torch.tensor(float('inf')), 20 * torch.log10(mean / (sigma + 1e-7)))
        return snr


    def _process_batch_pt(self, images):
        brightness = torch.stack([self._calculate_brightness_pt(img) for img in images])
        sharpness = torch.stack([self._calculate_sharpness_laplacian_pt(img) for img in images])
        channel_mean = torch.stack([self._calculate_channel_mean_pt(img) for img in images])
        snr = torch.stack([self._calculate_snr_pt(img) for img in images])
        channel_pixels = torch.stack([self._calculate_channel_histogram_pt(img) for img in images])
        return brightness, sharpness, channel_mean, snr, channel_pixels

# Utility functions for both TensorFlow and PyTorch
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
        sketch: A probabilistic data structure representing the sketch of the distribution.
        num_splits: Number of splits for the PMF (default: 30).

    Returns:
        A tuple containing x-axis values and the PMF.
    """
    if sketch.is_empty():
        return None, None
    xmin = sketch.get_min_value()
    try:
        step = (sketch.get_max_value() - xmin) / num_splits
    except ZeroDivisionError:
        print(f"Error: num_splits should be non-zero")
        return None, None
    if step == 0:
        step = 0.01

    splits = [xmin + (i * step) for i in range(0, num_splits)]
    pmf = sketch.get_pmf(splits)
    x = splits + [sketch.get_max_value()]  # Append max value for x-axis

    return x, pmf

