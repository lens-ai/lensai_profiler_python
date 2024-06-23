import datasketches
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import tensorflow as tf
from .metrics import get_histogram_sketch, calculate_percentiles


class Sketches:
    """
    A class to represent and manage multiple KLL sketches for various metrics.

    Attributes:
        kll_brightness: KLL sketch for brightness values.
        kll_sharpness: KLL sketch for sharpness values.
        kll_snr: KLL sketch for signal-to-noise ratio values.
        kll_channel_mean: List of KLL sketches for channel mean values.
        kll_pixel_distribution: List of KLL sketches for pixel distribution values.
    """

    def __init__(self, num_channels):
        """
        Initialize the Sketches class with the specified number of channels.

        Args:
            num_channels: An integer representing the number of channels.
        """
        self.kll_brightness = datasketches.kll_floats_sketch()
        self.kll_sharpness = datasketches.kll_floats_sketch()
        self.kll_snr = datasketches.kll_floats_sketch()
        self.kll_channel_mean = [datasketches.kll_floats_sketch() for _ in range(num_channels)]
        self.kll_pixel_distribution = [datasketches.kll_floats_sketch() for _ in range(num_channels)]

    def update_kll_sketch(self, sketch, values):
        """
        Update a given KLL sketch with values.

        Args:
            sketch: A KLL sketch to update.
            values: A TensorFlow tensor of values to add to the sketch.
        """
        values = values.numpy()  # Convert tensor to numpy array
        values = np.squeeze(values)  # Ensure the values are 1D
        if len(values.shape) == 0:
            return  # Ignore scalar values
        try:
            for value in values:
                sketch.update(value)
        except:
            print("Error updating sketch with values:", values)

    def update_sketches(self, brightness, sharpness, channel_mean, snr, channel_pixels):
        """
        Update all KLL sketches with given metric values.

        Args:
            brightness: A TensorFlow tensor of brightness values.
            sharpness: A TensorFlow tensor of sharpness values.
            channel_mean: A TensorFlow tensor of channel mean values.
            snr: A TensorFlow tensor of signal-to-noise ratio values.
            channel_pixels: A TensorFlow tensor of pixel distribution values.
        """
        futures = []
        with ThreadPoolExecutor() as executor:
            futures.append(executor.submit(self.update_kll_sketch, self.kll_brightness, brightness))
            futures.append(executor.submit(self.update_kll_sketch, self.kll_sharpness, sharpness))
            futures.append(executor.submit(self.update_kll_sketch, self.kll_snr, snr))
            num_channels = channel_mean.shape[-1]
            for i in range(num_channels):
                futures.append(executor.submit(self.update_kll_sketch, self.kll_channel_mean[i], channel_mean[:, i]))
                futures.append(executor.submit(self.update_kll_sketch, self.kll_pixel_distribution[i], channel_pixels[:, i]))
            
            for future in as_completed(futures):
                future.result()  # Will raise exceptions if any occurred during execution

    def tf_update_sketches(self, brightness, sharpness, channel_mean, snr, channel_pixels):
        """
        Update sketches using TensorFlow py_function for compatibility.

        Args:
            brightness: A TensorFlow tensor of brightness values.
            sharpness: A TensorFlow tensor of sharpness values.
            channel_mean: A TensorFlow tensor of channel mean values.
            snr: A TensorFlow tensor of signal-to-noise ratio values.
            channel_pixels: A TensorFlow tensor of pixel distribution values.
        """
        tf.py_function(self.update_sketches, [brightness, sharpness, channel_mean, snr, channel_pixels], [])

    def save_sketches(self, save_path):
        """
        Save all KLL sketches to binary files.

        Args:
            save_path: A string representing the path to save the sketches.
        """
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'kll_brightness.bin'), 'wb') as f:
            f.write(self.kll_brightness.serialize())
        with open(os.path.join(save_path, 'kll_sharpness.bin'), 'wb') as f:
            f.write(self.kll_sharpness.serialize())
        with open(os.path.join(save_path, 'kll_snr.bin'), 'wb') as f:
            f.write(self.kll_snr.serialize())
        for i, sketch in enumerate(self.kll_channel_mean):
            with open(os.path.join(save_path, f'kll_channel_mean_{i}.bin'), 'wb') as f:
                f.write(sketch.serialize())
        for i, sketch in enumerate(self.kll_pixel_distribution):
            with open(os.path.join(save_path, f'kll_channel_pixels_{i}.bin'), 'wb') as f:
                f.write(sketch.serialize())

    def load_sketches(self, load_path):
        """
        Load all KLL sketches from binary files.

        Args:
            load_path: A string representing the path to load the sketches from.
        """
        with open(os.path.join(load_path, 'kll_brightness.bin'), 'rb') as f:
            self.kll_brightness = datasketches.kll_floats_sketch.deserialize(f.read())
        with open(os.path.join(load_path, 'kll_sharpness.bin'), 'rb') as f:
            self.kll_sharpness = datasketches.kll_floats_sketch.deserialize(f.read())
        with open(os.path.join(load_path, 'kll_snr.bin'), 'rb') as f:
            self.kll_snr = datasketches.kll_floats_sketch.deserialize(f.read())
        for i in range(len(self.kll_channel_mean)):
            with open(os.path.join(load_path, f'kll_channel_mean_{i}.bin'), 'rb') as f:
                self.kll_channel_mean[i] = datasketches.kll_floats_sketch.deserialize(f.read())
        for i in range(len(self.kll_pixel_distribution)):
            with open(os.path.join(load_path, f'kll_channel_pixels_{i}.bin'), 'rb') as f:
                self.kll_pixel_distribution[i] = datasketches.kll_floats_sketch.deserialize(f.read())

    def compute_thresholds(self, lower_percentile=0.1, upper_percentile=0.99):
        """
        Compute the lower and upper percentile thresholds for all sketches.

        Args:
            lower_percentile: A float representing the lower percentile.
            upper_percentile: A float representing the upper percentile.

        Returns:
            A dictionary containing the lower and upper percentile thresholds for all sketches.
        """
        thresholds = {}
        for attr in ['kll_brightness', 'kll_sharpness', 'kll_snr', 'kll_channel_mean', 'kll_pixel_distribution']:
            value = getattr(self, attr)
            if isinstance(value, list):
                thresholds[attr] = {}
                for idx, sketch in enumerate(value):
                    x, p = get_histogram_sketch(sketch)
                    lower_percentile_value, upper_percentile_value = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                    thresholds[attr][idx] = (lower_percentile_value, upper_percentile_value)
            else:
                x, p = get_histogram_sketch(value)
                lower_percentile_value, upper_percentile_value = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                thresholds[attr] = (lower_percentile_value, upper_percentile_value)
        return thresholds
