import datasketches
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
from .metrics import get_histogram_sketch, calculate_percentiles

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Sketches:
    """
    A generic class to manage and update KLL sketches for custom-defined metrics.

    Supports both TensorFlow and PyTorch tensors for compatibility with different deep learning frameworks.
    """

    def __init__(self):
        """
        Initialize the Sketches class with an empty registry for custom metrics.
        """
        self.sketch_registry = {}

    def register_metric(self, metric_name, num_channels=1):
        """
        Register a custom metric by name. Each metric will have a corresponding KLL sketch.

        Args:
            metric_name (str): The name of the custom metric.
            num_channels (int): The number of channels for the metric (e.g., 3 for RGB images). Default is 1.
        """
        if num_channels == 1:
            self.sketch_registry[metric_name] = datasketches.kll_floats_sketch()
        else:
            self.sketch_registry[metric_name] = [datasketches.kll_floats_sketch() for _ in range(num_channels)]

    def update_kll_sketch(self, sketch, values):
        """
        Update a KLL sketch with values from a tensor.

        Args:
            sketch (datasketches.kll_floats_sketch): The KLL sketch to be updated.
            values (TensorFlow or PyTorch tensor): The tensor containing values to update the sketch with.
        """
        # Convert TensorFlow or PyTorch tensor to numpy array
        if isinstance(values, tf.Tensor):
            values = values.numpy()
        elif isinstance(values, torch.Tensor):
            values = values.cpu().numpy()

        # Squeeze the array to ensure it is 1D or scalar
        values = np.squeeze(values)

        try:
            # If values is a scalar, update the sketch directly
            if values.ndim == 0:
                sketch.update(values.item())
            else:
                # Flatten the values to 1D array before updating the sketch
                flattened_values = values.flatten()
                for value in flattened_values:
                    sketch.update(value)
        except Exception as e:
            print(f"Error updating sketch with values shape: {values.shape}, Error: {e}")

    def _calculate_channel_mean(self, values):
        """
        Calculate the mean for each channel in the image.

        Args:
            values (np.ndarray): The input image array with shape [height, width, channels].

        Returns:
            np.ndarray: An array containing the mean of each channel.
        """
        # Calculate the mean across the height and width dimensions, leaving only the channels
        return np.mean(values, axis=(0, 1))

    def update_sketches(self, **kwargs):
        """
        Update all registered KLL sketches in parallel using the provided metric values.

        Args:
            **kwargs: Keyword arguments where the key is the metric name and the value is the corresponding data tensor.
        """
        futures = []
        with ThreadPoolExecutor() as executor:
            for metric_name, values in kwargs.items():
                sketch = self.sketch_registry.get(metric_name)
                if sketch is None:
                    print(f"Warning: No sketch registered for metric '{metric_name}'")
                    continue

                if isinstance(sketch, list):
                    num_channels = len(sketch)
                    if values.ndim > 1:
                        for i in range(num_channels):
                            futures.append(executor.submit(self.update_kll_sketch, sketch[i], values[:, i]))
                    else:
                        print(f"Expected multi-channel data for '{metric_name}', but received scalar. Updating all channels with same value.")
                        for i in range(num_channels):
                            futures.append(executor.submit(self.update_kll_sketch, sketch[i], values))
                else:
                    futures.append(executor.submit(self.update_kll_sketch, sketch, values))

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise exceptions if any occurred during execution

    def tf_update_sketches(self, **kwargs):
        """
        TensorFlow-specific method to update KLL sketches using tf.py_function.

        Args:
            **kwargs: Keyword arguments where the key is the metric name and the value is the corresponding TensorFlow tensor.
        """
        # Use TensorFlow's py_function to call update_sketches
        tf.py_function(self.update_sketches, kwargs.values(), [])

    def pt_update_sketches(self, **kwargs):
        """
        PyTorch-specific method to update KLL sketches by directly calling update_sketches.

        Args:
            **kwargs: Keyword arguments where the key is the metric name and the value is the corresponding PyTorch tensor.
        """
        # Directly call update_sketches with PyTorch tensors
        self.update_sketches(**kwargs)

    def save_sketches(self, save_path):
        """
        Save all registered KLL sketches to binary files for later use.

        Args:
            save_path (str): Path to the directory where sketches will be saved.
        """
        os.makedirs(save_path, exist_ok=True)
        for metric_name, sketch in self.sketch_registry.items():
            if isinstance(sketch, list):
                for i, s in enumerate(sketch):
                    with open(os.path.join(save_path, f'{metric_name}_{i}.bin'), 'wb') as f:
                        f.write(s.serialize())
            else:
                with open(os.path.join(save_path, f'{metric_name}.bin'), 'wb') as f:
                    f.write(sketch.serialize())

    def load_sketches(self, load_path):
        """
        Load all KLL sketches from binary files.

        Args:
            load_path (str): Path to the directory from which sketches will be loaded.
        """
        for metric_name, sketch in self.sketch_registry.items():
            if isinstance(sketch, list):
                for i in range(len(sketch)):
                    with open(os.path.join(load_path, f'{metric_name}_{i}.bin'), 'rb') as f:
                        self.sketch_registry[metric_name][i] = datasketches.kll_floats_sketch.deserialize(f.read())
            else:
                with open(os.path.join(load_path, f'{metric_name}.bin'), 'rb') as f:
                    self.sketch_registry[metric_name] = datasketches.kll_floats_sketch.deserialize(f.read())

    def compute_thresholds(self, lower_percentile=0.1, upper_percentile=0.99):
        """
        Compute the lower and upper percentile thresholds for all registered KLL sketches.

        Args:
            lower_percentile (float): Lower percentile value (default is 0.1).
            upper_percentile (float): Upper percentile value (default is 0.99).

        Returns:
            dict: A dictionary containing the computed thresholds for all sketches.
        """
        thresholds = {}
        for metric_name, sketch in self.sketch_registry.items():
            if isinstance(sketch, list):
                thresholds[metric_name] = {}
                for idx, s in enumerate(sketch):
                    x, p = get_histogram_sketch(s)
                    lower_percentile_value, upper_percentile_value = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                    thresholds[metric_name][idx] = (lower_percentile_value, upper_percentile_value)
            else:
                x, p = get_histogram_sketch(sketch)
                lower_percentile_value, upper_percentile_value = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                thresholds[metric_name] = (lower_percentile_value, upper_percentile_value)
        return thresholds
