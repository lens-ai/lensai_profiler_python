import os
import numpy as np
import datasketches
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(self, num_channels, metrics):
        """
        Initialize the Sketches class.

        Args:
            num_channels: Number of channels in the images (e.g., 3 for RGB).
            metrics: A dictionary where keys are metric names and values are metric functions.
        """
        self.kll_sketches = {
            metric_name: [datasketches.kll_floats_sketch() for _ in range(num_channels)]
            for metric_name in metrics.keys()
        }
        self.metrics = metrics

    def update_kll_sketch(self, sketch, values):
        """
        Update the KLL sketch with values.

        Args:
            sketch: The KLL sketch to update.
            values: The values to update the sketch with.
        """
        if TENSORFLOW_AVAILABLE and isinstance(values, tf.Tensor):
            values = values.numpy()  # Convert TensorFlow tensor to numpy array
        elif TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            values = values.cpu().numpy()  # Convert PyTorch tensor to numpy array

        values = np.squeeze(values)  # Ensure the values are 1D
        if len(values.shape) == 0:
            return  # Ignore scalar values

        try:
            for value in values:
                sketch.update(value)
        except Exception as e:
            print(f"Error updating sketch with values: {values}, Error: {e}")

    def update_sketches(self, **metric_results):
        """
        Update all the sketches using the provided metric results.

        Args:
            metric_results: A dictionary containing metric results to be logged.
        """
        futures = []
        with ThreadPoolExecutor() as executor:
            for metric_name, results in metric_results.items():
                for i, result in enumerate(results):
                    futures.append(executor.submit(self.update_kll_sketch, self.kll_sketches[metric_name][i], result))

            for future in as_completed(futures):
                future.result()  # Will raise exceptions if any occurred during execution

    def tf_update_sketches(self, **metric_results):
        """
        TensorFlow wrapper to update sketches using TensorFlow tensors.

        Args:
            metric_results: A dictionary containing metric results to be logged.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TensorFlow to use this feature.")
        tf.py_function(self.update_sketches, [], metric_results)

    def save_sketches(self, save_path):
        """
        Save the sketches to a file.

        Args:
            save_path: The path where the sketches should be saved.
        """
        os.makedirs(save_path, exist_ok=True)
        for metric_name, sketches in self.kll_sketches.items():
            for i, sketch in enumerate(sketches):
                with open(os.path.join(save_path, f'{metric_name}_{i}.bin'), 'wb') as f:
                    f.write(sketch.serialize())

    def load_sketches(self, load_path):
        """
        Load the sketches from a file.

        Args:
            load_path: The path where the sketches are stored.
        """
        for metric_name, sketches in self.kll_sketches.items():
            for i in range(len(sketches)):
                with open(os.path.join(load_path, f'{metric_name}_{i}.bin'), 'rb') as f:
                    self.kll_sketches[metric_name][i] = datasketches.kll_floats_sketch.deserialize(f.read())

    def compute_thresholds(self, lower_percentile=0.1, upper_percentile=0.99):
        """
        Compute thresholds based on percentiles.

        Args:
            lower_percentile: Lower percentile.
            upper_percentile: Upper percentile.

        Returns:
            A dictionary containing the thresholds for each metric.
        """
        thresholds = {}
        for metric_name, sketches in self.kll_sketches.items():
            thresholds[metric_name] = {}
            for idx, sketch in enumerate(sketches):
                x, p = get_histogram_sketch(sketch)
                lower_value, upper_value = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                thresholds[metric_name][idx] = (lower_value, upper_value)
        return thresholds

