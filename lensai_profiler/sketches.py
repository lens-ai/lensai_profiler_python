import datasketches
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import numpy as np
from .metrics import get_histogram_sketch, calculate_percentiles
from .utils import tar_and_gzip_folder, post_file_to_endpoint

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

    def register_metric(self, metric_name, n=1, l=[]):
        """
        Register a custom metric by name. Each metric will have a corresponding KLL sketch.

        Args:
            metric_name (str): The name of the custom metric.
            num_channels (int): The number of channels for the metric (e.g., 3 for RGB images). Default is 1.
        """
        if l :
            self.cls = l
            self.sketch_registry[metric_name] = [datasketches.kll_floats_sketch() for _ in l]
        elif n == 1:
            self.sketch_registry[metric_name] = datasketches.kll_floats_sketch()
        elif n > 1:
            self.sketch_registry[metric_name] = [datasketches.kll_floats_sketch() for _ in range(n)]

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

    def get_cls_index(self, cls):
        return self.cls.index(cls)

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
                    if isinstance(values, tuple):
                        cls, cls_values = values
                        cls_index = get_cls_index(cls)
                        futures.append(executor.submit(self.update_kll_sketch, sketch[cls_index], cls_values))
                    else:
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
    
    def save_sketches(self, base_path):
        """
        Save all registered KLL sketches to binary files for later use.

        Args:
            base_path (str): Path to the base directory where sketches will be saved.
        """
        # Loop through the registered sketches
        for metric_name, sketch in self.sketch_registry.items():
            # Determine the subdirectory based on the metric name
            if metric_name in ['brightness', 'noise', 'sharpness'] or metric_name.startswith(('mean_', 'pixel_')):
                subfolder = 'imgstats'
            elif metric_name == 'embeddings':
                subfolder = 'modelstats'
            else:
                # For custom sketches, use the 'customstats' subfolder
                subfolder = 'customstats'

            # Construct the full path to save the sketches
            save_path = os.path.join(base_path, subfolder)
            os.makedirs(save_path, exist_ok=True)

            # Save the sketches
            if isinstance(sketch, list):
                for i, s in enumerate(sketch):
                    file_path = os.path.join(save_path, f'{metric_name}_{i}.bin')
                    with open(file_path, 'wb') as f:
                        f.write(s.serialize())
            else:
                file_path = os.path.join(save_path, f'{metric_name}.bin')
                with open(file_path, 'wb') as f:
                    f.write(sketch.serialize())

    def load_sketches(self, base_path):
        """
        Load all KLL sketches from binary files.

        Args:
            base_path (str): Path to the base directory from which sketches will be loaded.
        """
        def load_from_folder(folder_path):
            """
            Load sketches from a specific folder into the sketch registry.

            Args:
                folder_path (str): Path to the folder containing the binary files.
            """
            if not os.path.exists(folder_path):
                return
        
            for filename in os.listdir(folder_path):
                if filename.endswith('.bin'):
                    metric_name, index_part = _parse_filename(filename)
                    if index_part is not None:
                        _load_indexed_sketch(folder_path, filename, metric_name, int(index_part))
                    else:
                        _load_non_indexed_sketch(folder_path, filename, metric_name)
    
        def _parse_filename(filename):
            """
            Parse the filename to extract the metric name and index.

            Args:
                filename (str): The filename of the binary file.

            Returns:
                tuple: (metric_name, index) where index is None for non-indexed files.
            """
            base_name = filename.replace('.bin', '')
            if '_' in base_name:
                metric_name, index_part = base_name.rsplit('_', 1)
                return metric_name, index_part
            else:
                return base_name, None

        def _load_indexed_sketch(folder_path, filename, metric_name, index):
            """
            Load an indexed sketch from a binary file.

            Args:
                folder_path (str): Path to the folder containing the binary file.
                filename (str): The filename of the binary file.
                metric_name (str): The name of the metric.
                index (int): The index of the sketch.
            """
            if metric_name not in self.sketch_registry:
                self.sketch_registry[metric_name] = []
        
            while len(self.sketch_registry[metric_name]) <= index:
                self.sketch_registry[metric_name].append(None)
        
            try:
                with open(os.path.join(folder_path, filename), 'rb') as file:
                    self.sketch_registry[metric_name][index] = datasketches.kll_floats_sketch.deserialize(file.read())
            except IOError as e:
                print(f"Error loading file {filename}: {e}")

        def _load_non_indexed_sketch(folder_path, filename, metric_name):
            """
            Load a non-indexed sketch from a binary file.

            Args:
                folder_path (str): Path to the folder containing the binary file.
                filename (str): The filename of the binary file.
                metric_name (str): The name of the metric.
            """
            if metric_name not in self.sketch_registry:
                try:
                    with open(os.path.join(folder_path, filename), 'rb') as file:
                        self.sketch_registry[metric_name] = datasketches.kll_floats_sketch.deserialize(file.read())
                except IOError as e:
                    print(f"Error loading file {filename}: {e}")

        # Process each folder
        load_from_folder(os.path.join(base_path, 'imgstats'))
        load_from_folder(os.path.join(base_path, 'modelstats'))
        load_from_folder(os.path.join(base_path, 'customstats'))




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

    def publish_sketches(self, folder_path, endpoint_url, sensor_id="reference"):
        """
        Compress a folder and post it to an endpoint, then clean up intermediate files.

        Args:
            folder_path (str): The path to the folder to compress and post.
            endpoint_url (str): The URL of the endpoint to post to.
            sensor_id (str): The value for the 'sensorid' header.
        """
        # Generate a Unix timestamp
        timestamp = int(time.time())

        # Compress the folder into a tar.gz file
        tar_gz_path = tar_and_gzip_folder(folder_path, "stats")

        # Post the compressed file to the endpoint
        post_file_to_endpoint(tar_gz_path, endpoint_url, sensor_id, timestamp, "stats")

        # Delete the tar.gz file after posting
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
            print(f"Deleted temporary file: {tar_gz_path}")
        else:
            print(f"Temporary file not found: {tar_gz_path}")

