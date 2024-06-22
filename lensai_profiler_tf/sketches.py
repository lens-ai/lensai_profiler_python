import datasketches
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import tensorflow as tf
from lensai_profiler_tf.metrics import get_histogram_sketch, calculate_percentiles


class Sketches:
    def __init__(self, num_channels):
        self.kll_brightness = datasketches.kll_floats_sketch()
        self.kll_sharpness = datasketches.kll_floats_sketch()
        self.kll_snr = datasketches.kll_floats_sketch()
        self.kll_channel_mean = [datasketches.kll_floats_sketch() for _ in range(num_channels)]
        self.kll_pixel_distribution = [datasketches.kll_floats_sketch() for _ in range(num_channels)]

    def update_kll_sketch(self, sketch, values):
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
        futures = []
        with ThreadPoolExecutor() as executor:
            futures.append(executor.submit(self.update_kll_sketch, self.kll_brightness, brightness))
            futures.append(executor.submit(self.update_kll_sketch, self.kll_sharpness, sharpness))
            futures.append(executor.submit(self.update_kll_sketch, self.kll_snr, snr))
            num_channels = channel_mean.shape[-1]
            for i in range(num_channels):
                futures.append(executor.submit(self.update_kll_sketch, self.kll_channel_mean[i], channel_mean[:, i]))
                futures.append(executor.submit(self.update_kll_sketch, self.kll_pixel_distribution[i], channel_pixels[:,i]))
            
            for future in as_completed(futures):
                future.result()  # Will raise exceptions if any occurred during execution

    def tf_update_sketches(self, brightness, sharpness, channel_mean, snr, channel_pixels):
        tf.py_function(self.update_sketches, [brightness, sharpness, channel_mean, snr, channel_pixels], [])

    def save_sketches(self, save_path):
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
        thresholds = {}
        for attr in ['kll_brightness', 'kll_sharpness', 'kll_snr', 'kll_channel_mean', 'kll_pixel_distribution']:
            value = getattr(self, attr)
            if isinstance(value, list):
                thresholds[attr] = {}
                for idx, sketch in enumerate(value):
                    x, p = get_histogram_sketch(sketch)
                    (lower_percentile_value, upper_percentile_value) = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                    thresholds[attr][idx] = (lower_percentile_value, upper_percentile_value) 
            else:
                x, p = get_histogram_sketch(value)
                (lower_percentile_value, upper_percentile_value) = calculate_percentiles(x, p, lower_percentile, upper_percentile)
                thresholds[attr] = (lower_percentile_value, upper_percentile_value)
        return thresholds
