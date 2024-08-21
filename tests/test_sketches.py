import unittest
import numpy as np
import os
import shutil

# Mock TensorFlow and PyTorch based on availability
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

from lensai_profiler.metrics import (
    calculate_brightness,
    calculate_sharpness_laplacian,
    calculate_channel_mean,
    calculate_snr,
    calculate_channel_histogram
)
from lensai_profiler.sketches import Sketches

class TestSketches(unittest.TestCase):

    def setUp(self):
        """
        Set up the testing environment before each test.
        """
        num_channels = 3
        self.metrics = {
            "brightness": calculate_brightness,
            "sharpness": calculate_sharpness_laplacian,
            "channel_mean": calculate_channel_mean,
            "snr": calculate_snr,
            "channel_histogram": calculate_channel_histogram
        }
        self.sketches = Sketches(num_channels=num_channels, metrics=self.metrics)

        # Sample data for testing
        self.sample_image_np = np.random.rand(100, 100, 3).astype(np.float32)  # Random RGB image
        if tf_available:
            self.sample_image_tf = tf.convert_to_tensor(self.sample_image_np)
        if torch_available:
            self.sample_image_torch = torch.tensor(self.sample_image_np)

        # Temporary directory for saving sketches
        self.temp_dir = './temp_sketches'
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        """
        Clean up after each test.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialize_sketches(self):
        """
        Test the initialization of the Sketches class.
        """
        self.assertEqual(len(self.sketches.kll_sketches), len(self.metrics))
        for metric_name in self.metrics.keys():
            self.assertEqual(len(self.sketches.kll_sketches[metric_name]), 3)

    def test_update_sketches_with_tensorflow(self):
        """
        Test updating sketches using TensorFlow tensors.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        brightness = self.metrics["brightness"](self.sample_image_tf)
        sharpness = self.metrics["sharpness"](self.sample_image_tf)
        channel_mean = self.metrics["channel_mean"](self.sample_image_tf)
        snr = self.metrics["snr"](self.sample_image_tf)
        channel_histogram = self.metrics["channel_histogram"](self.sample_image_tf)
        
        self.sketches.update_sketches(
            brightness=[brightness],
            sharpness=[sharpness],
            channel_mean=[channel_mean],
            snr=[snr],
            channel_histogram=[channel_histogram]
        )

        # Check if sketches were updated (i.e., not empty)
        for metric_name, sketches in self.sketches.kll_sketches.items():
            for sketch in sketches:
                self.assertGreater(sketch.get_n(), 0)

    def test_update_sketches_with_pytorch(self):
        """
        Test updating sketches using PyTorch tensors.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        brightness = self.metrics["brightness"](self.sample_image_torch)
        sharpness = self.metrics["sharpness"](self.sample_image_torch)
        channel_mean = self.metrics["channel_mean"](self.sample_image_torch)
        snr = self.metrics["snr"](self.sample_image_torch)
        channel_histogram = self.metrics["channel_histogram"](self.sample_image_torch)
        
        self.sketches.update_sketches(
            brightness=[brightness],
            sharpness=[sharpness],
            channel_mean=[channel_mean],
            snr=[snr],
            channel_histogram=[channel_histogram]
        )

        # Check if sketches were updated (i.e., not empty)
        for metric_name, sketches in self.sketches.kll_sketches.items():
            for sketch in sketches:
                self.assertGreater(sketch.get_n(), 0)

    def test_save_and_load_sketches(self):
        """
        Test saving and loading of sketches.
        """
        # Update sketches with random data
        if tf_available:
            brightness = self.metrics["brightness"](self.sample_image_tf)
            sharpness = self.metrics["sharpness"](self.sample_image_tf)
            self.sketches.update_sketches(
                brightness=[brightness],
                sharpness=[sharpness]
            )
        
        self.sketches.save_sketches(self.temp_dir)

        # Create a new Sketches object and load the saved sketches
        new_sketches = Sketches(num_channels=3, metrics=self.metrics)
        new_sketches.load_sketches(self.temp_dir)

        # Check that the sketches are loaded correctly
        for metric_name in self.metrics.keys():
            for i in range(3):
                self.assertEqual(self.sketches.kll_sketches[metric_name][i].get_n(), 
                                 new_sketches.kll_sketches[metric_name][i].get_n())

    def test_compute_thresholds(self):
        """
        Test the computation of thresholds based on percentiles.
        """
        # Update sketches with random data
        if tf_available:
            brightness = self.metrics["brightness"](self.sample_image_tf)
            sharpness = self.metrics["sharpness"](self.sample_image_tf)
            self.sketches.update_sketches(
                brightness=[brightness],
                sharpness=[sharpness]
            )

        thresholds = self.sketches.compute_thresholds(lower_percentile=0.1, upper_percentile=0.9)

        # Verify that thresholds are calculated and are within expected ranges
        for metric_name, channel_thresholds in thresholds.items():
            for _, (lower, upper) in channel_thresholds.items():
                self.assertTrue(0 <= lower <= upper <= 1)


if __name__ == '__main__':
    unittest.main()
