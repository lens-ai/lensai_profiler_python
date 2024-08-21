import unittest
import os
import numpy as np
import tensorflow as tf
import torch
from lensai_profiler import Sketches

class TestSketches(unittest.TestCase):

    def setUp(self):
        """Setup common test data and instances."""
        self.num_channels = 3
        self.sketches = Sketches(num_channels=self.num_channels)
        self.save_path = "./sketches_test"
        os.makedirs(self.save_path, exist_ok=True)

    def tearDown(self):
        """Cleanup any created files and directories."""
        if os.path.exists(self.save_path):
            for f in os.listdir(self.save_path):
                os.remove(os.path.join(self.save_path, f))
            os.rmdir(self.save_path)

    def test_initialization(self):
        """Test that the Sketches class initializes correctly."""
        self.assertEqual(len(self.sketches.kll_channel_mean), self.num_channels)
        self.assertEqual(len(self.sketches.kll_pixel_distribution), self.num_channels)

    def test_update_kll_sketch_tensorflow(self):
        """Test updating KLL sketch with TensorFlow tensor data."""
        tf_values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.kll_brightness, tf_values)
        # Ensure the sketch has been updated
        self.assertGreater(self.sketches.kll_brightness.get_n(), 0)

    def test_update_kll_sketch_pytorch(self):
        """Test updating KLL sketch with PyTorch tensor data."""
        torch_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.sketches.update_kll_sketch(self.sketches.kll_brightness, torch_values)
        # Ensure the sketch has been updated
        self.assertGreater(self.sketches.kll_brightness.get_n(), 0)

    def test_update_sketches_tensorflow(self):
        """Test updating all sketches with TensorFlow tensor data."""
        brightness = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        sharpness = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
        channel_mean = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        snr = tf.constant([7.0, 8.0, 9.0], dtype=tf.float32)
        channel_pixels = tf.constant([[10.0, 11.0, 12.0]], dtype=tf.float32)

        self.sketches.update_sketches(brightness, sharpness, channel_mean, snr, channel_pixels)

        self.assertGreater(self.sketches.kll_brightness.get_n(), 0)
        self.assertGreater(self.sketches.kll_sharpness.get_n(), 0)
        self.assertGreater(self.sketches.kll_snr.get_n(), 0)
        self.assertGreater(self.sketches.kll_channel_mean[0].get_n(), 0)
        self.assertGreater(self.sketches.kll_pixel_distribution[0].get_n(), 0)

    def test_save_and_load_sketches(self):
        """Test saving and loading sketches."""
        # Update sketches with some dummy data
        brightness = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.kll_brightness, brightness)
        
        # Save sketches to disk
        self.sketches.save_sketches(self.save_path)

        # Create a new Sketches instance and load from disk
        new_sketches = Sketches(num_channels=self.num_channels)
        new_sketches.load_sketches(self.save_path)

        # Check if the loaded sketch is the same as the saved one
        self.assertEqual(new_sketches.kll_brightness.get_n(), self.sketches.kll_brightness.get_n())

    def test_compute_thresholds(self):
        """Test the computation of thresholds."""
        brightness = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.kll_brightness, brightness)

        thresholds = self.sketches.compute_thresholds(lower_percentile=0.2, upper_percentile=0.8)

        self.assertIn('kll_brightness', thresholds)
        self.assertEqual(len(thresholds['kll_brightness']), 2)

    def test_custom_metric_logging(self):
        """Test logging a custom metric like embedding vectors."""
        embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
        self.sketches.update_kll_sketch(self.sketches.kll_brightness, embeddings)

        self.assertGreater(self.sketches.kll_brightness.get_n(), 0)

if __name__ == '__main__':
    unittest.main()

