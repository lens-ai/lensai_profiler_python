import unittest
import os
import numpy as np
import tensorflow as tf
import torch
from lensai_profiler import Sketches  # Replace with the actual import path

class TestSketches(unittest.TestCase):

    def setUp(self):
        """Setup common test data and instances."""
        self.num_channels = 3
        self.sketches = Sketches()  # Initialize the Sketches class
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
        self.sketches.register_metric('test_metric', num_channels=self.num_channels)
        self.assertIn('test_metric', self.sketches.sketch_registry)
        self.assertEqual(len(self.sketches.sketch_registry['test_metric']), self.num_channels)

    def test_update_kll_sketch_tensorflow(self):
        """Test updating KLL sketch with TensorFlow tensor data."""
        self.sketches.register_metric('brightness')
        tf_values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.sketch_registry['brightness'], tf_values)
        # Ensure the sketch has been updated
        self.assertGreater(self.sketches.sketch_registry['brightness'].n, 0)

    def test_update_kll_sketch_pytorch(self):
        """Test updating KLL sketch with PyTorch tensor data."""
        self.sketches.register_metric('brightness')
        torch_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.sketches.update_kll_sketch(self.sketches.sketch_registry['brightness'], torch_values)
        # Ensure the sketch has been updated
        self.assertGreater(self.sketches.sketch_registry['brightness'].n, 0)

    def test_update_sketches_tensorflow(self):
        """Test updating all sketches with TensorFlow tensor data."""
        self.sketches.register_metric('brightness')
        self.sketches.register_metric('sharpness')
        self.sketches.register_metric('channel_mean', num_channels=self.num_channels)
        self.sketches.register_metric('snr')
        self.sketches.register_metric('channel_pixels', num_channels=self.num_channels)

        brightness = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        sharpness = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
        channel_mean = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        snr = tf.constant([7.0, 8.0, 9.0], dtype=tf.float32)
        channel_pixels = tf.constant([[10.0, 11.0, 12.0]], dtype=tf.float32)
        print(self.sketches.sketch_registry['channel_mean'][0].n)
        self.sketches.update_sketches(brightness=brightness, sharpness=sharpness, channel_mean=channel_mean, snr=snr, channel_pixels=channel_pixels)

        self.assertGreater(self.sketches.sketch_registry['brightness'].n, 0)
        self.assertGreater(self.sketches.sketch_registry['sharpness'].n, 0)
        self.assertGreater(self.sketches.sketch_registry['snr'].n, 0)
        self.assertGreater(self.sketches.sketch_registry['channel_mean'][0].n, 0)
        self.assertGreater(self.sketches.sketch_registry['channel_pixels'][0].n, 0)

    def test_save_and_load_sketches(self):
        """Test saving and loading sketches."""
        # Register and update a sketch with some dummy data
        self.sketches.register_metric('brightness')
        brightness = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.sketch_registry['brightness'], brightness)
        
        # Save sketches to disk
        self.sketches.save_sketches(self.save_path)

        # Create a new Sketches instance and load from disk
        new_sketches = Sketches()
        new_sketches.register_metric('brightness')  # Register before loading
        new_sketches.load_sketches(self.save_path)

        # Check if the loaded sketch is the same as the saved one
        self.assertEqual(new_sketches.sketch_registry['brightness'].n, self.sketches.sketch_registry['brightness'].n)

    def test_compute_thresholds(self):
        """Test the computation of thresholds."""
        self.sketches.register_metric('brightness')
        brightness = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
        self.sketches.update_kll_sketch(self.sketches.sketch_registry['brightness'], brightness)

        thresholds = self.sketches.compute_thresholds(lower_percentile=0.2, upper_percentile=0.8)

        self.assertIn('brightness', thresholds)
        self.assertEqual(len(thresholds['brightness']), 2)

    def test_custom_metric_logging(self):
        """Test logging a custom metric like embedding vectors."""
        self.sketches.register_metric('embeddings')
        embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
        self.sketches.update_kll_sketch(self.sketches.sketch_registry['embeddings'], embeddings)

        self.assertGreater(self.sketches.sketch_registry['embeddings'].n, 0)

if __name__ == '__main__':
    unittest.main()
