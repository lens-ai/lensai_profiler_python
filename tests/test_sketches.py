import unittest
import os
import numpy as np
import tensorflow as tf
import torch
from lensai_profiler import Sketches  # Replace with the actual import path

class TestSketches(unittest.TestCase):

    def setUp(self):
        """Setup common test data and instances."""
        self.envs = ["pt", "tf"]  # Environments: 'pt' for PyTorch, 'tf' for TensorFlow
        self.num_channels = int(os.getenv("CHANNELS", 3))  # Number of channels
        self.save_path = "./sketches_test"
        os.makedirs(self.save_path, exist_ok=True)

    def tearDown(self):
        """Clean up any resources after each test."""
        if os.path.exists(self.save_path):
            for root, dirs, files in os.walk(self.save_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.save_path)

    def test_environments(self):
        """Test all functionalities in both environments."""
        for env in self.envs:
            with self.subTest(env=env):
                os.environ["ENV"] = env
                sketches = Sketches()  # Initialize the Sketches class

                # Test Initialization
                sketches.register_metric('test_metric', num_channels=self.num_channels)
                self.assertIn('test_metric', sketches.sketch_registry)
                self.assertEqual(len(sketches.sketch_registry['test_metric']), self.num_channels)
                del sketches.sketch_registry['test_metric']
                 
                # Test Update KLL Sketch
                sketches.register_metric('brightness')
                if env == "tf":
                    values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
                elif env == "pt":
                    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
                
                sketches.update_kll_sketch(sketches.sketch_registry['brightness'], values)
                self.assertGreater(sketches.sketch_registry['brightness'].n, 0, "Sketch should be updated.") 
                # Test Update Sketches
                sketches.register_metric('sharpness')
                sketches.register_metric('channel_mean', num_channels=self.num_channels)
                sketches.register_metric('snr')
                sketches.register_metric('channel_pixels', num_channels=self.num_channels)

                if env == "tf":
                    brightness = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
                    sharpness = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
                    channel_mean = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
                    snr = tf.constant([7.0, 8.0, 9.0], dtype=tf.float32)
                    channel_pixels = tf.constant([[10.0, 11.0, 12.0]], dtype=tf.float32)
                elif env == "pt":
                    brightness = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
                    sharpness = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
                    channel_mean = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
                    snr = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
                    channel_pixels = torch.tensor([[10.0, 11.0, 12.0]], dtype=torch.float32)

                sketches.update_sketches(brightness=brightness, sharpness=sharpness, channel_mean=channel_mean, snr=snr, channel_pixels=channel_pixels)
                self.assertGreater(sketches.sketch_registry['brightness'].n, 0)
                self.assertGreater(sketches.sketch_registry['sharpness'].n, 0)
                self.assertGreater(sketches.sketch_registry['snr'].n, 0)
                self.assertGreater(sketches.sketch_registry['channel_mean'][0].n, 0)
                self.assertGreater(sketches.sketch_registry['channel_pixels'][0].n, 0)

                # Test Save and Load Sketches
                sketches.save_sketches(self.save_path)
                new_sketches = Sketches()
                new_sketches.load_sketches(self.save_path)
                self.assertIn('brightness', new_sketches.sketch_registry)
                original_sketch = sketches.sketch_registry['brightness']
                loaded_sketch = new_sketches.sketch_registry['brightness']
                self.assertEqual(original_sketch.n, loaded_sketch.n)

                # Test Compute Thresholds
                thresholds = sketches.compute_thresholds(lower_percentile=0.2, upper_percentile=0.8)
                self.assertIn('brightness', thresholds)
                self.assertEqual(len(thresholds['brightness']), 2)

                # Test Custom Metric Logging
                sketches.register_metric('embeddings')
                embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32) if env == "pt" else tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=tf.float32)
                sketches.update_kll_sketch(sketches.sketch_registry['embeddings'], embeddings)
                self.assertGreater(sketches.sketch_registry['embeddings'].n, 0)

if __name__ == '__main__':
    unittest.main()
