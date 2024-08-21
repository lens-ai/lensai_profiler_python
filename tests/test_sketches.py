import unittest
import numpy as np
import tensorflow as tf
import torch
from lensai_profiler.sketches import Sketches

class TestSketches(unittest.TestCase):

    def setUp(self):
        """
        Set up the testing environment before each test.
        """
        self.sketches = Sketches()

        # Register metrics for testing
        self.sketches.register_metric('brightness', num_channels=3)
        self.sketches.register_metric('sharpness', num_channels=3)

        # Sample data for testing
        self.sample_image_np = np.random.rand(100, 100, 3).astype(np.float32)  # Random RGB image
        self.sample_image_tf = tf.convert_to_tensor(self.sample_image_np)
        self.sample_image_torch = torch.tensor(self.sample_image_np)

    def test_update_sketches_with_tensorflow(self):
        """
        Test updating sketches using TensorFlow tensors.
        """
        self.sketches.update_sketches(
            brightness=self.sample_image_tf,
            sharpness=self.sample_image_tf
        )

        # Verify that sketches were updated correctly
        for sketch in self.sketches.sketch_registry['brightness']:
            self.assertGreater(sketch.get_n(), 0)
        for sketch in self.sketches.sketch_registry['sharpness']:
            self.assertGreater(sketch.get_n(), 0)

    def test_update_sketches_with_pytorch(self):
        """
        Test updating sketches using PyTorch tensors.
        """
        self.sketches.update_sketches(
            brightness=self.sample_image_torch,
            sharpness=self.sample_image_torch
        )

        # Verify that sketches were updated correctly
        for sketch in self.sketches.sketch_registry['brightness']:
            self.assertGreater(sketch.get_n(), 0)
        for sketch in self.sketches.sketch_registry['sharpness']:
            self.assertGreater(sketch.get_n(), 0)

    def test_save_and_load_sketches(self):
        """
        Test saving and loading of sketches.
        """
        # Update sketches with random data
        self.sketches.update_sketches(
            brightness=self.sample_image_tf,
            sharpness=self.sample_image_tf
        )

        save_path = './test_sketches'
        self.sketches.save_sketches(save_path)

        # Create a new Sketches object and load the saved sketches
        new_sketches = Sketches()
        new_sketches.register_metric('brightness', num_channels=3)
        new_sketches.register_metric('sharpness', num_channels=3)
        new_sketches.load_sketches(save_path)

        # Verify that the sketches were loaded correctly
        for i in range(3):
            self.assertEqual(
                self.sketches.sketch_registry['brightness'][i].get_n(),
                new_sketches.sketch_registry['brightness'][i].get_n()
            )
            self.assertEqual(
                self.sketches.sketch_registry['sharpness'][i].get_n(),
                new_sketches.sketch_registry['sharpness'][i].get_n()
            )

    def test_compute_thresholds(self):
        """
        Test computing percentile thresholds.
        """
        # Update sketches with random data
        self.sketches.update_sketches(
            brightness=self.sample_image_tf,
            sharpness=self.sample_image_tf
        )

        # Compute thresholds
        thresholds = self.sketches.compute_thresholds(lower_percentile=0.1, upper_percentile=0.9)

        # Verify that thresholds are computed and are within expected ranges
        for key, channel_thresholds in thresholds['brightness'].items():
            lower_threshold, upper_threshold = channel_thresholds
            self.assertGreater(lower_threshold, 0)
            self.assertLess(upper_threshold, 1)

if __name__ == '__main__':
    unittest.main()

