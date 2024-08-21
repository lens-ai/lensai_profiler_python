import unittest
import tensorflow as tf
import numpy as np
import os

from lensai_profiler.metrics import Metrics, calculate_percentiles, get_histogram_sketch

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

class TestMetrics(unittest.TestCase):

    def setUp(self):
        """
        Set up the testing environment before each test.
        """
        self.image_shape = (100, 100, 3)  # RGB image with 3 channels
        self.num_images = 10
        self.sample_image_np = np.random.rand(*self.image_shape).astype(np.float32)  # Random image

        if tf_available:
            self.metrics_tf = Metrics(framework='tf')
            self.sample_image_tf = tf.convert_to_tensor(self.sample_image_np)
        if torch_available:
            self.metrics_pt = Metrics(framework='pt')
            self.sample_image_pt = torch.tensor(self.sample_image_np).permute(2, 0, 1)  # PyTorch: C x H x W

    def tearDown(self):
        """
        Clean up after each test.
        """
        pass

    def test_initialize_tf_metrics(self):
        """
        Test the initialization of the Metrics class with TensorFlow.
        """
        if tf_available:
            self.assertEqual(self.metrics_tf.framework, 'tf')
        else:
            self.skipTest("TensorFlow is not installed.")

    def test_initialize_pt_metrics(self):
        """
        Test the initialization of the Metrics class with PyTorch.
        """
        if torch_available:
            self.assertEqual(self.metrics_pt.framework, 'pt')
        else:
            self.skipTest("PyTorch is not installed.")

    def test_calculate_brightness_tf(self):
        """
        Test the calculation of brightness using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        brightness = self.metrics_tf.calculate_brightness(self.sample_image_tf)
        self.assertIsInstance(brightness, tf.Tensor)
        self.assertEqual(brightness.shape, ())

    def test_calculate_brightness_pt(self):
        """
        Test the calculation of brightness using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        brightness = self.metrics_pt.calculate_brightness(self.sample_image_pt)
        self.assertIsInstance(brightness, torch.Tensor)
        self.assertEqual(brightness.shape, ())

    def test_calculate_sharpness_tf(self):
        """
        Test the calculation of sharpness using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        sharpness = self.metrics_tf.calculate_sharpness_laplacian(self.sample_image_tf)
        self.assertIsInstance(sharpness, tf.Tensor)
        self.assertEqual(sharpness.shape, ())

    def test_calculate_sharpness_pt(self):
        """
        Test the calculation of sharpness using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        sharpness = self.metrics_pt.calculate_sharpness_laplacian(self.sample_image_pt)
        self.assertIsInstance(sharpness, torch.Tensor)
        self.assertEqual(sharpness.shape, ())

    def test_calculate_channel_mean_tf(self):
        """
        Test the calculation of channel mean using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        channel_mean = self.metrics_tf.calculate_channel_mean(self.sample_image_tf)
        self.assertIsInstance(channel_mean, tf.Tensor)
        self.assertEqual(channel_mean.shape, (3,))  # Should have 3 values for RGB

    def test_calculate_channel_mean_pt(self):
        """
        Test the calculation of channel mean using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        channel_mean = self.metrics_pt.calculate_channel_mean(self.sample_image_pt)
        self.assertIsInstance(channel_mean, torch.Tensor)
        self.assertEqual(channel_mean.shape, (3,))  # Should have 3 values for RGB

    def test_calculate_snr_tf(self):
        """
        Test the calculation of SNR using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        snr = self.metrics_tf.calculate_snr(self.sample_image_tf)
        self.assertIsInstance(snr, tf.Tensor)
        self.assertEqual(snr.shape, ())

    def test_calculate_snr_pt(self):
        """
        Test the calculation of SNR using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        snr = self.metrics_pt.calculate_snr(self.sample_image_pt)
        self.assertIsInstance(snr, torch.Tensor)
        self.assertEqual(snr.shape, ())

    def test_calculate_channel_histogram_tf(self):
        """
        Test the calculation of channel histogram using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        channel_histogram = self.metrics_tf.calculate_channel_histogram(self.sample_image_tf)
        self.assertIsInstance(channel_histogram, tf.Tensor)
        self.assertEqual(channel_histogram.shape[-1], 3)  # Should have 3 channels for RGB

    def test_calculate_channel_histogram_pt(self):
        """
        Test the calculation of channel histogram using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        channel_histogram = self.metrics_pt.calculate_channel_histogram(self.sample_image_pt)
        self.assertIsInstance(channel_histogram, torch.Tensor)
        self.assertEqual(channel_histogram.shape[0], 3)  # Should have 3 channels for RGB

    def test_process_batch_tf(self):
        """
        Test the processing of a batch of images using TensorFlow.
        """
        if not tf_available:
            self.skipTest("TensorFlow is not installed.")
        
        batch_images = tf.convert_to_tensor(np.random.rand(self.num_images, *self.image_shape).astype(np.float32))
        results = self.metrics_tf.process_batch(batch_images)
        self.assertEqual(len(results), 5)  # Expect 5 metric results
        for result in results:
            self.assertEqual(result.shape[0], self.num_images)

    def test_process_batch_pt(self):
        """
        Test the processing of a batch of images using PyTorch.
        """
        if not torch_available:
            self.skipTest("PyTorch is not installed.")
        
        batch_images = torch.tensor(np.random.rand(self.num_images, *self.image_shape).astype(np.float32)).permute(0, 3, 1, 2)
        results = self.metrics_pt.process_batch(batch_images)
        self.assertEqual(len(results), 5)  # Expect 5 metric results
        for result in results:
            self.assertEqual(result.shape[0], self.num_images)

    def test_calculate_percentiles(self):
        """
        Test the calculate_percentiles function.
        """
        x = np.linspace(0, 10, 100)
        probabilities = np.ones(100) / 100
        lower_percentile, upper_percentile = calculate_percentiles(x, probabilities, lower_percentile=0.1, upper_percentile=0.9)
        self.assertEqual(lower_percentile, 1.0)
        self.assertEqual(upper_percentile, 9.0)

    def test_get_histogram_sketch(self):
        """
        Test the get_histogram_sketch function.
        """
        sketch = datasketches.kll_floats_sketch()
        data = np.random.normal(loc=5, scale=2, size=1000)
        for value in data:
            sketch.update(value)
        
        x, pmf = get_histogram_sketch(sketch)
        self.assertIsNotNone(x)
        self.assertIsNotNone(pmf)
        self.assertEqual(len(x), 31)  # num_splits + 1
        self.assertEqual(len(pmf), 30)  # num_splits

class TestCalculatePercentiles(unittest.TestCase):

    def test_valid_inputs(self):
        x = [1, 2, 3, 4, 5]
        probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]
        lower, upper = calculate_percentiles(x, probabilities, 0.2, 0.8)
        self.assertAlmostEqual(lower, 2)
        self.assertAlmostEqual(upper, 4)

    def test_mismatched_lengths(self):
        x = [1, 2, 3]
        probabilities = [0.1, 0.2]
        with self.assertRaises(ValueError):
            calculate_percentiles(x, probabilities)

    def test_probabilities_not_summing_to_one(self):
        x = [1, 2, 3]
        probabilities = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError):
            calculate_percentiles(x, probabilities)

    def test_edge_case_single_nonzero_probability(self):
        x = [1, 2, 3]
        probabilities = [0, 1, 0]
        lower, upper = calculate_percentiles(x, probabilities, 0.01, 0.99)
        self.assertEqual(lower, 2)
        self.assertEqual(upper, 2)

if __name__ == '__main__':
    unittest.main()

