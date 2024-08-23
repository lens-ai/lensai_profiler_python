import unittest
import numpy as np
from lensai_profiler.metrics import Metrics, calculate_percentiles, get_histogram_sketch

class TestLensaiMetrics(unittest.TestCase):

    def setUp(self):
        # Create test images for both TF and PT
        self.frameworks = ['tf', 'pt']
        self.images_rgb = {}
        self.images_rgba = {}

        for framework in self.frameworks:
            if framework == 'tf':
                import tensorflow as tf
                self.images_rgb[framework] = tf.constant([
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                ], dtype=tf.float32)
                self.images_rgba[framework] = tf.constant([
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
                    [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]
                ], dtype=tf.float32)
            elif framework == 'pt':
                import torch
                self.images_rgb[framework] = torch.tensor([
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                ], dtype=torch.float32)
                self.images_rgba[framework] = torch.tensor([
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
                    [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]
                ], dtype=torch.float32)

    def test_calculate_brightness(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            brightness = metrics.calculate_brightness(image_rgb)
            if framework == 'tf':
                import tensorflow as tf
                expected_brightness = tf.reduce_mean(tf.image.rgb_to_grayscale(image_rgb))
                self.assertTrue(np.isclose(brightness.numpy(), expected_brightness.numpy()))
            elif framework == 'pt':
                import torch
                expected_brightness = torch.mean(torch.mean(image_rgb, dim=0))
                self.assertTrue(np.isclose(brightness.item(), expected_brightness.item()))

    def test_calculate_snr(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            snr = metrics.calculate_snr(image_rgb)
            if framework == 'tf':
                import tensorflow as tf
                grayscale = tf.image.rgb_to_grayscale(image_rgb)
                mean, variance = tf.nn.moments(grayscale, axes=[0, 1])
                sigma = tf.sqrt(variance)
                expected_snr = tf.where(sigma == 0, np.inf, 20 * tf.math.log(mean / (sigma + 1e-7)) / tf.math.log(10.0))
                self.assertTrue(np.isclose(snr.numpy(), expected_snr.numpy()).all())
            elif framework == 'pt':
                import torch
                grayscale = torch.mean(image_rgb, dim=0, keepdim=True)
                mean = torch.mean(grayscale)
                sigma = torch.std(grayscale)
                expected_snr = torch.where(sigma == 0, torch.tensor(float('inf')), 20 * torch.log10(mean / (sigma + 1e-7)))
                self.assertTrue(np.isclose(snr.item(), expected_snr.item()))


    def test_calculate_sharpness_laplacian(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            sharpness = metrics.calculate_sharpness_laplacian(image_rgb)

            if framework == 'tf':
                import tensorflow as tf
                kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
                kernel = tf.reshape(kernel, [3, 3, 1, 1])  # Kernel shape for TensorFlow [height, width, in_channels, out_channels]

                if image_rgb.shape[-1] != 1:  # Check if the image is not grayscale
                    grayscale = tf.image.rgb_to_grayscale(image_rgb)
                else:
                    grayscale = image_rgb

                grayscale = tf.expand_dims(grayscale, axis=0)  # Add batch dimension
                expected_sharpness = tf.nn.conv2d(grayscale, kernel, strides=[1, 1, 1, 1], padding='SAME')
                expected_sharpness = tf.reduce_mean(tf.abs(expected_sharpness))
                self.assertTrue(np.isclose(sharpness.numpy(), expected_sharpness.numpy()))

            elif framework == 'pt':
                import torch
                # Define the Laplacian kernel for edge detection
                kernel = torch.tensor([[-1, -1, -1], 
                               [-1,  8, -1], 
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # Kernel shape: [out_channels, in_channels, height, width]

                # Convert the image to grayscale if it has more than one channel
                if image_rgb.size(1) != 1:  # Check if the image has more than one channel
                    grayscale = torch.mean(image_rgb, dim=0, keepdim=True)  # Convert to grayscale
                else:
                    grayscale = image_rgb  # Image is already grayscale

                # Ensure the grayscale image has the correct shape for convolution
                grayscale = grayscale.unsqueeze(0)  # Add a batch dimension if necessary, shape becomes [1, 1, H, W]

                # Apply the Laplacian kernel using conv2d
                expected_sharpness = torch.nn.functional.conv2d(grayscale, kernel, stride=1, padding=1)
                expected_sharpness = torch.mean(torch.abs(expected_sharpness))  # Calculate mean absolute sharpness value
                self.assertTrue(np.isclose(sharpness.item(), expected_sharpness.item()))
    
    def test_calculate_channel_mean(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            channel_mean = metrics.calculate_channel_mean(image_rgb)
            if framework == 'tf':
                import tensorflow as tf
                expected_mean = tf.reduce_mean(image_rgb, axis=[0, 1])
                self.assertTrue(np.allclose(channel_mean.numpy(), expected_mean.numpy()))
            elif framework == 'pt':
                import torch
                expected_mean = torch.mean(image_rgb, dim=[1, 2])
                self.assertTrue(np.allclose(channel_mean.numpy(), expected_mean.numpy()))

    def test_calculate_channel_histogram(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            channel_histogram = metrics.calculate_channel_histogram(image_rgb)
            expected_histogram_shape = (3, 256)
            self.assertTrue(channel_histogram.shape, expected_histogram_shape)

    def test_process_batch(self):
        for framework in self.frameworks:
            metrics = Metrics(framework=framework)
            image_rgb = self.images_rgb[framework]

            if framework == 'tf':
                import tensorflow as tf
                images = tf.stack([image_rgb, image_rgb], axis=0)
            elif framework == 'pt':
                import torch
                images = torch.stack([image_rgb, image_rgb], dim=0)

            brightness, sharpness, channel_mean, snr, channel_pixels = metrics.process_batch(images)

            # Check the shapes of the results
            self.assertEqual(brightness.shape[0], 2)
            self.assertEqual(sharpness.shape[0], 2)
            self.assertEqual(channel_mean.shape, (2, 3))
            self.assertEqual(snr.shape[0], 2)
            self.assertEqual(channel_pixels.shape, (2, 3, 256))  # Assuming image size is 2x2 with 3 channels

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
