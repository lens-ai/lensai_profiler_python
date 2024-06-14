import unittest
import tensorflow as tf
import numpy as np
from lensai_profiler_tf.metrics import calculate_brightness, calculate_snr, calculate_sharpness_laplacian, calculate_channel_mean, process_batch, calculate_channel_histogram

class TestLensaiMetrics(unittest.TestCase):

    def setUp(self):
        # Create a test image of size 2x2 with 3 channels (RGB)
        self.image_rgb = tf.constant([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ], dtype=tf.float32)

        # Create a test image of size 2x2 with 4 channels (RGBA)
        self.image_rgba = tf.constant([
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
        ], dtype=tf.float32)

    def test_calculate_brightness(self):
        brightness = calculate_brightness(self.image_rgb)
        expected_brightness = tf.reduce_mean(tf.image.rgb_to_grayscale(self.image_rgb))
        self.assertTrue(np.isclose(brightness.numpy(), expected_brightness.numpy()))

    def test_calculate_snr(self):
        snr_rgb = calculate_snr(self.image_rgb)
        grayscale = tf.image.rgb_to_grayscale(self.image_rgb)
        mean, variance = tf.nn.moments(grayscale, axes=[0, 1])
        sigma = tf.sqrt(variance)
        signal = mean
        noise = sigma
        expected_snr_rgb = tf.where(noise == 0, np.inf, 20 * tf.math.log(signal / noise) / tf.math.log(10.0))
        self.assertTrue(np.isclose(snr_rgb.numpy(), expected_snr_rgb.numpy()).all())

        snr_rgba = calculate_snr(self.image_rgba)
        grayscale = tf.image.rgb_to_grayscale(self.image_rgba[..., :3])
        mean, variance = tf.nn.moments(grayscale, axes=[0, 1])
        sigma = tf.sqrt(variance)
        signal = mean
        noise = sigma
        expected_snr_rgba = tf.where(noise == 0, np.inf, 20 * tf.math.log(signal / noise) / tf.math.log(10.0))
        self.assertTrue(np.isclose(snr_rgba.numpy(), expected_snr_rgba.numpy()).all())

    def test_calculate_sharpness_laplacian(self):
        sharpness = calculate_sharpness_laplacian(self.image_rgb)
        kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        grayscale = tf.image.rgb_to_grayscale(self.image_rgb)
        grayscale = tf.expand_dims(grayscale, axis=0)
        expected_sharpness = tf.nn.conv2d(grayscale, kernel, strides=[1, 1, 1, 1], padding='SAME')
        expected_sharpness = tf.reduce_mean(tf.abs(expected_sharpness))
        self.assertTrue(np.isclose(sharpness.numpy(), expected_sharpness.numpy()))

    def test_calculate_channel_mean(self):
        channel_mean = calculate_channel_mean(self.image_rgb)
        expected_mean = tf.reduce_mean(self.image_rgb, axis=[0, 1])
        self.assertTrue(np.allclose(channel_mean.numpy(), expected_mean.numpy()))

    def test_calculate_channel_histogram(self):
        channel_histogram = calculate_channel_histogram(self.image_rgb)
        expected_histogram = tf.reshape(self.image_rgb, [-1, 3])
        self.assertTrue(np.allclose(channel_histogram.numpy(), expected_histogram.numpy()))

    def test_process_batch(self):
        images = tf.stack([self.image_rgb, self.image_rgb], axis=0)
        brightness, sharpness, channel_mean, snr, channel_pixels = process_batch(images)

        # Check the shapes of the results
        self.assertEqual(brightness.shape, (2,))
        self.assertEqual(sharpness.shape, (2,))
        self.assertEqual(channel_mean.shape, (2, images.shape[-1]))
        self.assertEqual(snr.shape, (2, 1))
        self.assertEqual(channel_pixels.shape, (2, 2*2, images.shape[-1]))

if __name__ == '__main__':
    unittest.main()

