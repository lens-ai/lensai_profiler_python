# lensai_profiler/__init__.py

import sys

# Define the functions to be conditionally imported based on available frameworks
__all__ = [
    'calculate_brightness',
    'calculate_snr',
    'calculate_sharpness_laplacian',
    'calculate_channel_mean',
    'process_batch',
    'get_histogram_sketch',
    'calculate_percentiles',
    'Sketches'
]

try:
    from .metrics import (
        calculate_brightness,
        calculate_snr,
        calculate_sharpness_laplacian,
        calculate_channel_mean,
        process_batch,
        get_histogram_sketch,
        calculate_percentiles
    )
except ImportError as e:
    if 'tensorflow' in str(e) or 'torch' in str(e):
        sys.stderr.write(
            "Error: TensorFlow or PyTorch is required to use some metrics functions. "
            "Please install the appropriate package with 'pip install your_package_name[tensorflow]' "
            "or 'pip install your_package_name[torch]'.\n"
        )
        sys.exit(1)
    else:
        raise

from .sketches import Sketches
