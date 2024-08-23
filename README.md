![Build Status](https://github.com/lens-ai/lensai_profiler_python/actions/workflows/python-app.yml/badge.svg)


# Lens AI - Data and Model Monitoring Library for Edge AI Applications

Lensai Profiler Python is a  Python library designed to profile data using probabilistic data structures. These profiles are essential for computing model and data drift, ensuring efficient and accurate monitoring of your AI models on edge devices.
The Pytorch support is planned in the upcoming releases.

## The following Lens AI tools 
- Lens AI Python Profiler
- Lens AI C++ Profiler
- Lens AI Monitoring Server

![alt text](https://github.com/lens-ai/lensai_profiler_python/blob/main/blockdiagram.png)

## Model monitoring pipeline for Edge devices

### Model Training - Use the lensai_profiler_tf 
During the model training process, use the lensai_profiler_tf library to profile the training data. This step is crucial for establishing baseline profiles for data and model performance.

### Model Deployment - Use the lensai_c++_profiler (As most of the edge models use C++ , python support in next release)
During model deployment, integrate the Lensai C++ profiling code from the following repository: GitHub Repository. This integration profiles the real-time data during inference, enabling continuous monitoring.

### Monitoring - Use lensai Monitoring Server 
Use the LensAI Server to monitor data and model drift by comparing profiles from the training phase and real-time data obtained during inference. Detailed instructions and implementation examples are available in the examples directory.


## Key Advantages
- The time complexity of the distribution profiles are sublinear space and time
- The space complexity is  𝑂(1/𝜖log(𝜖𝑁)) where as traditionally it is O(N).
- The time complexity while insertion and query time is O(log(1/ϵ)), where as traditionally it is O(N log(N))


### Scalability:
The KLL sketch scales well with large datasets due to its logarithmic space complexity. This makes it suitable for applications involving big data where traditional methods would be too costly.
### Streaming Data:
KLL sketches are particularly well-suited for streaming data scenarios. They allow for efficient updates and can provide approximate quantiles on-the-fly without needing to store the entire dataset.
### Accuracy:
The KLL sketch provides a controllable trade-off between accuracy and space. By adjusting the error parameter 𝜖, one can balance the precision of the quantile approximations against the memory footprint.
### Ease of Use:
The KLL sketch can be easily merged, allowing for distributed computation and aggregation of results from multiple data streams or partitions.
### Practical Performance:
In practice, the KLL sketch offers high accuracy for quantile estimation with significantly less memory usage than exact methods, making it a practical choice for many real-world applications.

## Installation
Install the Python package using pip:
```sh
pip install lensai_profiler_tf
```

## Features
- Computes the distribution of image metrics 
- Saves the metrics in corresponding files for further analysis
- Generated the metrics qunatiles that can be used on the edge device for sampling
- ✨Magic ✨

## Documentation:
Please refer for more detailed documentation:

[a link](https://lens-ai.github.io/lensai_profiler_python)

## Usage
For detailed usage, please refer to the provided iPython notebook. : This gives the metrics distributions and also their quantiles. These are used as the base line when computing data drift.

### Tensor Flow Usage

```python

import tensorflow as tf
from lensai_profiler import Metrics, calculate_percentiles
from lensai_profiler.sketches import Sketches
import os
import matplotlib.pyplot as plt
import numpy as np

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

metrics = Metrics(framework="tf")

# Initialize Sketches class
num_channels = 3  # Assuming RGB images
sketches = Sketches()

sketches.register_metric('brightness')
sketches.register_metric('sharpness')
sketches.register_metric('snr')

sketches.register_metric('channel_mean', num_channels=3)
sketches.register_metric('channel_histogram', num_channels=3)


# Process a batch of images in parallel
train_dataset = train_dataset.map(lambda images, labels: metrics.process_batch(images), num_parallel_calls=tf.data.AUTOTUNE)

# Iterate through the dataset and update the KLL sketches in parallel
for batch in train_dataset:
    brightness, sharpness, channel_mean, snr, channel_pixels = batch
    sketches.update_sketches(
        brightness=brightness,
        sharpness=sharpness,
        channel_mean=channel_mean,
        snr=snr,
        channel_histogram=channel_pixels
    )

# Save the KLL sketches to a specified directory
save_path = '/content/sample_data/'
sketches.save_sketches(save_path)

```

### Pytorch Usage

```python
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from lensai_profiler import Metrics
from lensai_profiler.sketches import Sketches
import os

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Path setup (equivalent to TensorFlow code)
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Image transformation and dataset setup
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # Set num_workers to 0 to avoid threading issues

# Initialize Metrics and Sketches classes
metrics = Metrics(framework="pt")
sketches = Sketches()

num_channels = 3  # Assuming RGB images
sketches.register_metric('brightness')
sketches.register_metric('sharpness')
sketches.register_metric('snr')
sketches.register_metric('channel_mean', num_channels=3)
sketches.register_metric('channel_histogram', num_channels=3)

# Process the dataset and update the KLL sketches
for images, labels in train_loader:
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
    brightness, sharpness, channel_mean, snr, channel_pixels = metrics.process_batch(images)
    
    sketches.update_sketches(
        brightness=brightness.cpu(),
        sharpness=sharpness.cpu(),
        channel_mean=channel_mean.cpu(),
        snr=snr.cpu(),
        channel_histogram=channel_pixels.cpu()
    )

# Save the KLL sketches to a specified directory
save_path = '/content/'
sketches.save_sketches(save_path)

```

## Complete Model monitoring pipeline 

### Model Training
During the model training process, use the lensai_profiler_tf library to profile the training data. This step is crucial for establishing baseline profiles for data and model performance.

### Model Deployment
During model deployment, integrate the Lensai C++ profiling code from the following repository: GitHub Repository. This integration profiles the real-time data during inference, enabling continuous monitoring.

### Monitoring
Use the LensAI Server to monitor data and model drift by comparing profiles from the training phase and real-time data obtained during inference. Detailed instructions and implementation examples are available in the examples directory.

## Repository Link: GitHub Repository
Examples and API Documentation
For implementation details, refer to the examples directory. The API documentation can be found at the following link: API Documentation.

## Future Development
The current library focuses on image data profiling and monitoring. Future releases will extend support to other data types.

## Contributions
We welcome new feature requests and bug reports. Feel free to create issues and contribute to the project.

Thank you for using Lens AI for your edge AI applications. For more information, please visit our GitHub Repository.




Lens AI Profiler Python uses a number of open source projects to work properly:

- [Datasketches]
- [Numpy]
- [Scipy]

And of course lens_ai_profiler itself is open source with a [public repository][dill]
 on GitHub.

## Upcoming features

Lens AI tensorflow is currently extended with the following features.

| Feature | Version |
| ------ | ------ |
| Adding Pytorch Support | 1.1.0 |
| Adding inference Support | 1.2.0 |
| Adding support for time series data | 1.3.0 |
| Adding Inference time profiling | 1.4.0 |

## Development

Want to contribute? Great!

## License

AGPL



