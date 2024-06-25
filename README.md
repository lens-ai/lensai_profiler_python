![Build Status](https://github.com/lens-ai/lensai_profiler/actions/workflows/python-app.svg)


# Lens AI - Data and Model Monitoring Library for Edge AI Applications

Lensai Profiler TF is a TensorFlow-supported Python library designed to profile data using probabilistic data structures. These profiles are essential for computing model and data drift, ensuring efficient and accurate monitoring of your AI models on edge devices.

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


```python
import tensorflow as tf
from lensai_profiler_tf.metrics import process_batch
from lensai_profiler_tf.sketches import Sketches
import os

# Download and prepare the dataset
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

# Initialize Lens AI sketches
num_channels = 3  # Assuming RGB images
sketches = Sketches(num_channels)

# Apply map function in parallel to compute metrics
train_dataset = train_dataset.map(
    lambda images, labels: process_batch(images),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Iterate through the dataset and update the KLL sketches in parallel
for brightness, sharpness, channel_mean, snr, channel_pixels in train_dataset:
    sketches.tf_update_sketches(brightness, sharpness, channel_mean, snr, channel_pixels)

# Save the KLL sketches to a specified directory
save_path = '/content/sample_data/'
sketches.save_sketches(save_path)
thresholds = sketches.get_thresholds()

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




Dillinger uses a number of open source projects to work properly:

- [Datasketches] - HTML enhanced for web apps!
- [Numpy] - awesome web-based text editor
- [Scipy] - Markdown parser done right. Fast and easy to extend.

And of course lens_ai_profiler itself is open source with a [public repository][dill]
 on GitHub.

## Upcoming features

Lens AI tensorflow is currently extended with the following features.

| Feature | Version |
| ------ | ------ |
| Adding Pytorch Support | 1.0.1 |
| Adding inference Support | 1.0.1 |
| Adding support for time series data | 1.0.2 |
| Adding Inference time profiling | 1.0.2 |

## Development

Want to contribute? Great!

## License

MIT

**Free Software, Hell Yeah!**

