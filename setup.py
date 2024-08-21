from setuptools import setup, find_packages
import sys

# Check if TensorFlow or PyTorch is installed
try:
    import tensorflow as tf
except ImportError:
    try:
        import torch
    except ImportError:
        sys.stderr.write(
            "Error: Either TensorFlow or PyTorch must be installed to use this library.\n"
            "Please install the package with 'pip install your_package_name[tensorflow]' "
            "or 'pip install your_package_name[torch]'.\n"
        )
        sys.exit(1)

setup(
    name='lensai_profiler',
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'datasketches',
        # Other core dependencies
    ],tests_require=[
        'pytest',
    ],
    extras_require={
        'tensorflow': ['tensorflow'],
        'torch': ['torch'],
    },
    entry_points={
        'console_scripts': [
            # Define command-line scripts here if you have any
        ],
    },
    author='Venkata Pydipalli',
    author_email='vsnm.tej@gmail.com',
    description='LensAI: A Python Library for Image and Model Profiling with Probabilistic Data Structures using TensorFlow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lens-ai/lensai_profiler',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
