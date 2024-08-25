from setuptools import setup, find_packages
import sys

setup(
    name='lensai_profiler',
    version='1.1.2',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'datasketches',
        # Other core dependencies
    ],tests_require=[
        'pytest',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.0.0'],
        'torch': ['torch>=1.0.0'],
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
