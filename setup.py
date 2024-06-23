# setup.py

from setuptools import setup, find_packages

setup(
    name='lensai_profiler',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'datasketches'
    ],tests_require=[
        'pytest',
    ],
    entry_points={
        'console_scripts': [],
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
    python_requires='>=3.9',
)
