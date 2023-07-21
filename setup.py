#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='rockml',
    version='0.20.0',
    description='Python Distribution Utilities',
    author='IBM Research',
    author_email='sallesd@br.ibm.com',
    url='https://github.com/IBM/rockml',
    packages=find_packages(),
    install_requires=[
        'graphviz',
        'h5py',
        'numpy',
        'pandas',
        'pillow',
        'pydot',
        'pyyaml',
        'scikit-image',
        'scipy',
        'tensorflow',
        'pytest',
        'murmurhash',
        'lasio'
    ],
    license='Internal use only / IBM only',
    include_package_data=True,
    python_requires='>=3.10',
    zip_safe=False,
)
