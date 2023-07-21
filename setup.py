#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='rockml',
    version='0.19.0',
    description='Python Distribution Utilities',
    author='IBM Research',
    author_email='sallesd@br.ibm.com',
    url='https://github.ibm.com/BRLML/rockml',
    packages=find_packages(),
    install_requires=[
        'graphviz>=0.2,<=2.40.1',
        'h5py>=2.10.0',
        'numpy',
        'pandas==0.25.3',
        'pillow>=6.2.1',
        'pydot>=1.2.4',
        'pyyaml>=5.3',
        'scikit-image>=0.15',
        'scipy>=1.4.1',
        'tensorflow>=2.0.0',
        'pytest>=5.3.2',
    ],
    license='Internal use only / IBM only',
    include_package_data=True,
    python_requires='>=3.6',
    zip_safe=False,
)
