#!/usr/bin/env python

"""
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vehicle-predictor",
    version="1.0.0",
    description="",
    long_description=long_description,
    author="Parikshit Iyengar",
    author_email="piyengar9@gatech.edu",
    license="MIT",
    keywords="computer-vision",
    packages=find_packages(),
    python_requires=">= 3.5",
    install_requires=[
        "pytest",
        "matplotlib",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "pytorch-lightning-bolts",
        "ipywidgets",
        "torchmetrics",
        "efficientnet_pytorch",
    ]
)