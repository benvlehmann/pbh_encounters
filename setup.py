#!/bin/env python

from setuptools import setup, find_packages

setup(
    name="pbh_encounters",
    version="0.1.0dev",
    author="Benjamin V. Lehmann",
    author_email="benvlehmann@gmail.com",
    description=("This is a package for simulating primordial black hole "
                 "encounters with the solar system and extracting "
                 "perturbations to planetary orbits."),
    url="https://github.com/benvlehmann/pbh_encounters",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "requests",
        "rebound",
        "emcee",
        "schwimmbad"
    ],
    python_requires='>=3.7',
    scripts=['bin/pbhe_mcmc', 'bin/pbhe_sample']
)
