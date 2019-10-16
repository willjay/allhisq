#!/usr/bin/env python
""" Setup script for allhisq. """
import os
import io
import setuptools
from numpy.distutils.core import setup
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding

HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with io.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='allhisq',
    version='1.0',
    description='Database, IO, and conventions related to allhisq calculations',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/willjay/allhisq',
    author='William Jay',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'pandas',
        'sqlalchemy',
        # postgres support with sqlalchemy
        # Must have either 'psycopg2' or 'psycopg2-binary'
        # I have found issues trying to install 'psycopg2'
        'psycopg2-binary', 
    ],
    # Provide executable script to run the main code
    entry_points={},
    #package_data={'': ['data/*']},
    extras_require={
        'test': ['pytest', 'coverage', 'pytest-cov'],
    },
)
