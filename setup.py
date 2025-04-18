#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pyCEPS allows to import, visualize and translate clinical EAM data.
#     Copyright (C) 2023  Robert Arnold
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

# adopted from: https://github.com/kennethreitz/setup.py

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'pyCEPS'
DESCRIPTION = ('pyceps provides methods for importing EP studies from '
               'commercial Clinical Mapping Systems and to export data to '
               'openCARP compatible data formats.'
               )
LICENSE = 'GPLv3+'
URL = 'https://github.com/medunigraz/pyCEPS'
EMAIL = 'robert.arnold@medunigraz.at'
AUTHOR = 'Robert Arnold'
REQUIRES_PYTHON = '>=3.8'
VERSION = '{{VERSION_PLACEHOLDER}}'
PROJECT_URLS = {
    'Github': 'https://github.com/medunigraz/pyCEPS',
    'Changelog': 'https://github.com/medunigraz/pyCEPS/blob/main/CHANGELOG.md',
}
CLASSIFIERS = [
    # Trove classifiers
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
# Note: this will only work if 'requirements.txt' is present in your
# MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        REQUIRED = f.readlines()
except FileNotFoundError:
    REQUIRED = []

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls=PROJECT_URLS,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['pyceps'],
    entry_points={
        # command = package.module:function
        'console_scripts': ['pyceps=pyceps.cli:run'],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,  # include package data listed in MANIFEST.in
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
