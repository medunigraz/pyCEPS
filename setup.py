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
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'pyceps'
DESCRIPTION = ('pyceps provides methods for importing EP studies from '
               'commercial Clinical Mapping Systems and to export data to '
               'openCARP compatible data formats.'
               )
LICENSE = 'GPLv3+'
URL = 'https://github.com/me/myproject'
EMAIL = 'robert.arnold@medunigraz.at'
AUTHOR = 'Robert Arnold'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '{{VERSION_PLACEHOLDER}}'

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


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        # self.status('Uploading the package to PyPI via Twine…')
        # os.system('twine upload dist/*')
        self.status('Uploading the package to TestPyPI via Twine…')
        os.system('twine upload --repository testpypi dist/*')

        # self.status('Pushing git tags…')
        # os.system('git tag v{0}'.format(VERSION))
        # os.system('git push --tags')

        sys.exit()


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
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Development Status :: 4 - Beta',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
