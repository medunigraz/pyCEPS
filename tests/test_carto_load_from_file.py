# -*- coding: utf-8 -*-

import os
import pytest
import zipfile

from pyceps.fileio.cartoio import CartoStudy
from pyceps.fileio.pathtools import Repository


@pytest.fixture(scope='module')
def test_data_dir():
    return os.path.dirname(__file__)


@pytest.fixture(scope='module')
def my_repository():
    repo_path = os.path.join('.', 'Export_VT-dummy-02_14_2024-11-23-36.zip')
    return Repository(repo_path)


@pytest.fixture(scope='module')
def my_study(test_data_dir):
    study_file = os.path.join(test_data_dir,
                              'VT dummy 02_14_2024 11-23-36.pyceps'
                              )
    study = CartoStudy(study_repo='',
                       pwd='',
                       encoding='cp1252'
                       )
    study.load(study_file, repo_path='')

    return study


def test_create_repository(test_data_dir):
    repo_path = os.path.join(test_data_dir,
                             'Export_VT-dummy-02_14_2024-11-23-36.zip'
                             )
    study_repo = Repository(repo_path)

    assert isinstance(study_repo.base, zipfile.Path)


def test_study_name(my_study):
    assert my_study.name == 'VT dummy 02_14_2024 11-23-36'


def test_study_system(my_study):
    assert my_study.system == 'carto3'
