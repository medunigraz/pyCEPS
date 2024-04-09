# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest

from pyceps.fileio.cartoutils import (read_mesh_file,
                                      read_ecg_file_header, read_ecg_file,
                                      read_visitag_file)


@pytest.fixture(scope='module')
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'fixtures')


def test_read_mesh_file(test_data_dir):
    mesh_path = os.path.join(test_data_dir, 'test.mesh')
    with open(mesh_path, mode='rb') as fid:
        surface = read_mesh_file(fid, encoding='cp1252')

    expected = (10176, 5090)

    assert (len(surface.tris), len(surface.X)) == expected


def test_read_ecg_file_header(test_data_dir):
    ecg_file = os.path.join(test_data_dir, 'ECG_Export_v4.0.txt')
    with open(ecg_file, mode='rb') as fid:
        header = read_ecg_file_header(fid, encoding='cp1252')

    expected = dict(
        gain=0.003,
        header_lines=4,
        name_bip='20A_3-4',
        name_ref='V5',
        name_uni='20A_3',
        version='4.0',
        ecg_names=['CS1(11)',
                   'V1(22)',
                   'V2(23)',
                   'V3(24)',
                   'V4(25)',
                   'V5(26)',
                   'V6(27)',
                   '20A_3(33)',
                   '20A_4(34)',
                   'I(110)',
                   'II(111)',
                   'III(112)',
                   '20A_3-4(115)',
                   'aVL(171)',
                   'aVR(172)',
                   'aVF(173)'],
    )

    assert header == expected


def test_read_ecg_file(test_data_dir):
    ecg_file = os.path.join(test_data_dir, 'ECG_Export_v4.0.txt')
    with open(ecg_file, mode='rb') as fid:
        data = read_ecg_file(fid, skip_rows=4, encoding='cp1252')

    expected = (np.ndarray, (2500, 16), np.float32)

    assert (type(data), data.shape, data.dtype) == expected


def test_read_visitag_file(test_data_dir):
    visi_file = os.path.join(test_data_dir, 'Visitag_Sites.txt')
    with open(visi_file, mode='rb') as fid:
        data, names = read_visitag_file(fid, encoding='cp1252')

    expected = (
        np.ndarray,
        (43, 15),
        np.float32,
        [
            'Session',
            'ChannelID',
            'SiteIndex',
            'X', 'Y', 'Z',
            'DurationTime',
            'AverageForce',
            'MaxTemperature',
            'MaxPower',
            'BaseImpedance',
            'ImpedanceDrop',
            'FTI',
            'RFIndex',
            'TagIndexStatus'
        ]
    )

    assert (type(data), data.shape, data.dtype, names) == expected
