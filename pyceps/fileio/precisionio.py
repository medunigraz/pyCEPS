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

import logging
import os
import numpy as np
import xml.etree.ElementTree as xml
import re

from pyceps.datatypes.surface import Surface, SurfaceSignalMap, SurfaceLabel
from pyceps.datatypes.precision.precisiontypes import (PrecisionSurfaceLabel,
                                                       dxlDataHeader, CFEDetection,
                                                       PrecisionLesion
                                                       )
from pyceps.datatypes.signals import Trace


logger = logging.getLogger(__name__)


class _CommentedTreeBuilder(xml.TreeBuilder):
    """
    XML TreeBuilder that preserves comments.

    Comments are added to the tree as '!comment' elements.
    """
    def comment(self, data):
        self.start('!comment', {})
        self.data(data)
        self.end('!comment')


def read_landmark_geo(filename):
    """
    Load Precision Volume.

    Loads surface vertex and triangle data and vertex normals.
    Adds surface maps to surface:
        vertex data (P-P voltage, LAT Isochronal, ...)
    Adds surface labels to surface:
        vertex status (good, scar, ...)
        surface of origin (volume ID from ModelGroups.xml)

    Parameters:
        filename : string
            path to DxLandmarkGeo.xml
    Raises:
        AttributeError : If file version is not supported
        AttributeError : If more than 1 volumes in file
    Returns:
        surface : Surface object
    """

    # create child logger
    log = logging.getLogger('{}.read_landmark_geo'.format(__name__))

    if not filename.endswith('.xml'):
        log.warning('XML file expected')
        return Surface([], [])

    if not os.path.exists(filename):
        log.warning('Model Group file {} not found'.format(filename))
        return Surface([], [])

    log.debug('reading Precision Models from {}'.format(filename))

    # create placeholders for surface map data
    verts = np.empty((0, 3), dtype=np.single)
    polys = np.empty((0, 3), dtype=int)
    norms = np.empty((0, 3), dtype=np.single)
    map_data = []
    view_matrix = None
    labels = []

    # build XML tree and get root element
    tree = xml.parse(filename,
                     parser=xml.XMLParser(target=_CommentedTreeBuilder()))
    root = tree.getroot()

    # check version
    version = root.find('DIFHeader').find('Version').text
    if not version == 'SJM_DIF_5.0':
        raise AttributeError('file version {} not supported'.format(version))

    # check volumes in file
    volumes = root.find('DIFBody').find('Volumes')
    if not int(volumes.get('number')) == 1:
        raise AttributeError('only 1 volume allowed per file')

    # extract volume data
    volume = volumes.find('Volume')
    name = volume.get('name')
    for i, elem in enumerate(volume):
        if elem.tag == 'Vertices':
            n_verts = int(elem.get('number'))
            verts = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=np.single)
            if not verts.shape[0] == n_verts:
                log.warning('inconsistency found: number of vertices does not '
                            'match data!')

        if elem.tag == 'Polygons':
            n_polys = int(elem.get('number'))
            polys = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=int)
            if not polys.shape[0] == n_polys:
                log.warning('inconsistency found: number of polygons does not '
                            'match data!')
            # adjust 1-based indexing
            polys = polys - 1

        if elem.tag == 'Normals':
            n_norms = int(elem.get('number'))
            norms = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=np.single)
            if not norms.shape[0] == n_norms:
                log.warning('inconsistency found: number of normals does not '
                            'match data!')

        if elem.tag == 'Map_data':
            # the type of map_data is listed in the comment BEFORE
            comment = volume[i - 1].text
            if 'Data values at each vertex of DxL map' not in comment:
                log.warning('unable to fetch type of map data!')
            data_name = comment.split('Data values at each vertex of DxL map')[
                1]
            data_name = data_name.strip().replace(' ', '_')
            n_data = int(elem.get('number'))
            values = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=np.single)
            if not values.shape[0] == n_data:
                log.warning('inconsistency found: number of data points does '
                            'not match data!')
            # build surface signal map
            map_data.append(SurfaceSignalMap(data_name,
                                             np.expand_dims(values, axis=1),
                                             'pointData',
                                             description='Precision generated '
                                                         'signal map'
                                             )
                            )

        if elem.tag == 'Map_status':
            # map status is listed in the comment BEFORE
            comment = volume[i - 1].text
            if 'Map_status for each vertex:' not in comment:
                log.warning('unable to fetch map status description!')
            comment = comment.split('Map_status for each vertex:')[1]
            # work out the value description
            status_desc = [{'value': int(s.split('=')[0].strip()),
                            'description': s.split('=')[1].strip()
                            }
                           for s in comment.split(',')
                           ]
            n_status = int(elem.get('number'))
            status = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=np.single)
            if not status.shape[0] == n_status:
                log.warning('inconsistency found: number of status points '
                            'does not match data!')
            # build surface signal map
            labels.append(SurfaceLabel('status',
                                       np.expand_dims(status, axis=1),
                                       'pointData',
                                       description=status_desc
                                       )
                            )

        if elem.tag == 'Surface_of_origin':
            n_origin = int(elem.get('number'))
            poly_origin = np.genfromtxt(
                (line for line in elem.text.splitlines()[1:]),
                dtype=np.single)
            if not poly_origin.shape[0] == n_origin:
                log.warning('inconsistency found: number of surface origin '
                            'points does not match data!')
            # build surface signal map
            labels.append(SurfaceLabel('origin',
                                       np.expand_dims(poly_origin, axis=1),
                                       'cellData'
                                       )
                            )

        if elem.tag == 'AP_MapViewMatrix':
            view_matrix = [float(x) for x in elem.text.split()]
            view_matrix = np.reshape(view_matrix, (4, 4), order='F')

    # read labels
    surf_labels = []
    for label in root.find('DIFBody').find('Labels').findall('Label'):
        surf_labels.append(
            PrecisionSurfaceLabel(
                label.get('Name'),
                np.array([float(v) for v in label.text.strip().split()],
                         dtype=np.single)
            )
        )

    # read ObjectMap
    obj_map = root.find('DIFBody').find('ObjectMap')
    rotation = [float(v) for v in obj_map.find('Rotation').text.split()]
    translation = [float(v) for v in obj_map.find('Rotation').text.split()]
    scaling = [float(v) for v in obj_map.find('Rotation').text.split()]

    # build surface and return
    return Surface(verts, polys,
                   vertices_normals=norms,
                   signal_maps=map_data,
                   labels=labels
                   )


def load_dxl_data(filename):
    """
    Read Precision DxL data file containing point information.

    Parameters:
        filename : str

    Raises:
        IOError : If file not found
        ValueError : If point data is inconsistent
        ValueError : If ECG data is inconsistent

    Returns:
        header : dxlDataHeader
        point_data : dict
        ecg_data : dict
        cfe_data : CFEDetection
    """

    # create child logger
    log = logging.getLogger('{}.load_dxl_data'.format(__name__))

    if not os.path.isfile(filename):
        raise IOError('DxL data file {} not found!'.format(filename))

    # read data at once, files are only ~19MB
    with open(filename, mode='r') as fid:
        data = fid.read()

    # extract header
    start_pos = 0
    end_pos = data.find('Begin data\n', start_pos)
    header_string = data[start_pos:end_pos]
    # read header lines at end of data section
    start_pos = data.find('Seg data len', end_pos)
    end_pos = data.find('rov trace', start_pos)
    header_string += data[start_pos:end_pos]
    # get header information
    header = parse_dxl_header(header_string)

    # read point data
    start_pos = data.find('pt number:')
    end_pos = data.find('Seg data len:', start_pos)
    point_string = data[start_pos:end_pos]
    # build data
    point_data = {}
    for line in point_string.splitlines():
        name = line.split(',')[0][:-1]
        values = line.split(',')[1:]
        point_data[name] = values

    # check if number of points is consistent
    if not all(len(item) == header.nPoints for item in point_data.values()):
        raise ValueError('number of data points does not match data!')

    # read electrogram data
    n_samples = np.ceil(header.sampleRate * header.exportedSeconds)
    start_pos = data.find('rov trace:', end_pos)
    end_pos = data.find('FFT spectrum is available for FFT maps only',
                        start_pos
                        )
    ecg_string = data[start_pos:end_pos]
    ecg_data = parse_dxl_egm_data(ecg_string)

    # check if number of points is consistent
    if not all(ecg_data[d]['values'].shape[0] == n_samples for d in ecg_data):
        raise ValueError('number of ecg points does not match data!')

    # read  CFE information
    start_pos = data.find('CFE detection rov trace:', end_pos)
    end_pos = data.find('EOF', start_pos)
    cfe_string = data[start_pos:end_pos]
    cfe_data = parse_dxl_cfe_data(cfe_string)

    return header, point_data, ecg_data, cfe_data


def parse_dxl_header(header_str):
    """
    Parse header information of DxL data file

    Parameters:
        header_str : string

    Raises:
        AttributeError: If version is not supported
        AttributeError: If data element in file is not 'dxl'

    Returns:
        info : dxlDataHeader

    """

    version = ''
    data_element = ''
    n_points = -1
    map_name = ''
    file_num = -1
    n_files = -1
    seg_data_len = np.nan
    exported_sec = np.nan
    sample_rate = np.nan
    cfe_sens = np.nan
    cfe_width = np.nan
    cfe_refractory = np.nan

    for line in header_str.splitlines():
        if line.startswith('St. Jude Medical. File Revision :'):
            version = line.split(':')[1].strip()
            if version not in ['5.2', '5.6']:
                raise AttributeError('file format {} not supported'
                                     .format(version))

        if line.startswith('Export Data Element :'):
            data_element = line.split(':')[1].strip()
            if not data_element.lower() == 'dxl':
                raise AttributeError('unexpected data element {}'
                                     .format(data_element))

        if line.startswith('Total number of data points (columns):'):
            n_points = int(line.split(',')[1])

        if re.findall(r'This is file \d+ of \d+ for map', line):
            line, map_name = line.split(',')
            file_num, n_files = [int(s) for s in re.findall(r'\d+', line)]

        if line.startswith('Seg data len:'):
            seg_data_len = float(line.split(',')[1])

        if line.startswith('Exported seconds:'):
            exported_sec = float(line.split(',')[1])

        if line.startswith('Sample rate:'):
            sample_rate = float(line.split(',')[1])

        if line.startswith('CFE P-P sensitivity (mv)'):
            cfe_sens = float(line.split(',')[1])

        if line.startswith('CFE Width (ms)'):
            cfe_width = float(line.split(',')[1])

        if line.startswith('CFE Refractory (ms)'):
            cfe_refractory = float(line.split(',')[1])

    return dxlDataHeader(version=version,
                         dataElement=data_element,
                         nPoints=n_points,
                         mapName=map_name,
                         fileNumber=[file_num, n_files],
                         segmentDataLength=seg_data_len,
                         exportedSeconds=exported_sec,
                         sampleRate=sample_rate,
                         cfeSensitivity=cfe_sens,
                         cfeWidth=cfe_width,
                         cfeRefractory=cfe_refractory
                         )


def parse_dxl_egm_data(data_str):
    """
    Parse ECG data section in DxL data file.

    Parameters:
         data_str : string

    Returns:
        ecg_data : dict of dict
            keys: rov, ref, spare1, spare2, spare3
            each ecg dict has keys name (channel that collected the data) and
            values
    """
    # read rov trace
    start_pos = data_str.find('rov trace:')
    end_pos = data_str.find('ref trace:', start_pos)
    rov_string = data_str[start_pos:end_pos]
    rov_names = rov_string.splitlines()[0].split(',')[1:]
    rov_data = rov_string.split('\n')[1:-1]
    rov_data = np.genfromtxt(rov_data, delimiter=',', dtype=np.single)

    # read ref trace
    start_pos = data_str.find('ref trace:')
    end_pos = data_str.find('spare1 trace:', start_pos)
    ref_string = data_str[start_pos:end_pos]
    ref_names = ref_string.splitlines()[0].split(',')[1:]
    ref_data = ref_string.split('\n')[1:-1]
    ref_data = np.genfromtxt(ref_data, delimiter=',', dtype=np.single)

    # read spare trace 1
    # TODO: spare traces are not always saved (?)
    start_pos = data_str.find('spare1 trace:')
    end_pos = data_str.find('spare2 trace:', start_pos)
    spare_string = data_str[start_pos:end_pos]
    spare1_names = spare_string.splitlines()[0].split(',')[1:]
    spare1_data = spare_string.split('\n')[1:-1]
    spare1_data = np.genfromtxt(spare1_data, delimiter=',', dtype=np.single)

    # read spare trace 2
    start_pos = data_str.find('spare2 trace:')
    end_pos = data_str.find('spare3 trace:', start_pos)
    spare_string = data_str[start_pos:end_pos]
    spare2_names = spare_string.splitlines()[0].split(',')[1:]
    spare2_data = spare_string.split('\n')[1:-1]
    spare2_data = np.genfromtxt(spare2_data, delimiter=',', dtype=np.single)

    # read spare trace 3
    start_pos = data_str.find('spare3 trace:')
    end_pos = -1
    spare_string = data_str[start_pos:end_pos]
    spare3_names = spare_string.splitlines()[0].split(',')[1:]
    spare3_data = spare_string.split('\n')[1:]
    spare3_data = np.genfromtxt(spare3_data, delimiter=',', dtype=np.single)

    return {'rov': {'names': rov_names, 'values': rov_data[:, 1:]},
            'ref': {'names': ref_names, 'values': ref_data[:, 1:]},
            'spare1': {'names': spare1_names, 'values': spare1_data[:, 1:]},
            'spare2': {'names': spare2_names, 'values': spare2_data[:, 1:]},
            'spare3': {'names': spare3_names, 'values': spare3_data[:, 1:]},
            }


def parse_dxl_cfe_data(cfe_str):
    """
    Parse CFE detection data in DxL data file.

    Parameters:
        cfe_str : string

    Returns:
        CFEDetection object
    """
    # trace for detection
    start_pos = cfe_str.find('CFE detection rov trace:')
    end_pos = cfe_str.find('CFE detection count', start_pos)
    trace_string = cfe_str[start_pos:end_pos]
    trace = trace_string.split(',')[1:]

    # detection count
    start_pos = cfe_str.find('CFE detection count')
    end_pos = cfe_str.find('CFE detection sample index', start_pos)
    count_string = cfe_str[start_pos:end_pos]
    count = np.genfromtxt(count_string.split(',')[1:],
                          delimiter=',',
                          dtype=int)

    # detection sample index
    start_pos = cfe_str.find('CFE detection sample index')
    end_pos = -1
    sample_string = cfe_str[start_pos:end_pos]

    sample_idx = np.genfromtxt(sample_string.splitlines()[1:],
                               delimiter=',',
                               dtype=int)[:, 1:]

    # sanity check
    if len(trace) != count.shape[0] and len(trace) != sample_idx.shape[1]:
        print('CFE data is inconsistent, cannot build data!')
        return {}

    # build CFE data
    cfe_str = []
    for i, name in enumerate(trace):
        cfe_str.append(CFEDetection(trace=name,
                                    count=count[i],
                                    sampleIndex=sample_idx[:, i]
                                    )
                       )
    return cfe_str


def load_ecg_data(filename):
    """
    Load  Precision ECG data.

    Parameters:
        filename : string

    Raises:
        IOError : If file not found

    Returns:
        list of Trace objects
    """

    # create child logger
    log = logging.getLogger('{}.load_ecg_data'.format(__name__))

    if not os.path.isfile(filename):
        raise IOError('ECG data file {} not found!'.format(filename))

    # read data at once, files are only ~kB
    fid = open(filename, mode='r')
    data = fid.read()
    fid.close()

    start_idx = data.find('Number of waves (columns):', 0)
    end_idx = data.find('\n', start_idx)
    n_waves = int(data[start_idx:end_idx].strip().split(',')[1])

    start_idx = data.find('Number of samples (rows):', 0)
    end_idx = data.find('\n', start_idx)
    n_samples = int(data[start_idx:end_idx].strip().split(',')[1])

    start_idx = end_idx + 1
    end_idx = data.find('\n', start_idx)
    ecg_header = data[start_idx:end_idx].strip().split(',')

    start_idx = end_idx + 1
    end_idx = data.find('EOF', start_idx)
    ecg_string = data[start_idx:end_idx]
    ecg_data_raw = np.genfromtxt(ecg_string.splitlines(),
                                 delimiter=',',
                                 dtype=str
                                 )
    # remove last column, this should be empty
    if np.all(ecg_data_raw[:, -1] == ''):
        ecg_data_raw = ecg_data_raw[:, :-1]

    # check data size
    if ecg_data_raw.shape != (n_samples, n_waves):
        log.warning('ECG data read from file differs from expected shape!')

    # work out sampling rate
    t_idx = ecg_header.index('t_ref')
    t = ecg_data_raw[:, t_idx].astype(float)
    fs = 1 / np.mean(np.diff(t))

    # build traces
    names = [n for n in ecg_header]
    traces = []
    for i, name in enumerate(names):
        # convert strings to data
        if name == 't_dws':
            data = ecg_data_raw[:, i]
        elif name.endswith('_ds') or name.endswith('_ps'):
            data = ecg_data_raw[:, i].astype(int)
        else:
            data = ecg_data_raw[:, i].astype(float)

        traces.append(
            Trace(name=name, data=data, fs=fs)
        )

    return traces


def load_lesion_data(filename):
    """
    Load Precision lesion data.

    Parameters:
        filename : string

    Raises:
        IOError : If file not found

    Returns:
        list of Lesion objects

    """

    # create child logger
    log = logging.getLogger('{}.load_lesion_data'.format(__name__))

    lesions = []

    if not os.path.isfile(filename):
        raise IOError('lesion data file {} not found!'.format(filename))

    # read data at once, files are only ~kB
    fid = open(filename, mode='r')
    data = fid.read()
    fid.close()

    start_idx = data.find('Number of waves (columns):', 0)
    end_idx = data.find('\n', start_idx)
    n_waves = int(data[start_idx:end_idx].strip().split(',')[1])

    start_idx = data.find('Number of samples (rows):', 0)
    end_idx = data.find('\n', start_idx)
    n_samples = int(data[start_idx:end_idx].strip().split(',')[1])

    start_idx = end_idx + 1
    end_idx = data.find('\n', start_idx)
    lesion_header = data[start_idx:end_idx].strip().split(',')

    start_idx = end_idx + 1
    end_idx = data.find('EOF', start_idx)
    lesion_string = data[start_idx:end_idx]
    lesion_data_raw = np.genfromtxt(lesion_string.splitlines(),
                                    delimiter=',',
                                    dtype=str
                                    )

    # find relevant data
    DATA_NAMES = ['x', 'y', 'z', 'Diameter',
                  'Type', 'Surface', 'Display', 'Visible',
                  'R', 'G', 'B'
                  ]
    # check if all data available
    if not all(n in lesion_header for n in DATA_NAMES):
        log.warning('could not all necessary lesion data, missing {}!'
                    .format([n for n in DATA_NAMES if n not in lesion_header]))
        return lesions

    # build lesion objects
    for i in range(n_samples):
        lesions.append(PrecisionLesion(
            X=np.array([float(lesion_data_raw[i, lesion_header.index('xw')]),
                        float(lesion_data_raw[i, lesion_header.index('yw')]),
                        float(lesion_data_raw[i, lesion_header.index('zw')]),
                        ]
                       ),
            diameter=float(lesion_data_raw[i, lesion_header.index(
                'Diameter')]),
            Type=lesion_data_raw[i, lesion_header.index('Type')],
            Surface=lesion_data_raw[i, lesion_header.index('Surface')],
            display=bool(int(lesion_data_raw[i, lesion_header.index(
                'Display')])),
            visible=bool(int(lesion_data_raw[i, lesion_header.index(
                'Visible')])),
            color=[float(lesion_data_raw[i, lesion_header.index('R')]),
                   float(lesion_data_raw[i, lesion_header.index('G')]),
                   float(lesion_data_raw[i, lesion_header.index('B')]),
                   ]
            )
        )

    return lesions
