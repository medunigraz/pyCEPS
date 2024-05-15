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

import os
import logging
from typing import IO, Union, Optional, Tuple, List, Dict
import re
import numpy as np

from pyceps.datatypes.surface import Surface, SurfaceSignalMap, SurfaceLabel
from pyceps.datatypes.carto.cartotypes import PointForces
from pyceps.datatypes.carto.paso import PaSoConfiguration, PaSoCorrelation


def read_mesh_file(
        fid: IO,
        invisible_groups: bool = False,
        encoding: str = 'cp1252'
) -> Surface:
    """
    Read a CARTO3 mesh file.

    Non-active vertices and triangles with GroupID=-1000000 are removed.
    If no file is found, a Surface object with no points and triangles is
    returned.

    Parameters:
        fid : file-like
            file handle to *.mesh file
        invisible_groups : boolean (optional)
            If False only triangles with ID>=0 (visible groups) are
            imported, else all invisible groups with negative IDs are
            imported.
        encoding : str
            file encoding used to read file

    Raises:
        ValueError : If end of sections is not recognized, i.e. no CRLF at end

    Returns:
        Surface
    """

    # create child logger
    log = logging.getLogger('{}.read_mesh_file'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    # create placeholders for surface map data
    n_verts = 0
    n_tris = 0
    verts = []
    verts_normals = []
    verts_group_id = []
    tris = []
    tris_normals = []
    tris_group_id = []
    verts_color_header = []
    verts_color = []
    verts_attr_header = []
    verts_attr_desc = []
    verts_attr = []

    # parse file
    line = fid.readline().decode(encoding=encoding)
    if 'triangulatedmeshversion2.0' not in line.lower():
        log.warning('unexpected version number in Carto3 mesh file')

    while True:
        line = fid.readline().decode(encoding=encoding)
        if not line:
            # either end of file or just a blank line.....
            break
        elif line.startswith('\r\n'):
            continue

        if line.startswith('NumVertex '):
            # space at end needed to not confuse with NumVertexColors
            n_verts = int(line.split('=')[1])
            log.debug('found {} vertices'.format(n_verts))

        elif line.startswith('NumTriangle '):
            n_tris = int(line.split('=')[1])
            log.debug('found {} triangles'.format(n_tris))

        elif line.startswith('[VerticesSection]'):
            log.debug('reading vertices section')
            # skip header line
            _ = fid.readline()
            # there is one blank line after header
            line = fid.readline().decode(encoding=encoding)
            if line != '\r\n':
                raise ValueError('unexpected vertices section in Carto3 '
                                 'mesh file')

            verts = np.full((n_verts, 3), np.nan, dtype=np.single)
            verts_normals = np.full((n_verts, 3), np.nan, dtype=np.single)
            verts_group_id = np.full((n_verts, 1),
                                     np.iinfo(int).min,
                                     dtype=int)

            for i in range(n_verts):
                line = fid.readline().decode(encoding=encoding)
                values = line.split('=')[1].split()
                verts[i, :] = np.array(values[0:3]).astype(np.single)
                verts_normals[i, :] = np.array(values[3:6]).astype(np.single)
                verts_group_id[i] = int(values[6])

            # next line must be blank
            line = fid.readline().decode(encoding=encoding)
            if line != '\r\n':
                raise ValueError('unexpected end of vertices section in '
                                 'mesh file')

        elif line.startswith('[TrianglesSection]'):
            log.debug('reading triangles section')
            # skip header line
            _ = fid.readline()
            # there is one blank line after header
            line = fid.readline().decode(encoding=encoding)
            if line != '\r\n':
                raise ValueError('unexpected triangles section in  '
                                 'mesh file')

            tris = np.full((n_tris, 3),
                           np.iinfo(int).min,
                           dtype=int)
            tris_normals = np.full((n_tris, 3),
                                   np.nan,
                                   dtype=float)
            tris_group_id = np.full((n_tris, 1),
                                    np.iinfo(int).min,
                                    dtype=int)

            for i in range(n_tris):
                line = fid.readline().decode(encoding=encoding)
                values = line.split('=')[1].split()
                tris[i, :] = np.array(values[0:3]).astype(int)
                tris_normals[i, :] = np.array(values[3:6]).astype(float)
                tris_group_id[i] = int(values[6])

            # next line must be blank or EOF
            line = fid.readline().decode(encoding=encoding)
            if not line:
                break
            if line != '\r\n':
                raise ValueError('unexpected end of triangles '
                                 'section in mesh file {}')

        elif line.startswith('[VerticesColorsSection]'):
            log.debug('reading vertices color section')
            prev_line = line
            line = fid.readline().decode(encoding=encoding)
            while not line == '\r\n':
                prev_line = line
                line = fid.readline().decode(encoding=encoding)

            # line before empty line (data) contains header information
            verts_color_header = prev_line.split(';')[1].split()
            # check last header name, my contain 2 values
            if verts_color_header[-1].endswith(']'):
                extra_header = verts_color_header[-1][:-1].split('[')
                # remove old and append new
                del verts_color_header[-1]
                verts_color_header.extend(extra_header)
            # get number of color maps from header
            n_colors = len(verts_color_header)
            # get number of color maps from data
            last_pos = fid.tell()
            line = fid.readline().decode(encoding=encoding)
            n_values = len(line.split('=')[1].split())
            if not n_values == n_colors:
                log.warning('VerticesColorSection header does not match '
                            'data, trying my best...')
                n_colors = n_values
            fid.seek(last_pos)

            verts_color = np.full((n_verts, n_colors),
                                  np.nan,
                                  dtype=float)

            for i in range(n_verts):
                line = fid.readline().decode(encoding=encoding)
                verts_color[i, :] = np.array(line.split('=')[1].split()
                                             ).astype(float)

            # next line must be blank
            line = fid.readline().decode(encoding=encoding)
            if line != '\r\n':
                raise ValueError('unexpected end of vertices color '
                                 'section in mesh file')

        elif line.startswith('[VerticesAttributesSection]'):
            log.debug('reading vertices attributes section')

            # read comments, comments start with ;
            line = fid.readline().decode(encoding=encoding)
            verts_attr_desc = [line]
            last_pos = fid.tell()
            while line.startswith(';'):
                last_pos = fid.tell()
                line = fid.readline().decode(encoding=encoding)
                verts_attr_desc.append(line)

            # attribute description contain "=", header line not
            verts_attr_header = [comment for comment in verts_attr_desc
                                 if '=' not in comment][0]
            verts_attr_header = verts_attr_header.split(';')[1].split()
            # get number of attributes from header
            n_attr = len(verts_attr_header)

            verts_attr = np.full((n_verts, n_attr),
                                 np.iinfo(int).min,
                                 dtype=int)

            # now check if there is data
            if line:
                # line is not empty, reset read pos and read rest of data
                fid.seek(last_pos)

                for i in range(n_verts):
                    line = fid.readline().decode(encoding=encoding)
                    verts_attr[i, :] = np.array(line.split('=')[1].split()
                                                ).astype(bool)

    # build surface
    log.debug('build surface object')
    surface = Surface(verts, tris,
                      vertices_normals=verts_normals,
                      tris_normals=tris_normals)
    # build surface signal maps
    log.debug('build surface signal maps from mesh data')
    if len(verts_color) > 0:
        try:
            act_col = verts_color_header.index('LAT')
            bip_col = verts_color_header.index('Bipolar')
            uni_col = verts_color_header.index('Unipolar')
            imp_col = verts_color_header.index('Impedance')
            frc_col = verts_color_header.index('Force')
        except ValueError:
            log.warning('one or more headers for vertices colors '
                        'could not be extracted! Using best guess '
                        'for data columns...')
            act_col = 2
            bip_col = 1
            uni_col = 0
            imp_col = 3
            frc_col = 10

        try:
            description = 'Carto3 generated signal map'
            map_data = [
                SurfaceSignalMap(
                    'LAT_system',
                    np.expand_dims(verts_color[:, act_col], axis=1),
                    'pointData',
                    description=description
                ),
                SurfaceSignalMap(
                    'BIP_system',
                    np.expand_dims(verts_color[:, bip_col], axis=1),
                    'pointData',
                    description=description
                ),
                SurfaceSignalMap(
                    'UNI_system',
                    np.expand_dims(verts_color[:, uni_col], axis=1),
                    'pointData',
                    description=description
                ),
                SurfaceSignalMap(
                    'IMP_system',
                    np.expand_dims(verts_color[:, imp_col], axis=1),
                    'pointData',
                    description=description
                ),
                SurfaceSignalMap(
                    'FRC_system',
                    np.expand_dims(verts_color[:, frc_col], axis=1),
                    'pointData',
                    description=description
                )
            ]
            # add maps to surface
            surface.add_signal_maps(map_data)
        except Exception as err:
            log.warning('failed to import surface signal maps: {}'.format(err))

    # build surface labels
    log.debug('build surface labels')
    labels = [SurfaceLabel(name='vertices_group_id',
                           values=verts_group_id,
                           location='pointData',
                           description='invalid == -1000000'),
              SurfaceLabel(name='triangulation_group_id',
                           values=tris_group_id,
                           location='cellData',
                           description='non active == -1000000; '
                                       'invisible < 0')
              ]
    # add labels to surface
    surface.add_labels(labels)

    # build surface attributes and add them as labels to surface
    # TODO: is this useful information?
    # get attribute names from comments
    log.debug('build surface attribute labels')
    attributes = []
    for i in range(len(verts_attr_header)):
        attributes.append(
            SurfaceLabel(
                name=verts_attr_header[i],
                values=np.expand_dims(verts_attr[:, i], axis=1),
                location='pointData',
                description=';'.join(verts_attr_desc)
            )
        )
    if attributes:
        surface.add_labels(attributes)

    # check for unreferenced vertices and remove them
    verts_unref = 1 + np.argwhere(
        np.diff(np.unique(surface.tris.flatten())) > 1
    )
    if verts_unref.shape[0] > 0:
        log.info('found unreferenced vertices: {}'.format(verts_unref))
        surface.remove_vertices(verts_unref[:, 0])

    # check for invalid vertices and remove them
    verts_invalid = np.argwhere(verts_group_id == -1000000)
    if verts_invalid.shape[0] > 0:
        log.info('found {} invalid vertices, removing and renumbering '
                 'triangles...'.format(verts_invalid.shape[0]))
        surface.remove_vertices(verts_invalid[:, 0])

    # remove non-active triangles, this should be safe now...
    tris_invalid = np.argwhere(tris_group_id == -1000000)
    if tris_invalid.shape[0] > 0:
        log.info('found {} invalid triangles, removing...'
                 .format(tris_invalid.shape[0]))
        surface.remove_tris(tris_invalid[:, 0])

    # remove invisible groups if not requested
    if not invisible_groups:
        # triangles with ID < 0 are invisible
        tris_id = surface.get_label('triangulation_group_id')
        tris_invisible = np.argwhere(tris_id.values < 0)
        if tris_invisible.shape[0] > 0:
            log.info('removing invisible groups...')
            surface.remove_tris(tris_invisible[:, 0])
            # TODO: What if no triangle has ID >= 0?
            if surface.tris.shape[0] < 1:
                log.warning('mesh has no visible triangle-groups!')

    return surface


def read_ecg_file_header(
        fid: IO,
        encoding: str = 'cp1252'
) -> Dict[str, Union[float, int, str, List[str]]]:
    """Reads a Carto3 ECG file header.

    Parameters:
        fid : file-like
            file handle to ECG file
        encoding : str
            file encoding used to read file

    Returns:
        dict
            gain : float
                scale factor of value
            name_bip : string
                bipolar channel name
            name_uni : string
                unipolar channel name
            name_ref : string
                reference channel name
            ecg_names : list of string
                channel names in file
            header_lines : int
                number of header lines
    """

    SUPPORTED_VERSIONS = ['ecg_export_4.0', 'ecg_export_4.1']

    # create child logger
    log = logging.getLogger('{}.read_ecg_file_header'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    file_header = {'version': '',
                   'gain': np.nan,
                   'name_bip': '',
                   'name_uni': '',
                   'name_ref': '',
                   'ecg_names': [],
                   'header_lines': 0}

    # read file version
    version = fid.readline().decode(encoding=encoding).rstrip()
    if not version.lower() in SUPPORTED_VERSIONS:
        log.info('version of Carto3 ECG file is not supported')
        return file_header
    file_header['version'] = version.split('_')[-1]
    file_header['header_lines'] = file_header['header_lines'] + 1

    # read gain
    line = fid.readline().decode(encoding=encoding).rstrip()
    if not line.lower().startswith('raw'):
        log.warning('unexpected header line (2) in Cart3 ECG file')
    file_header['header_lines'] = file_header['header_lines'] + 1
    file_header['gain'] = float(line.lower().split('=')[1])
    if not file_header['gain'] == 0.003:
        log.warning('unexpected gain ({}) in Carto3 ECG file'
                    .format(file_header['gain']))

    # read mapping channels
    if file_header['version'] == '4.0':
        # channel names are included up to version 4.0
        line = fid.readline().decode(encoding=encoding).rstrip()
        if not line.lower().startswith('unipolar'):
            log.warning('unexpected header line (3) in Carto3 ECG file, '
                        'trying next'
                        )
            line = fid.readline().decode(encoding=encoding).rstrip()
            if not line.lower().startswith('unipolar'):
                log.info('unexpected file header in Carto3 ECG file')
                return file_header
            file_header['header_lines'] = file_header['header_lines'] + 1
        file_header['header_lines'] = file_header['header_lines'] + 1

        uni_token = 'unipolar mapping channel='
        bip_token = 'bipolar mapping channel='
        ref_token = 'reference channel='

        str_start = line.lower().find(uni_token) + len(uni_token)
        str_end = line.lower().find(bip_token)
        file_header['name_uni'] = line[str_start:str_end].strip()

        str_start = line.lower().find(bip_token) + len(bip_token)
        str_end = line.lower().find(ref_token)
        file_header['name_bip'] = line[str_start:str_end].strip()

        str_start = line.lower().find(ref_token) + len(ref_token)
        file_header['name_ref'] = line[str_start:].split()[0].strip()
        # TODO: compare this to MATLAB version, i.e. uni2 name??

    # read column names
    line = fid.readline().decode(encoding=encoding).rstrip()
    file_header['ecg_names'] = ['{})'.format(x.strip())
                                for x in line.split(')')]
    # remove last occurrence of ")"
    if file_header['ecg_names'][-1] == ')':
        file_header['ecg_names'].pop()

    file_header['header_lines'] = file_header['header_lines'] + 1

    return file_header


def read_ecg_file(
        fid: IO,
        column_indices: Optional[List[int]] = None,
        skip_rows: Optional[int] = None,
        encoding: str = 'cp1252'
) -> np.ndarray:
    """Read data from a Carto3 ECG file.

    Parameters:
        fid : file-like
            file handle to ECG file
        column_indices : list of int
            data columns to import, None reads all columns
        skip_rows : int
            number of header lines to skip. If None, header is read in first.
        encoding : str
            file encoding used to read file

    Raises:
        ValueError : If number of data points per channel is not 2500

    Returns:
        ndarray (2500, n_channels)
    """

    # create child logger
    log = logging.getLogger('{}.read_ecg_file'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    if not skip_rows:
        ecg_header = read_ecg_file_header(fid, encoding=encoding)
        skip_rows = ecg_header['header_lines']

    data = np.loadtxt(fid,
                      dtype=np.float32,  # int in files, but converted to float
                      skiprows=skip_rows,
                      usecols=column_indices
                      )

    if not data.shape[0] == 2500:
        log.error('unexpected data size in Carto3 ECG file')
        raise ValueError

    return data


def read_force_file(
        fid: IO,
        encoding: str = 'cp1252'
) -> PointForces:
    """Reads a Carto3 point force file.

    Parameters:
        fid : file-like
            file handle to force file
        encoding : str
            file encoding used to read file

    Returns:
        PointForces object
    """

    # create child logger
    log = logging.getLogger('{}.read_force_file'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    force_data = {'force': np.nan,
                  'axialAngle': np.nan,
                  'lateralAngle': np.nan,
                  't_time': np.empty(0),
                  't_force': np.empty(0),
                  't_axialAngle': np.empty(0),
                  't_lateralAngle': np.empty(0),
                  'systemTime': np.empty(0)}

    # read file version
    version = fid.readline().decode(encoding=encoding).rstrip()
    if not version.lower().endswith('contactforce.txt_2.0'):
        log.warning('version in Carto3 point force file is not supported')
        return PointForces()

    line = fid.readline().decode(encoding=encoding).rstrip()
    token = 'Rate='
    rate = line[line.find(token) + len(token):].split()[0]
    if not rate == '50':
        log.debug('unexpected rate ({}) found in Carto3 point force file')
    token = 'Number ='
    num_points = line[line.find(token) + len(token):].split()[0]
    if not num_points == '200':
        log.debug('unexpected number of points ({}) found in Carto3 '
                  'point force file'
                  .format(num_points))

    line = fid.readline().decode(encoding=encoding).rstrip()
    token = 'Mode='
    mode = line[line.find(token) + len(token):].split()[0]
    if not mode == '0':
        log.debug('unexpected mode ({}) found in Carto3 point force file'
                  .format(mode))

    # read force value from IntervalGraph section
    line = fid.readline().decode(encoding=encoding).rstrip()
    if not line.lower().startswith('intervalgraph'):
        log.debug('unexpected IntervalGraph section (line 5) found in point '
                  'force file!')
    else:
        line = fid.readline().decode(encoding=encoding).rstrip()
        items = line.split()
        force_data['force'] = float(items[1])
        force_data['axialAngle'] = float(items[2])
        force_data['lateralAngle'] = float(items[3])

    # ignore lines 6 - 7
    _ = fid.readline()
    _ = fid.readline()

    line = fid.readline().decode(encoding=encoding).rstrip()
    if not line.lower().startswith('index'):
        log.debug('unexpected header line (8) in Carto3 point force file, '
                  'trying next line')
        line = fid.readline().decode(encoding=encoding).rstrip()
        if not line.lower().startswith('index'):
            log.warning('unexpected file header in Carto3 point force file')
            return PointForces()

    data = np.loadtxt(fid,
                      dtype=np.float32,
                      usecols=[1, 3, 4, 5, 2])

    force_data['t_time'] = data[:, 0].astype(np.int32)
    force_data['t_force'] = data[:, 1].astype(np.float32)
    force_data['t_axialAngle'] = data[:, 2].astype(np.float32)
    force_data['t_lateralAngle'] = data[:, 3].astype(np.float32)
    force_data['systemTime'] = data[:, 4].astype(np.int32)

    return PointForces(force=force_data['force'],
                       axial_angle=force_data['axialAngle'],
                       lateral_angle=force_data['lateralAngle'],
                       t_time=force_data['t_time'],
                       t_force=force_data['t_force'],
                       t_axial_angle=force_data['t_axialAngle'],
                       t_lateral_angle=force_data['t_lateralAngle'],
                       system_time=force_data['systemTime']
                       )


def read_visitag_file(
        fid: IO,
        encoding: str = 'cp1252'
) -> Tuple[np.ndarray, List[str]]:
    """Reads a Carto3 VisiTag file.

    Any type of VisiTag file can be read. File header is always one line
    followed by data.

    Parameters:
        fid : file-like
            file handle to VisiTag file
        encoding :
            file encoding used to read file

    Raises:
        ValueError : If number of column headers does not match data shape

    Returns:
        ndarray (n_cols, n_data) : file data
        list of string : column header names
    """

    # create child logger
    log = logging.getLogger('{}.read_visitag_file'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    # read header information
    col_headers = fid.readline().decode(encoding=encoding).rstrip().split()
    # save current read pos for later
    last_pos = fid.tell()
    if not fid.readline().decode(encoding=encoding):
        # file is empty!
        log.debug('VisiTag file {} is empty'.format(fid.name))
        return np.array([], dtype=np.float32, ndmin=2), []

    # reset reading position
    fid.seek(last_pos)
    data = np.loadtxt(fid,
                      dtype=np.float32,
                      skiprows=0,
                      ndmin=2,
                      )

    return data, col_headers


def read_paso_config(
        fid: IO,
        encoding: str = 'cp1252'
) -> PaSoConfiguration:
    """Read PaSo Configuration file.

    Parameters:
        fid: file-like
            file handle to PaSo configuration file
        encoding: str
            file encoding used for binary files

    Returns:
        PaSoConfiguration
    """

    # create child logger
    log = logging.getLogger('{}.read_paso_config'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    config = fid.readlines()
    config = [line.decode(encoding=encoding) for line in config]

    line = [line for line in config
            if line.startswith('IS-IS Correlation Threshold:')
            ][0]
    isis_corr_threshold = float(line.split(': ')[1].strip())
    line = [line for line in config
            if line.startswith('PM-IS Correlation Threshold:')
            ][0]
    pmis_corr_threshold = float(line.split(': ')[1].strip())
    line = [line for line in config
            if line.startswith('IS-IS Minimum Correlated Channels:')
            ][0]
    isis_min_channels = int(line.split(': ')[1].strip())
    line = [line for line in config
            if line.startswith('PM-IS Minimum Correlated Channels:')
            ][0]
    pmis_min_channels = int(line.split(': ')[1].strip())
    line = [line for line in config
            if line.startswith('IS Default Prefix Name:')
            ][0]
    is_prefix_name = line.split(': ')[1].strip()
    line = [line for line in config
            if line.startswith('PM Default Prefix Name:')
            ][0]
    pm_prefix_name = line.split(': ')[1].strip()

    return PaSoConfiguration(
        isisCorrelationThreshold=isis_corr_threshold,
        pmisCorrelationThreshold=pmis_corr_threshold,
        isisMinCorrelatedChannels=isis_min_channels,
        pmisMinCorrelatedChannels=pmis_min_channels,
        isDefaultPrefix=is_prefix_name,
        pmDefaultPrefix=pm_prefix_name
    )


def read_paso_correlations(
        fid: IO,
        encoding: str = 'cp1252'
) -> List[PaSoCorrelation]:
    """Read PaSo Correlations file.

    Parameters:
        fid: file-like
            file handle to PaSo configuration file
        encoding: str
            file encoding used for binary files

    Returns:
        list of PaSoCorrelation
    """

    # create child logger
    log = logging.getLogger('{}.read_paso_config'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    correlations = []

    while True:
        line = fid.readline().decode(encoding=encoding)
        if not line:
            break

        regex = re.compile(r"""(\d+)\sCorrelated to\s(\d+)""")
        match = regex.findall(line)
        if match:
            new_corr = PaSoCorrelation()

            # get type from file name, "ISIS" or "PMIS"
            new_corr.type = os.path.basename(fid.name).split('Correlations')[0]

            new_corr.ID = int(match[0][0])
            new_corr.correlatedTo = int(match[0][1])

            line = fid.readline().decode(encoding='cp1252')
            regex = re.compile(
                r""".*User Defined Average:\s([+-]?\d+\.?\d*)"""
            )
            match = regex.findall(line)
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.UserAverage = float(match[0])

            line = fid.readline().decode(encoding='cp1252')
            match = line.split('User Defined Channels: ')[1]
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.UserChannels = [float(val) for val in match.split()]

            line = fid.readline().decode(encoding='cp1252')
            regex = re.compile(r""".*User Defined WOI:\s*(\d+)\s*(\d+)""")
            match = regex.findall(line)
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.UserWOI = [int(match[0][0]), int(match[0][1])]

            line = fid.readline().decode(encoding='cp1252')
            regex = re.compile(r""".*System Average:\s([+-]?\d+\.?\d*)""")
            match = regex.findall(line)
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.SystemAverage = float(match[0])

            line = fid.readline().decode(encoding='cp1252')
            match = line.split('System Channels: ')[1]
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.SystemChannels = [float(val) for val in match.split()]

            line = fid.readline().decode(encoding='cp1252')
            regex = re.compile(r""".*System WOI:\s*(\d+)\s*(\d+)""")
            match = regex.findall(line)
            if not match:
                log.warning('wrong format in line {}'.format(line))
                break
            new_corr.SystemWOI = [int(match[0][0]), int(match[0][1])]

            correlations.append(new_corr)

    return correlations


def read_electrode_pos_file(
        fid: IO,
        encoding: str = 'cp1252'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a Carto3 electrode position file.

    Parameters:
        fid : file-like
            file handle to electrode position file
        encoding :
            file encoding used for binary files

    Raises:
        TypeError : If version is not supported

    Returns:
        ndarray (n_pos, 1) : electrode index
        ndarray (n_pos, 1) : recording time stamp
        ndarray (n_pos, 3) : electrode coordinates
    """

    # create child logger
    log = logging.getLogger('{}.read_electrode_pos_file'.format(__name__))

    log.debug('reading file {}'.format(fid.name))

    # reader file format info
    line = fid.readline().decode(encoding=encoding)
    if not line.rstrip().lower().endswith('_positions_2.0'):
        raise TypeError('version number of position file {} is not supported'
                        .format(line))

    # read position data, skip header line
    data = np.loadtxt(fid,
                      dtype=np.float32,
                      skiprows=1,
                      )

    idx = data[:, 0].astype(int)
    time = data[:, 1].astype(int)
    xyz = data[:, [2, 3, 4]]

    # get range of electrode positions
    lim = np.where(idx[:-1] != idx[1:])[0]

    # positions of last 3 electrodes for MEC is always identical to first
    if 'MEC' in fid.name:
        idx = idx[:lim[-2]]
        time = time[:lim[-2]]
        xyz = xyz[:lim[-2], :]

    return idx, time, xyz
