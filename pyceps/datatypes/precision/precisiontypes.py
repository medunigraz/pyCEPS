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

from collections import namedtuple


PrecisionSurfaceLabel = namedtuple('PrecisionSurfaceLabel',
                                   ['name', 'X'])

dxlDataHeader = namedtuple('dxlDataHeader',
                           ['version',
                            'dataElement',
                            'nPoints',
                            'mapName',
                            'fileNumber',
                            'segmentDataLength',
                            'exportedSeconds',
                            'sampleRate',
                            'cfeSensitivity',
                            'cfeWidth',
                            'cfeRefractory'
                            ]
                           )

DetectionAlgorithm = namedtuple('DetectionAlgorithm',
                                ['code', 'parameter'])
DetectionAlgorithm.__doc__ = """
A named tuple representing detection algorithm in DxL data.

Fields:
    code : str
        detection algorithm code
    parameter : str
        detection algorithm parameter
"""

CFEDetection = namedtuple('CFEDetection',
                          ['trace', 'count', 'sampleIndex'])
CFEDetection.__doc__ = """
A named tuple representing CFE detection details

Fields:
    trace : str
        detection roving trace name
    count : int
        CFE detection count
    sampleIndex : ndarray of type int
        CFE detection sample index
"""

UniChannelConfig = namedtuple('ChannelConfig',
                              ['channel', 'cath', 'electrode', 'visible'])

BipChannelConfig = namedtuple('BipChannelConfig',
                              ['channel', 'uni1channel', 'uni2channel'])

PrecisionLesion = namedtuple('PrecisionLesion',
                             ['X',
                              'diameter',
                              'Type',
                              'Surface',
                              'display',
                              'visible',
                              'color'
                              ]
                             )
