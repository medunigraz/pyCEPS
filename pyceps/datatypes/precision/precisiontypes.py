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
from dataclasses import dataclass
import numpy as np

RepositoryInfo = namedtuple('RepositoryInfo',
                            ['studyName',
                             'surfaceFile',
                             'surfaceFilePath'])
RepositoryInfo.__doc__ = """
A named tuple with details about the structure of a Precision data repository.

Fields:
    studyName : str
        name extracted from anatomical shell data.
    surfaceFile : str
        name of the file containing anatomical shell data
    surfaceFilePath : str
        path to the surface file, relative to repository root folder
"""

StudyInfo = namedtuple('StudyInfo',
                       ['name',
                        'mapInfo'
                        ])
StudyInfo.__doc__ = """
A named tuple with details of a Precision study.

Fields:
    name : str
        name of the study
    mapInfo : list of MapInfo
        detailed information about the maps stored with this study
"""

MapInfo = namedtuple('MapInfo',
                     ['name',
                      'surfaceFile',
                      'surfaceFilePath',
                      'dataLocation',
                      'mapType',
                      'version'
                      ])
MapInfo.__doc__ = """
A named tuple with details of a mapping procedure for a Precision study.

Fields:
    name : list of str
    surfaceFile : str
        name of the surface file for this map
    surfaceFilePath : str
        path to the surface file, relative to repository root folder
    dataLocation : str
        path to data files for this map, relative to repository root folder
    mapType : str
        type specifier
    version : str
        file version used in export from the EAM system
"""

FileInfo = namedtuple('FileInfo',
                      ['studyName',
                       'mapNames',
                       'dataPath',
                       'mapType',
                       'version'])
FileInfo.__doc__ = """
A named tuple with information extracted from data file headers.

Fields:
    studyName : str
        name of the study the files belong to
    mapNames : list of str 
        names of maps extracted from data files
    dataPath : str
        path to the data files, relative to repository root folder
    mapType : list of str
        type specifier for each map name
    version : str
        file version of data files
"""

@dataclass
class PrecisionXFileHeader:
    """Class for header information contained in PrecisionX CSV files.

    Parameters:
        version : str
        studyName : str
        mapName : str
        dataOffset : int
        numPoints : int
        waveName : str
            only for Wave files
        fs : float
            only for Wave files
        numSamples : int
            only for Wave files
    """
    version: str = ''
    studyName: str = ''
    mapName: str = ''
    mapType : str = ''
    dataOffset: int = -1
    numPoints: int = -1
    waveName: str = ''
    fs: float = 0.0
    numSamples: int = -1


@dataclass
class PrecisionXWaveData:
    """ Class for wave data contained in PrecisionX CSV files.

    Parameters:
        name : str
            name of the wave
        pointNumber : ndarray of int
        freezeGroup : ndarray of int
        traceName: ndarray of str
            name of the trace, i.e. the channel names
        fs : float
            sampling rate
        data : ndarray
    """
    name: str = ''
    pointNumber: np.ndarray = np.array([], dtype=np.int32)
    freezeGroup: np.ndarray = np.array([], dtype=np.int32)
    traceName: np.ndarray = np.array([], dtype=str)
    fs: float = 0.0
    data: np.ndarray = np.array([], dtype=np.single)

    def is_valid(self):
        return self.data.size != 0


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
