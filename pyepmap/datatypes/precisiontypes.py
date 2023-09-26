# -*- coding: utf-8 -*-
# Created by Robert at 23.08.2023

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

CFEDetection = namedtuple('CFEDetection',
                          ['trace', 'count', 'sampleIndex'])

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
