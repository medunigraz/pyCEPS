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
from typing import Optional
import numpy as np


CartoUnits = namedtuple('CartoUnits', ['Distance', 'Angle'])
CartoUnits.__doc__ = """
A namedtuple representing units used in Carto3

Fields:
    Distance : str
        distance unit used in Carto3, standard is "mm"
    Angle : str
        angle unit used in Carto3, standard is "radian"
"""

Coloring = namedtuple('Coloring', ['Id',
                                   'Name',
                                   'TextureInvert',
                                   'Propagation',
                                   'Units'])
Coloring.__doc__ = """
A namedtuple representing coloring information.

Fields:
    Id : int
        unique ID for coloring type
    Name : str
        name of coloring type, for example 'LAT'
    TextureInvert : int
    Propagation : int
    Units : str
"""

ColoringRange = namedtuple('ColoringRange', ['Id', 'min', 'max'])
ColoringRange.__doc__ = """
A namedtuple representing color ranges for a coloring type.

Fields:
    Id : int
        unique ID of coloring type this range is applied to
    min : float
    max : float
"""

# SurfaceAttributes = namedtuple('SurfaceAttributes',
#                                ['name', 'description', 'value'])

SurfaceErrorTable = namedtuple('SurfaceErrorTable',
                               ['BadErrorColor',
                                'MedErrorColor',
                                'GoodErrorColor',
                                'BadErrorThreshold',
                                'MedErrorThreshold',
                                'GoodErrorThreshold'])
SurfaceErrorTable.__doc__ = """
A namedtuple representing color information for error visualization.

Fields:
    BadErrorColor : list of float
    MedErrorColor : list of float
    GoodErrorColor : list of float
    BadErrorThreshold : float
    MedErrorThreshold : float
    GoodErrorThreshold : float
"""

CFAEColoringTable = namedtuple('CFAEColoringTable',
                               ['IgnoreBelowColor',
                                'IclMediumColor',
                                'IclHighColor',
                                'IgnoreBelowThreshold',
                                'IclMediumThreshold',
                                'IclHighThreshold'])
CFAEColoringTable.__doc__ = """
A namedtuple representing color information for CFAE visualization.

Fields:
    IgnoreBelowColor : list of float
    IclMediumColor : list of float
    IclHighColor : list of float
    IgnoreBelowThreshold : float
    IclMediumThreshold : float
    IclHighThreshold : float
"""

Tag = namedtuple('Tag', ['ID',
                         'ShortName',
                         'FullName',
                         'Color',
                         'Radius'])
Tag.__doc__ = """
A namedtuple representing tags.

Fields:
    ID : int
    ShortName : str
    FullName : str
    Color : list of float
    Radius : float
"""

CartoMappingParameters = namedtuple('CartoMappingParameters',
                                    ['ColoringTable',
                                     'SurfaceErrorTable',
                                     'PasoTable',
                                     'CFAEColoringTable',
                                     'TagsTable'])
CartoMappingParameters.__doc__ = """
A namedtuple representing Carto mapping parameters for visualization.

Fields:
    ColoringTable : list of Coloring
    SurfaceErrorTable : SurfaceErrorTable
    PasoTable : PasoTable
    CFAEColoringTable : CFAEColoringTable
    TagsTable : list of Tag
"""

RefAnnotationConfig = namedtuple('RefAnnotationConfig',
                                 ['algorithm', 'connector'])
RefAnnotationConfig.__doc__ = """
A namedtuple representing configuration of reference detection method.

Fields:
    Algorithm : int
        algorithm used for detection of reference annotation. '1' is 
        supposedly indicating energy operator calculated from V1-V6
    Connector : int
"""


class RFAblationParameters:
    """
    A class representing Carto3 ablation data.

    Attributes:
        time : ndarray
            timestamps as type np.int32
        irrigation : ndarray
            irrigation as type np.int32
        powerMode : ndarray
            mode as type np.int32
        ablationTime : ndarray
            ablation time as type np.int32
        power : ndarray
            power as type np.int32
        impedance : ndarray
            impedance as type np.int32
        distalTemp : ndarray
            distal temperature in °C as type np.int32
        proximalTemp : ndarray
            proximal temperature in °C as type np.int32
    """

    def __init__(
            self,
            time: np.ndarray = np.empty(0, dtype=np.int32),
            irrigation: np.ndarray = np.empty(0, dtype=np.int32),
            power_mode: np.ndarray = np.empty(0, dtype=np.int32),
            abl_time: np.ndarray = np.empty(0, dtype=np.int32),
            power: np.ndarray = np.empty(0, dtype=np.int32),
            impedance: np.ndarray = np.empty(0, dtype=np.int32),
            distal_temp: np.ndarray = np.empty(0, dtype=np.int32),
            proximal_temp: np.ndarray = np.empty(0, dtype=np.int32),
    ) -> None:
        """Constructor."""

        self.time = time
        self.irrigation = irrigation
        self.powerMode = power_mode
        self.ablationTime = abl_time
        self.power = power
        self.impedance = impedance
        self.distalTemp = distal_temp
        self.proximalTemp = proximal_temp


class RFForce:
    """
    A class representing RF contact force data.

    Attributes:
        time : ndarray
            timestamps as type np.int32
        force : ndarray
            contact force in gr(?) as type np.float32
        axialAngle : ndarray
            axial angle in °(?) as type np.float32
        lateralAngle : ndarray
            lateral angle in °(?) as type np.float32
        ablationPoint : ndarray
            ID of CartoPoint if point has Tag 'Ablation' as type np.int32
        position : ndarray
            position (projected onto surface) of CartoPoint if point has Tag
            'Ablation' as type np.float32
    """

    def __init__(
            self,
            time: np.ndarray = np.empty(0, dtype=np.int32),
            force: np.ndarray = np.empty(0, dtype=np.float32),
            axial_angle: np.ndarray = np.empty(0, dtype=np.float32),
            lateral_angle: np.ndarray = np.empty(0, dtype=np.float32),
            abl_point: np.ndarray = np.empty(0, dtype=np.int32),
            position: np.ndarray = np.empty((0, 3), dtype=np.float32)
    ) -> None:
        """Constructor."""

        self.time = time
        self.force = force
        self.axialAngle = axial_angle
        self.lateralAngle = lateral_angle
        self.ablationPoint = abl_point
        self.position = position


class MapRF:
    """
    A class representing force and ablation data during a mapping procedure.

    Attributes:
        force : RFForce
        ablParams : RFAblationParameters
    """

    def __init__(
            self,
            force: Optional[RFForce] = None,
            ablation_parameters: Optional[RFAblationParameters] = None
    ) -> None:
        """Constructor."""

        self.force = force
        self.ablParams = ablation_parameters


class PointForces:
    """
    A class representing contact force data recorded with a Carto point.

    Attributes:
        force : float
            contact force in g(?) at LAT annotation
        axialAngle : float
            axial angle in °(?) at LAT annotation
        lateralAngle : float
            lateral angle in °(?) at LAT annotation
        time : ndarray
            time stamps in 'point time' as type np.int32
        timeForce : ndarray
            force values in g(?) recorded over time as type np.float32
        timeAxialAngle : ndarray
            axial angle in °(?) recorded over time as type np.float32
        timeLateralAngle : ndarray
            lateral angle in °(?) recorded over time as type np.float32
        systemTime : ndarray
            system time stamps as type np.int32
    """

    def __init__(
            self,
            force: float = np.nan,
            axial_angle: float = np.nan,
            lateral_angle: float = np.nan,
            t_time: np.ndarray = np.empty(0, dtype=np.int32),
            t_force: np.ndarray = np.empty(0, dtype=np.float32),
            t_axial_angle: np.ndarray = np.empty(0, dtype=np.float32),
            t_lateral_angle: np.ndarray = np.empty(0, dtype=np.float32),
            system_time: np.ndarray = np.empty(0, dtype=np.int32)
    ) -> None:
        """Constructor."""

        self.force = force
        self.axialAngle = axial_angle
        self.lateralAngle = lateral_angle
        self.time = t_time
        self.timeForce = t_force
        self.timeAxialAngle = t_axial_angle
        self.timeLateralAngle = t_lateral_angle
        self.systemTime = system_time


class PointImpedance:
    """
    A class representing impedance data recorded with a Carto point.

    Attributes:
        time : ndarray
            time stamps in 'point time' as type np.int32
        value : ndarray
            impedance values in Ohm(?) recorded over time as type np.float32
    """

    def __init__(
            self,
            time: np.ndarray = np.empty(0, dtype=np.int32),
            value: np.ndarray = np.empty(0, dtype=np.float32)
    ) -> None:
        """Constructor."""

        self.time = time
        self.value = value
