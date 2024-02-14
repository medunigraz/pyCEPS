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
import numpy as np
from scipy import integrate


CartoUnits = namedtuple('CartoUnits', ['Distance', 'Angle'])

Coloring = namedtuple('Coloring', ['Id',
                                   'Name',
                                   'TextureInvert',
                                   'Propagation',
                                   'Units'])

ColoringRange = namedtuple('ColoringRange', ['Id', 'min', 'max'])

SurfaceAttributes = namedtuple('SurfaceAttributes',
                               ['name', 'description', 'value'])

SurfaceErrorTable = namedtuple('SurfaceErrorTable',
                               ['BadErrorColor',
                                'MedErrorColor',
                                'GoodErrorColor',
                                'BadErrorThreshold',
                                'MedErrorThreshold',
                                'GoodErrorThreshold'])

PasoTable = namedtuple('PasoTable', 'ISName')

CFAEColoringTable = namedtuple('CFAEColoringTable',
                               ['IgnoreBelowColor',
                                'IclMediumColor',
                                'IclHighColor',
                                'IgnoreBelowThreshold',
                                'IclMediumThreshold',
                                'IclHighThreshold'])

Tag = namedtuple('Tag', ['ID',
                         'Short_Name',
                         'Full_Name',
                         'Color',
                         'Radius'])

CartoMappingParameters = namedtuple('CartoMappingParameters',
                                    ['ColoringTable',
                                     'SurfaceErrorTable',
                                     'PasoTable',
                                     'CFAEColoringTable',
                                     'TagsTable'])

RefAnnotationConfig = namedtuple('RefAnnotationConfig',
                                 ['algorithm', 'connector'])


class RFAblationParameters:
    """A class representing ablation data."""

    def __init__(self,
                 time=np.empty(0, dtype=np.int32),
                 power=np.empty(0, dtype=np.int32),
                 impedance=np.empty(0, dtype=np.int32),
                 distal_temp=np.empty(0, dtype=np.int32)):
        """Constructor."""

        self.time = time
        self.power = power
        self.impedance = impedance
        self.distalTemp = distal_temp


class MapRF:
    """A class representing force and ablation data."""

    def __init__(self,
                 force=None,
                 ablation_parameters=None):
        """Constructor."""

        self.force = force
        self.ablParams = ablation_parameters


class RFForce:
    """A class representing force data."""

    def __init__(self,
                 time=np.empty(0, dtype=np.int32),
                 force=np.empty(0, dtype=np.int32),
                 axial_angle=np.empty(0, dtype=np.int32),
                 lateral_angle=np.empty(0, dtype=np.int32),
                 abl_point=np.empty(0, dtype=np.int32),
                 position=np.empty((0, 3), dtype=np.float32)):
        """Constructor."""

        self.time = time
        self.force = force
        self.axialAngle = axial_angle
        self.lateralAngle = lateral_angle
        self.ablationPoint = abl_point
        self.position = position


class PointForces:
    """A class representing contact force data."""

    def __init__(self,
                 force=np.empty(0, dtype=float),
                 axial_angle=np.empty(0, dtype=float),
                 lateral_angle=np.empty(0, dtype=float),
                 t_time=np.empty((0, 0), dtype=float),
                 t_force=np.empty((0, 0), dtype=float),
                 t_axial_angle=np.empty((0, 0), dtype=float),
                 t_lateral_angle=np.empty((0, 0), dtype=float),
                 system_time=np.empty((0, 0), dtype=float)):
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
    """A class representing an impedance trace for a mapping point."""

    def __init__(self,
                 time=None,
                 value=None):
        """Constructor."""

        self.time = time
        self.value = value


class Visitag:
    """A class representing Carto 3 Visitag data."""

    def __init__(self,
                 sites=None,
                 grid=None):
        """Constructor."""

        self.sites = sites
        self.grid = grid


class VisitagAblationSite:
    """A class representing ablation tags from Carto3 Visitag module."""

    def __init__(self,
                 site_index,
                 session_index=None,
                 channel_id=None,
                 tag_index_status=None,
                 coordinates=np.full(3, np.nan, dtype=float),
                 avg_force=np.nan,
                 fti=np.nan,
                 max_power=np.nan,
                 max_temp=np.nan,
                 duration=np.nan,
                 base_impedance=np.nan,
                 impedance_drop=np.nan,
                 rf_index=None):
        """Constructor."""

        self.siteIndex = site_index
        self.sessionIndex = session_index
        self.channelID = channel_id
        self.tagIndexStatus = tag_index_status

        self.X = coordinates
        self.avgForce = avg_force
        self.fti = fti
        self.maxPower = max_power
        self.maxTemp = max_temp
        self.duration = duration
        self.baseImp = base_impedance
        self.impDrop = impedance_drop

        self.RFIndex = []
        self.add_rf_index(rf_index)

    def add_rf_index(self, rf_index):
        """Add an RF index to this ablation site."""

        if not rf_index:
            return

        if (issubclass(type(rf_index), list)
                and all(isinstance(x, VisitagRFIndex) for x in rf_index)):
            self.RFIndex.extend(rf_index)
        elif isinstance(rf_index, VisitagRFIndex):
            self.RFIndex.append(rf_index)
        else:
            raise TypeError('argument rf_index must be of type {}'
                            .format(type(VisitagRFIndex)))


class VisitagAblationGrid:
    """A class representing the ablation grid of CARTO3 Visitag module."""

    def __init__(self, site_index,
                 session=None,
                 points=None,
                 first_pos_time_stamp=-1,
                 first_pos_passed_filter_time_stamp=-1,
                 last_pos_time_stamp=-1):
        """Constructor."""

        self.siteIndex = site_index
        self.session = session
        self.firstPosTimeStamp = first_pos_time_stamp
        self.firstPosPassedFilterTimeStamp = first_pos_passed_filter_time_stamp
        self.lastPosTimeStamp = last_pos_time_stamp

        self.points = []
        self.add_points(points)

    def add_points(self, points):
        """Add VisitagGridPoints to this ablation grid."""

        if not points:
            return

        if (issubclass(type(points), list)
                and all(isinstance(x, VisitagGridPoint) for x in points)):
            self.points.extend(points)
        elif isinstance(points, VisitagGridPoint):
            self.points.append(points)
        else:
            raise TypeError('argument points must be of type {}'
                            .format(type(VisitagGridPoint)))

    def calc_rfi(self):
        """Calculate RF index for this grid."""

        rf_name = 'CustomRFI'
        rfi_trace = self.build_rfi_evolution()

        if rfi_trace.size == 0:
            return VisitagRFIndex(name=rf_name, value=np.nan)

        return VisitagRFIndex(name=rf_name, value=rfi_trace[-1])

    def build_rfi_evolution(self):
        """Calculate the evolution of RF index based on CARTO's formula."""

        # TODO: retrieve values from fpti-formulas.xml
        k = 975.00
        a = 0.68
        b = 1.63
        c = 0.35
        dt = 1E-3  # time resolution of VisiTag Module

        # get data from all points in grid
        force = np.array([])
        power = np.array([])
        time = np.array([])
        for point in self.points:
            force = np.append(force, point.force)
            power = np.append(power, point.power)
            time = np.append(time, point.time)

        # TODO: implement thresholding and clipping criteria

        # sort data
        p = np.argsort(time)
        time = time[p]
        force = force[p]
        power = power[p]

        # calculate RFIndex
        CFP = np.multiply(np.power(force, a), np.power(power, b))
        # TODO: find better initial value
        iCFP = integrate.cumulative_trapezoid(CFP, time*dt, initial=0)

        return np.power(k * iCFP, c)


class VisitagGridPoint:
    """A class representing grid data from Carto3 Visitag module."""

    def __init__(self,
                 coordinates=np.full(3, np.nan, dtype=float),
                 time=np.empty(0, dtype=int),
                 temperature=np.empty(0, dtype=float),
                 power=np.empty(0, dtype=float),
                 force=np.empty(0, dtype=float),
                 axial_angle=np.empty(0, dtype=float),
                 lateral_angle=np.empty(0, dtype=float),
                 base_impedance=np.empty(0, dtype=float),
                 impedance_drop=np.empty(0, dtype=float),
                 rf_index=None,
                 passed=np.empty(0, dtype=int)
                 ):
        """Constructor."""

        self.X = coordinates
        self.time = time
        self.temperature = temperature
        self.power = power
        self.force = force
        self.axialAngle = axial_angle
        self.lateralAngle = lateral_angle
        self.baseImp = base_impedance
        self.impDrop = impedance_drop
        self.passed = passed
        self.RFIndex = []
        self.add_rf_index(rf_index)

    def add_rf_index(self, rf_index):
        """Add an RF index to this ablation site."""

        if (issubclass(type(rf_index), list)
                and all(isinstance(x, VisitagRFIndex) for x in rf_index)):
            self.RFIndex = self.RFIndex.extend(rf_index)
        elif isinstance(rf_index, VisitagRFIndex):
            self.RFIndex.append(rf_index)
        else:
            raise TypeError('argument rf_index must be of type {}'
                            .format(type(VisitagRFIndex)))


VisitagRFIndex = namedtuple('VisitagRFIndex', ['name', 'value'])
