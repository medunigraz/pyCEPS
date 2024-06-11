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
from typing import Optional, List, Union
import numpy as np
from scipy import integrate

from pyceps.datatypes.lesions import Lesions, AblationSite, RFIndex


log = logging.getLogger(__name__)


class VisitagAblationSite:
    """
    A class representing ablation tags from Carto3 Visitag module.

    Attributes:
        siteIndex : int
        sessionIndex : int
        channelID : int
        tagIndexStatus : int
        X : ndarray (3,)
            coordinates of the site with np.float32 type.
        avgForce : numpy.float32
        fti : numpy.float32
        maxPower : numpy.float32
        maxTemp : numpy.float32
        duration : numpy.float32
        baseImp : numpy.float32
        impDrop : numpy.float32
        RFIndex : list of RFIndex
    """

    def __init__(
            self,
            site_index: int,
            session_index: int = np.iinfo(np.int32),
            channel_id: int = np.iinfo(np.int32),
            tag_index_status: int = np.iinfo(np.int32),
            coordinates=np.full(3, np.nan, dtype=np.float32),
            avg_force: float = np.nan,
            fti: float = np.nan,
            max_power: float = np.nan,
            max_temp: float = np.nan,
            duration: float = np.nan,
            base_impedance: float = np.nan,
            impedance_drop: float = np.nan,
            rf_index: Optional[Union[RFIndex, List[RFIndex]]] = None
    ) -> None:
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

    def add_rf_index(
            self,
            rf_index: Union[RFIndex, List[RFIndex]]
    ) -> None:
        """
        Add an RF index to this ablation site.

        Parameter:
            rf_index : RFIndex, list of RFIndex

        Returns:
            None
        """

        if not rf_index:
            return

        if (issubclass(type(rf_index), list)
                and all(isinstance(x, RFIndex) for x in rf_index)):
            self.RFIndex.extend(rf_index)
        elif isinstance(rf_index, RFIndex):
            self.RFIndex.append(rf_index)
        else:
            raise TypeError('argument rf_index must be of type {}'
                            .format(type(RFIndex)))


class VisitagGridPoint:
    """
    A class representing grid points aggregated to an ablation site.

    Attributes:
        X : ndarray (n,3)
            coordinates of n points forming the grid as type np.float32.
        time : ndarray (t,1)
            time stamps as type np.int32
        temperature : ndarray (t, )
            temperature in °C as type np.float32
        power : ndarray (t, )
            delivered power in W as type np.float32
        force : ndarray (t, )
            contact force in ?N as type np.float32
        axialAngle : ndarray (t, )
            catheter angle in rad as type np.float32
        lateralAngle : ndarray (t, )
            catheter angle in rad as type np.float32
        baseImp : ndarray (t, )
            impedance in Ohm(?) as type np.float32
        impDrop : ndarray (t, )
            impedance drop in Ohm(?) as type np.float32
        passed : ndarray (t, )
            status as type np.int32
        RFIndex : list of RFIndex
            ablation metrics
    """

    def __init__(
            self,
            coordinates: np.ndarray = np.full(3, np.nan, dtype=float),
            time: np.ndarray = np.empty(0, dtype=int),
            temperature: np.ndarray = np.empty(0, dtype=float),
            power: np.ndarray = np.empty(0, dtype=float),
            force: np.ndarray = np.empty(0, dtype=float),
            axial_angle: np.ndarray = np.empty(0, dtype=float),
            lateral_angle: np.ndarray = np.empty(0, dtype=float),
            base_impedance: np.ndarray = np.empty(0, dtype=float),
            impedance_drop: np.ndarray = np.empty(0, dtype=float),
            rf_index: Optional[Union[RFIndex, List[RFIndex]]] = None,
            passed: np.ndarray = np.empty(0, dtype=int)
    ) -> None:
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

    def add_rf_index(
            self,
            rf_index: Union[RFIndex, List[RFIndex]]
    ) -> None:
        """
        Add an RF index to this ablation grid.

        Parameter:
            rf_index : RFIndex, list of RFIndex

        Returns:
            None
        """

        if not rf_index:
            return

        if (issubclass(type(rf_index), list)
                and all(isinstance(x, RFIndex) for x in rf_index)):
            self.RFIndex = self.RFIndex.extend(rf_index)
        elif isinstance(rf_index, RFIndex):
            self.RFIndex.append(rf_index)
        else:
            raise TypeError('argument rf_index must be of type {}'
                            .format(type(RFIndex)))


class VisitagAblationGrid:
    """
    A class representing the ablation grid of CARTO3 Visitag module.

    Attributes:
        siteIndex : int
        session : int
        firstPosTimeStamp : int
        firstPosPassedFilterTimeStamp : int
        lastPosTimeStamp : int
        points : list of VisitagGridPoint
    """

    def __init__(
            self,
            site_index: int,
            session: int = np.iinfo(np.int32),
            points: Optional[
                Union[VisitagGridPoint, List[VisitagGridPoint]]
            ] = None,
            first_pos_time_stamp: int = -1,
            first_pos_passed_filter_time_stamp: int = -1,
            last_pos_time_stamp: int = -1
    ) -> None:
        """Constructor."""

        self.siteIndex = site_index
        self.session = session
        self.firstPosTimeStamp = first_pos_time_stamp
        self.firstPosPassedFilterTimeStamp = first_pos_passed_filter_time_stamp
        self.lastPosTimeStamp = last_pos_time_stamp

        self.points = []
        self.add_points(points)

    def add_points(
            self,
            points: Union[VisitagGridPoint, List[VisitagGridPoint]]
    ) -> None:
        """
        Add VisitagGridPoints to this ablation grid.

        Parameters:
            points : VisitagGridPoint, list of VisitagGridPoint

        Returns:
            None
        """

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

    def calc_rfi(
            self
    ) -> RFIndex:
        """
        Calculate RF index for this grid.

        Returns:
            RFIndex with name "CustomRFI"
        """

        rf_name = 'CustomRFI'
        rfi_trace = self.build_rfi_evolution()

        if rfi_trace.size == 0:
            return RFIndex(name=rf_name, value=np.nan)

        return RFIndex(name=rf_name, value=rfi_trace[-1])

    def build_rfi_evolution(
            self
    ) -> np.ndarray:
        """
        Calculate the evolution of RF index based on CARTO3 formula.

        Returns:
            ndarray
        """

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


class Visitag:
    """
    A class representing Carto 3 Visitag data.

    Attributes:
        sites : list of VisitagAblationSite
        grid : VisitagAblationGrid
    """

    def __init__(
            self,
            sites: Optional[List[VisitagAblationSite]] = None,
            grid: Optional[List[VisitagAblationGrid]] = None
    ) -> None:
        """
        Constructor.

        Parameters:
            sites : list of VisitagAblationSite
            grid : VisitagAblationGrid
        """

        self.sites = sites
        self.grid = grid

    def to_lesions(
            self
    ) -> Lesions:
        """
        Convert VisiTags to base class lesions.

        Returns:
            Lesions
        """

        lesions = []
        for site in self.sites:
            rfi = [RFIndex(name=x.name, value=x.value) for x in site.RFIndex]
            lesions.append(
                AblationSite(X=site.X, diameter=6.0, RFIndex=rfi)
            )

        return Lesions(lesions)
