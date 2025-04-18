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


Trace = namedtuple('Trace', ['name', 'data', 'fs'])
Trace.__doc__ = """
A namedtuple representing a signal trace.

Fields:
    name : str
    data : ndarray (t, )
        signal data as type np.float32
    fs : float
"""

BodySurfaceECG = namedtuple('BodySurfaceECG',
                            ['method', 'refAnnotation', 'traces']
                            )
BodySurfaceECG.__doc__ = """
A namedtuple representing body surface ECG traces for a mapping procedure.

Fields:
    method : str
    refAnnotation : int
    traces : list of Trace
"""
