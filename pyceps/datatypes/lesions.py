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
from collections import namedtuple
from typing import List, Tuple, Union, Optional, TypeVar
import xml.etree.ElementTree as ET
import numpy as np

from pyceps.fileio.xmlio import xml_load_binary_data, xml_add_binary_numpy

TLesions = TypeVar('TLesions', bound='Lesions')  # workaround to type hint self


log = logging.getLogger(__name__)


RFIndex = namedtuple('RFIndex', ['name', 'value'])
RFIndex.__doc__ = """
A namedtuple to represent RF quality metrics.

Fields:
    name : str
    value : np.float32
"""

AblationSite = namedtuple('AblationSite', ['X', 'diameter', 'RFIndex'])
AblationSite.__doc__ = """
A namedtuple to represent a RF ablation site.

Fields:
    X : ndarray (3,)
    diameter : float
    RFIndex : RFIndex
"""


class Lesions:
    """
    A class representing ablation lesion data.

    Attributes:
        sites : list of AblationSite
    """

    XML_IDENTIFIER = 'Lesions'

    def __init__(
            self,
            sites: List[AblationSite]
    ) -> None:
        """Constructor."""

        self.sites = sites

    def get_rfi_names(
            self,
            return_counts=False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return unique RF parameter names in lesions data.

        Parameters:
            return_counts : bool
                number of times each of the unique values comes up

        Returns:
            names : ndarray
            counts : ndarray, optional
        """

        names = [x.name for lesion in self.sites for x in lesion.RFIndex]
        names, counts = np.unique(names, return_counts=True)

        if return_counts:
            return names, counts

        return names

    def add_to_xml(
            self, root: ET.Element,
            **kwargs
    ) -> None:
        """
        Add lesion data to XML.

        XML attributes:
            count : number of ablation sites

        Example:
            <Lesion
                count=num_lesions>
                <DataArray name="points"/>
                <DataArray name="diameter"/>
                <DataArray name="RFI"/>
                <DataArray name="name"/>
            </>

        Data is saved as base64 encoded bytes string.
        Extra attributes can be added by keyword arguments.

        Parameters:
            root : eTree.Element
                XML element to which data is added

        Returns:
            None
        """

        element = ET.SubElement(root, Lesions.XML_IDENTIFIER,
                                count=str(len(self.sites))
                                )
        # add extra attributes
        for key, value in kwargs:
            element.set(key, value)

        # add data
        xml_add_binary_numpy(element, 'points',
                             np.array([site.X for site in self.sites])
                             )
        xml_add_binary_numpy(element, 'diameter',
                             np.array([site.diameter for site in self.sites])
                             )
        xml_add_binary_numpy(element, 'RFI',
                             np.array([x.value for site in self.sites
                                       for x in site.RFIndex])
                             )
        xml_add_binary_numpy(element, 'name',
                             np.array([x.name for site in self.sites
                                       for x in site.RFIndex])
                             )

    @classmethod
    def load_from_xml(
            cls,
            element: ET.Element
    ) -> Optional[TLesions]:
        """
        Load binary Lesions object from etree element.

        Parameters:
            element : ET.Element
                XML Element to read data from

        Returns:
            Lesions, None
        """

        if not element.tag == Lesions.XML_IDENTIFIER:
            log.warning('cannot import Lesion from XML element {}!'
                        .format(element.tag)
                        )
            return None

        count = int(element.get('count'))
        lesions = []

        l_data = {}
        for arr in element.findall('DataArray'):
            name, data = xml_load_binary_data(arr)
            l_data[name] = data

        for i in range(count):
            rfi = []
            if isinstance(l_data['RFI'][i], np.ndarray):
                for k in range(l_data['RFI'][i].shape[0]):
                    rfi.append(RFIndex(name=l_data['name'][i][k],
                                       value=l_data['RFI'][i][k]))
            else:
                rfi.append(RFIndex(name=l_data['name'][i],
                                   value=l_data['RFI'][i])
                           )

            lesions.append(
                AblationSite(X=l_data['points'][i],
                             diameter=l_data['diameter'][i],
                             RFIndex=rfi)
            )

        return cls(sites=lesions)
