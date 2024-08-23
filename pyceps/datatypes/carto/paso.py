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
import re
from datetime import datetime
from typing import IO, List, TypeVar, Optional
import xml.etree.ElementTree as ET
import numpy as np

from pyceps.datatypes.signals import Trace
from pyceps.fileio.xmlio import xml_add_binary_trace, xml_load_binary_trace


TPaSo = TypeVar('TPaSo', bound='PaSo')  # workaround to type hint self


log = logging.getLogger(__name__)


PasoTable = namedtuple('PasoTable', 'ISName')
PasoTable.__doc__ = """
A namedtuple representing PaSo information from study XML.

Fields:
    ISName : str
"""

PaSoConfiguration = namedtuple('PaSoConfiguration',
                               [
                                   'isisCorrelationThreshold',
                                   'pmisCorrelationThreshold',
                                   'isisMinCorrelatedChannels',
                                   'pmisMinCorrelatedChannels',
                                   'isDefaultPrefix',
                                   'pmDefaultPrefix'
                               ])
PaSoConfiguration.__doc__ = """
A namedtuple representing PaSo module configuration.

Fields:
    isisCorrelationThreshold : float
    pmisCorrelationThreshold : float
    isisMinCorrelatedChannels : int
    pmisMinCorrelatedChannels : int
    isDefaultPrefix : str
    pmDefaultPrefix : str
"""


class PaSoTemplate:
    """
    Class representing a PaSo module VT Template.

    Attributes:
        ID : int
        name : str
        date : datetime
            creation date, format %Y-%m-%d %H:%M:%S
        currentMatched : int
        bestMatched : int
        cycleLength : int
        timestamp : list of int
        intervalStart : int
        currentWOI : list of int
        isReference : bool
        ecg : list of Trace
    """

    def __init__(
            self
    ) -> None:
        """Constructor."""

        self.ID = -1
        self.name = ''
        self.date = ''
        self.currentMatched = -1
        self.bestMatched = -1
        self.cycleLength = -1
        self.timestamp = [-1, -1]
        self.intervalStart = -1
        self.currentWOI = [-1, -1]
        self.isReference = False
        self.ecg = []

    def is_reference(
            self
    ) -> bool:
        """
        Check if this template was used as reference.

        Returns:
            bool
        """

        return self.isReference

    def load(
            self,
            fid: IO,
            encoding: str = 'cp1252'
    ) -> None:
        """
        Load template data from file. Data is added to attribute "ecg".

        Parameters:
            fid: file-like
                file object pointing to PaSo configuration file
            encoding: str
                file encoding used for binary files

        Returns:
            None
        """

        log.debug('loading PaSo configuration from file {}'.format(fid.name))

        line = fid.readline().decode(encoding=encoding).rstrip()
        skip_rows = 1

        while not line.rstrip() == 'ECG:':
            if line.startswith('ID:'):
                self.ID = int(line.split('ID: ')[1])

            if line.startswith('Name:'):
                self.name = line.split('Name: ')[1]

            if line.startswith('Year:'):
                regex = re.compile(
                    r"""Year:\s*(\d+)
                    \s*Month:\s*(\d+)
                    \s*Day:\s*(\d+)
                    \s*(\d+:\d+:\d+)""",
                    re.VERBOSE
                )
                date_str = regex.findall(line)
                if not date_str:
                    print('unknown date format!')
                    continue

                date_str = '-'.join(date_str[0])
                self.date = datetime.strptime(date_str, "%Y-%m-%d-%H:%M:%S")

            if line.startswith('Current Matched'):
                regex = re.compile(
                    r""".*:\s*(-?\d+).*:\s*(-?\d+)""",
                    re.VERBOSE
                )
                matches = regex.findall(line)
                if not matches:
                    print('unknown matches section!')
                    continue
                self.currentMatched = int(matches[0][0])
                self.bestMatched = int(matches[0][1])

            if line.startswith('Cycle Length'):
                regex = re.compile(
                    r""".*:\s*(\d+)""",
                    re.VERBOSE
                )
                matches = regex.findall(line)
                if not matches:
                    print('unknown cycle length section!')
                    continue
                self.cycleLength = int(matches[0])

            if line.startswith('Timestamp'):
                regex = re.compile(
                    r""".*:\s*(\d+).*:\s*(\d+)""",
                    re.VERBOSE
                )
                matches = regex.findall(line)
                if not matches:
                    print('unknown timestamp section!')
                    continue
                self.timestamp = [int(matches[0][0]), int(matches[0][1])]

            if line.startswith('Interval'):
                regex = re.compile(r""".*:\s*(\d+)""", re.VERBOSE)
                matches = regex.findall(line)
                if not matches:
                    print('unknown interval section!')
                    continue
                self.intervalStart = int(matches[0])

            if line.startswith('Current WOI'):
                regex = re.compile(r""".*:\s*(\d+)\s*(\d+)""", re.VERBOSE)
                matches = regex.findall(line)
                if not matches:
                    print('unknown WOI section!')
                    continue
                self.currentWOI = [int(matches[0][0]), int(matches[0][1])]

            if line.startswith('Reference IS'):
                self.isReference = True
            if line.startswith('Non-reference IS'):
                self.isReference = False

            # continue with next line
            line = fid.readline().decode(encoding=encoding).rstrip()
            skip_rows += 1

        # read complete header, now load ECG data
        ecg_names = fid.readline().decode(encoding=encoding).rstrip().split()
        skip_rows += 1

        # read data, fixed length columns
        data = np.genfromtxt(fid,
                             delimiter=[5] * len(ecg_names),
                             skip_header=skip_rows
                             )

        # build traces
        traces = []
        for i, name in enumerate(ecg_names):
            traces.append(
                Trace(
                    name=name,
                    data=data[:, i],
                    fs=1000.0
                )
            )

        self.ecg = traces


class PaSoCorrelation:
    """
    Class representing PaSo module correlation.

    Attributes:
        type : str
        ID : int
        correlatedTo : int
        UserAverage : float
        UserChannels : list of float
            correlation for each channel
        UserWOI : list of int
        SystemAverage : float
        SystemChannels : list of float
            correlation for each channel
        SystemWOI : list of int
    """

    def __init__(
            self
    ) -> None:
        """Constructor."""

        self.type = ''
        self.ID = -1
        self.correlatedTo = -1
        self.UserAverage = np.nan
        self.UserChannels = []
        self.UserWOI = [-1, -1]
        self.SystemAverage = np.nan
        self.SystemChannels = []
        self.SystemWOI = [-1, -1]

    def calc_user_average(
            self
    ) -> float:
        """
        Calculate mean correlation from user channels.

        Returns:
            float : mean value
                nan if no channel correlations found
        """

        if self.UserChannels:
            return np.mean(self.UserChannels)

        return np.nan

    def calc_system_average(
            self
    ) -> float:
        """
        Calculate mean correlation from system channels.

        Returns:
            float : mean value
                nan if no channel correlations
        """

        if self.SystemChannels:
            return np.mean(self.SystemChannels)

        return np.nan


class PaSo:
    """
    A class representing PaSo module data.

    Attributes:
        Configuration : PaSoConfiguration
        Templates : list of PaSoTemplate
        Correlations : list of PaSoCorrelation
    """

    def __init__(
            self,
            configuration: PaSoConfiguration,
            templates: List[PaSoTemplate],
            correlations: List[PaSoCorrelation]
    ) -> None:
        """
        Constructor.

        Parameters:
            configuration : PaSoConfiguration
            templates : list of PaSoTemplate
            correlations : list of PaSoCorrelation
        """

        self.Configuration = configuration
        self.Templates = templates
        self.Correlations = correlations

    def add_to_xml(
            self,
            root: ET.Element,
            **kwargs
    ) -> None:
        """
        Add PaSo data to XML.

        XML attributes:
        count : number of ablation sites

        Example:
            <PaSo
                isisCorrelationThreshold="0.8"
                pmisCorrelationThreshold="0.8"
                isisMinCorrelatedChannels="10"
                pmisMinCorrelatedChannels="10"
                isDefaultPrefix="IS"
                pmDefaultPrefix="PM"
                <Templates
                    count=1
                    <Template
                        ID="1"
                        name="IS"
                        date="2021-03-22 09:13:38"
                        currentMatched="-1"
                        bestMatched="1"
                        cycleLength="942"
                        timestamp="4137695 4200195"
                        intervalStart="4197271"
                        currentWOI="4198371 4198671"
                        isReference="False"
                        <Traces
                            name="ECG"
                            count="1"
                            <Trace>
                                <DataArray/>
                                <DataArray/>
                                ...
                            </>
                        </>
                    </>
                </>
                <Correlations
                    count=1
                    <Correlation
                        ID="1"
                        correlatedTo="1"
                        type="ISIS"
                        UserAverage="1.0"
                        UserChannels="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
                        UserWOI="4198371 4198671"
                        SystemAverage="1.0"
                        SystemChannels="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
                        SystemWOI="4198371 4198671"
                    </>
                </>
            </>

        Data is saved as base64 encoded bytes string.
        Extra attributes can be added by keyword arguments.

        Parameters:
            root : eTree.Element
                XML element to which data is added

        Returns: None
        """

        paso_item = ET.SubElement(
            root, 'PaSo',
            isisCorrelationThreshold=str(
                self.Configuration.isisCorrelationThreshold
            ),
            pmisCorrelationThreshold=str(
                self.Configuration.pmisCorrelationThreshold
            ),
            isisMinCorrelatedChannels=str(
                self.Configuration.isisMinCorrelatedChannels
            ),
            pmisMinCorrelatedChannels=str(
                self.Configuration.pmisMinCorrelatedChannels
            ),
            isDefaultPrefix=str(self.Configuration.isDefaultPrefix),
            pmDefaultPrefix=str(self.Configuration.pmDefaultPrefix),
        )

        # add extra attributes
        for key, value in kwargs:
            paso_item.set(key, value)

        templates_item = ET.SubElement(paso_item, 'Templates',
                                       count=str(len(self.Templates))
                                       )
        for template in self.Templates:
            template_item = ET.SubElement(
                templates_item, 'Template',
                ID=str(template.ID),
                name=template.name,
                date=str(template.date),
                currentMatched=str(template.currentMatched),
                bestMatched=str(template.bestMatched),
                cycleLength=str(template.cycleLength),
                timestamp=' '.join([str(x) for x in template.timestamp]),
                intervalStart=str(template.intervalStart),
                currentWOI=' '.join([str(x) for x in template.currentWOI]),
                isReference=str(template.isReference)
            )
            xml_add_binary_trace(template_item, 'ECG', template.ecg)

        correlations_item = ET.SubElement(
            paso_item, 'Correlations',
            count=str(len(self.Correlations))
        )
        for correlation in self.Correlations:
            ET.SubElement(
                correlations_item, 'Correlation',
                ID=str(correlation.ID),
                correlatedTo=str(correlation.correlatedTo),
                type=correlation.type,
                UserAverage=str(correlation.UserAverage),
                UserChannels=' '.join([str(x)
                                       for x in correlation.UserChannels
                                       ]),
                UserWOI=' '.join([str(x) for x in correlation.UserWOI]),
                SystemAverage=str(correlation.SystemAverage),
                SystemChannels=' '.join([str(x)
                                         for x in correlation.SystemChannels
                                         ]),
                SystemWOI=' '.join([str(x)
                                    for x in correlation.SystemWOI
                                    ]),
            )

    @classmethod
    def load_from_xml(
            cls,
            element: ET.Element
    ) -> Optional[TPaSo]:
        """
        Load PaSo data from XML.

        Parameters:
            element : eTree.Element
                XML element from which data is loaded

        Returns: PaSo, None
        """

        if not element.tag == 'PaSo':
            log.warning('cannot import PaSo from XML element {}!'
                        .format(element.tag)
                        )
            return None

        # load PaSo configuration
        paso_config = PaSoConfiguration(
            isisCorrelationThreshold=float(
                element.get('isisCorrelationThreshold')
            ),
            pmisCorrelationThreshold=float(
                element.get('pmisCorrelationThreshold')
            ),
            isisMinCorrelatedChannels=int(
                element.get('isisMinCorrelatedChannels')
            ),
            pmisMinCorrelatedChannels=int(
                element.get('pmisMinCorrelatedChannels')
            ),
            isDefaultPrefix=element.get('isDefaultPrefix'),
            pmDefaultPrefix=element.get('pmDefaultPrefix'),
        )

        # load PaSo templates
        templates = []
        templates_item = element.find('Templates')
        num_templates = int(templates_item.get('count'))
        for template in templates_item.iter('Template'):
            new = PaSoTemplate()
            new.ID = int(template.get('ID'))
            new.name = template.get('name')
            new.date = datetime.strptime(
                template.get('date'), "%Y-%m-%d %H:%M:%S"
            )
            new.currentMatched = int(template.get('currentMatched'))
            new.bestMatched = int(template.get('bestMatched'))
            new.cycleLength = int(template.get('cycleLength'))
            new.timestamp = [int(x)
                             for x in template.get('timestamp').split()
                             ]
            new.intervalStart = int(template.get('intervalStart'))
            new.currentWOI = [int(x)
                              for x in template.get('currentWOI').split()
                              ]
            new.isReference = template.get('isReference') == 'True'
            _, new.ecg = xml_load_binary_trace(template.find('Traces'))

            templates.append(new)

        # final sanity check
        if not len(templates) == num_templates:
            log.warning('number of templates expected {} differs from '
                        'templates loaded {}!'
                        .format(num_templates, len(templates)))

        # load PaSo correlations
        correlations = []
        templates_item = element.find('Correlations')
        num_correlations = int(templates_item.get('count'))
        for correlation in templates_item.iter('Correlation'):
            new = PaSoCorrelation()

            new.type = correlation.get('type')
            new.ID = int(correlation.get('ID'))
            new.correlatedTo = int(correlation.get('correlatedTo'))
            new.UserAverage = float(correlation.get('UserAverage'))
            new.UserChannels = [
                float(x) for x in correlation.get('UserChannels').split()
            ]
            new.UserWOI = [
                int(x) for x in correlation.get('UserWOI').split()
            ]
            new.SystemAverage = float(correlation.get('SystemAverage'))
            new.SystemChannels = [
                float(x) for x in correlation.get('SystemChannels').split()
            ]
            new.SystemWOI = [
                int(x) for x in correlation.get('SystemWOI').split()
            ]

            correlations.append(new)

        # final sanity check
        if not len(correlations) == num_correlations:
            log.warning('number of correlations expected {} differs from '
                        'correlations loaded {}!'
                        .format(num_correlations, len(correlations)))

        return cls(
            configuration=paso_config,
            templates=templates,
            correlations=correlations
        )
