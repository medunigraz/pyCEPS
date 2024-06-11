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

from typing import List, Union, Optional
import xml.etree.ElementTree as ET
import base64
import numpy as np

from pyceps.datatypes.signals import Trace, BodySurfaceECG


def xml_add_binary_numpy(
        root: ET.Element,
        name: str,
        data: np.ndarray,
        **kwargs
) -> None:
    """
    Create etree DataArray with binary numpy data.

    XML attributes:
        type : np.dtype extracted from data
        name : name of the DataArray
        numberOfComponents : dimension of data along axis=1
        format : always "binary"

        Example:
            <DataArray
                type=np.dtype
                name=name,
                numberOfComponents=n
                format="binary">
                DATA
            </>

    Data is saved as base64 encoded bytes string.
    Extra attributes can be added by keyword arguments.

    Parameters:
        root : ET.Element
            XML Element the data is added to
        name : str
            name of the data array
        data : np.ndarray
            data to be added

    Returns:
        None
    """

    try:
        numComponents = str(data.shape[1])
    except IndexError:
        numComponents = '1'

    # create Element
    element = ET.SubElement(root, 'DataArray',
                            type=data.dtype.str,
                            name=name,
                            numberOfComponents=numComponents,
                            format='binary',
                            )
    # add data
    element.text = base64.b64encode(data.tobytes()).decode('utf-8')

    # add extra attributes
    for key, value in kwargs:
        element.set(key, value)


def xml_add_binary_trace(
        root: ET.Element,
        name: str,
        data: Union[Trace, List[Trace]],
        **kwargs
) -> None:
    """
    Create etree Trace with binary data.

    XML attributes:
        name : name of the DataArray
        count : number of traces in data

        Example:
            <Traces
                name=name,
                count=numTraces>
                <Trace>
                    <DataArray/>
                    <DataArray/>
                    ...
                </>
            </>

    For each trace DataArrays for trace names ("name"), sampling frequency (
    "fs"), and data ("data") are added.
    Data is saved trace-wise with dimension (n_components x n_points).
    Data is saved as base64 encoded bytes string.
    Extra attributes can be added by keyword arguments.

    Parameters:
        root : ET.Element
            XML Element the data is added to
        name : str
            name of the data array
        data : Trace, list of Trace
            data to be added

    Returns:
        None
    """

    if not any(isinstance(el, list) for el in data):
        # single trace for each point, make list of lists to process
        data = [data]
    else:
        # multiple traces for each point, transpose to save trace-wise
        data = list(zip(*data))

    traces = ET.SubElement(root, 'Traces',
                           name=name,
                           count=str(len(data)))

    # add extra attributes
    for key, value in kwargs:
        traces.set(key, value)

    for d in data:
        trace = ET.SubElement(traces, 'Trace')
        xml_add_binary_numpy(trace,
                             'name',
                             np.array([t.name for t in d])
                             )
        xml_add_binary_numpy(trace,
                             'fs',
                             np.array([t.fs for t in d]).astype(np.float32)
                             )
        xml_add_binary_numpy(trace,
                             'data',
                             np.array([t.data for t in d])
                             )


def xml_add_binary_bsecg(
        root: ET.Element,
        data: List[BodySurfaceECG],
        **kwargs
) -> None:
    """Create etree BodySurfaceECG with binary data.

    XML attributes:
        count : number of body surface ECGs

        Example:
            <BSECG
                count=num_bsecgs>
                <BSECG
                    method=generation_method
                    count=num_ECG_traces
                    refAnnotation=ref_time_stamp>
                    <Traces
                        name=generation_method,
                        count=num_ECG_traces>
                        <Trace>
                            <DataArray/>
                            <DataArray/>
                            ...
                        </>
                    </>
            </>

    Data is saved as base64 encoded bytes string.
    Extra attributes can be added by keyword arguments.

    Parameters:
        root : ET.Element
            XML Element the data is added to
        data : BodySurfaceECG
            data to be added

    Returns:
        None
    """

    element = ET.SubElement(root, 'BSECGS',
                            count=str(len(data))
                            )
    # add extra attributes
    for key, value in kwargs:
        element.set(key, value)

    for trace in data:
        item = ET.SubElement(element, 'BSECG',
                             method=trace.method,
                             count=str(len(trace.traces)),
                             refAnnotation=str(trace.refAnnotation),
                             )
        xml_add_binary_trace(item, trace.method, trace.traces)


def xml_load_binary_data(
        element: ET.Element
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """
    Load binary data from etree DataArray.

    Parameters:
        element : ET.Element
            XML Element to read data from

    Raises:
        NotImplementedError : if data format is in XML is not supported

    Returns:
        name : str
        data : np.ndarray
    """

    data_format = element.get('format')
    if not data_format.lower() == 'binary':
        raise NotImplementedError

    name = element.get('name')
    data_type = element.get('type')
    xml_data = element.text
    if not xml_data:
        # no data found, nothing to decode
        return name, xml_data
    num_components = int(element.get('numberOfComponents'))

    data = np.frombuffer(base64.b64decode(xml_data), data_type)
    if num_components > 1:
        data = data.reshape(-1, num_components)

    return name, data


def xml_load_binary_trace(
        element: ET.Element
) -> tuple[Optional[str], Optional[Union[Trace, List[Trace]]]]:
    """
    Load binary Traces from etree element.

    Parameters:
        element : ET.Element
            XML Element to read data from

    Raises:
        ImportError : if number of expected traces does not match data

    Returns:
        name : str
        traces : list of Trace
            dimension is (n_points x num_traces)
    """

    trace_name = element.get('name')
    count = int(element.get('count'))

    traces = []
    for trace in element.findall('Trace'):
        t_data = {}
        for arr in trace.findall('DataArray'):
            name, data = xml_load_binary_data(arr)
            t_data[name] = data

        t = []
        for i in range(len(t_data['name'])):
            t.append(Trace(name=t_data['name'][i],
                           data=t_data['data'][i],
                           fs=t_data['fs'][i]
                           )
                     )
        traces.append(t)

    if not len(traces) == count:
        raise ImportError('expected {} traces, imported {}'
                          .format(count, len(traces))
                          )

    if len(traces) > 1:
        traces = [[row[i] for row in traces]
                  for i in range(max(len(r) for r in traces))]
    else:
        traces = traces[0]

    return trace_name, traces


def xml_load_binary_bsecg(
        element: ET.Element
) -> List[BodySurfaceECG]:
    """
    Load binary BodySurfaceECG object from etree element.

    Parameters:
        element : ET.Element
            XML Element to read data from

    Raises:
        ImportError : if number of expected BSECGs does not match data

    Returns:
        list of BodySurfaceECG
    """

    count = int(element.get('count'))
    bsecg = []

    for item in element.findall('BSECG'):
        _, traces = xml_load_binary_trace(item.find('Traces'))
        bsecg.append(
            BodySurfaceECG(method=item.get('method'),
                           refAnnotation=int(item.get('refAnnotation')),
                           traces=traces
                           )
                     )

    if not len(bsecg) == count:
        raise ImportError('expected {} BSEG(s), imported {}'
                          .format(count, len(bsecg))
                          )

    return bsecg
