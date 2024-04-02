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

import xml.etree.ElementTree as ET
import base64
import numpy as np

from pyceps.datatypes.surface import Surface, SurfaceSignalMap, SurfaceLabel
from pyceps.datatypes.signals import Trace, BodySurfaceECG
from pyceps.datatypes.lesions import Lesion, RFIndex


def xml_add_binary_numpy(root: ET.Element,
                         name: str,
                         data,
                         **kwargs):
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


def xml_add_binary_trace(root: ET.Element,
                         name: str,
                         data,
                         **kwargs):
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
        data : np.ndarray
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


def xml_add_binary_surface(root: ET.Element,
                           name: str,
                           data,
                           **kwargs):
    """
    Create etree Surface with binary data.

    XML attributes:
        numVertices : number of vertices
        numTriangles : number of faces

        Example:
            <Mesh
                numVertices=n_verts,
                numTriangles=n_tris>
                <DataArray name="vertices"/>
                <DataArray name="triangulation"/>
                <SurfaceLabels count=n_labels>
                    <SurfaceLabel location="pointData" or "cellData">
                        <DataArray/>
                    </>
                </>
                <SignalMaps count=n_labels>
                    <SignalMap location="pointData" or "cellData">
                        <DataArray/>
                    </>
                </>
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

    element = ET.SubElement(root, name,
                            numVertices=str(
                             data.X.shape[0]),
                            numTriangles=str(
                             data.tris.shape[0]),
                            )

    # add extra attributes
    for key, value in kwargs:
        element.set(key, value)

    # add triangulation data
    xml_add_binary_numpy(element, 'vertices', data.X)
    xml_add_binary_numpy(element, 'triangulation', data.tris)

    # add surface labels
    surf_labels = ET.SubElement(element, 'SurfaceLabels',
                                count=str(len(data.labels))
                                )
    for label in data.labels:
        s_label = ET.SubElement(surf_labels, 'SurfaceLabel',
                                location=label.location,
                                description=label.description
                                )
        xml_add_binary_numpy(s_label, label.name, label.values)

    # add surface parameter maps
    surf_maps = ET.SubElement(element, 'SignalMaps',
                              count=str(len(data.signalMaps))
                              )
    for signal_map in data.signalMaps:
        s_map = ET.SubElement(surf_maps, 'SignalMap',
                              location=signal_map.location,
                              description=signal_map.description
                              )
        xml_add_binary_numpy(s_map, signal_map.name, signal_map.values)


def xml_add_binary_lesion(root: ET.Element,
                          name: str,
                          data,
                          **kwargs
                          ):
    """Create etree Lesions with binary data."""

    lesions = ET.SubElement(root, 'Lesions',
                            count=str(len(data))
                            )

    xml_add_binary_numpy(lesions, 'points',
                         np.array([site.X for site in data])
                         )
    xml_add_binary_numpy(lesions, 'diameter',
                         np.array([site.diameter for site in data])
                         )
    xml_add_binary_numpy(lesions, 'RFI',
                         np.array([x.value for site in data
                                   for x in site.RFIndex])
                         )
    xml_add_binary_numpy(lesions, 'name',
                         np.array([x.name for site in data
                                   for x in site.RFIndex])
                         )


def xml_add_binary_bsecg(root: ET.Element,
                         name: str,
                         data,
                         **kwargs):
    """Create etree BodySurfaceECG with binary data."""

    bsecgs = ET.SubElement(root, name,
                           count=str(len(data))
                           )
    # add extra attributes
    for key, value in kwargs:
        bsecgs.set(key, value)

    for trace in data:
        item = ET.SubElement(bsecgs, 'BSECG',
                             method=trace.method,
                             count=str(len(trace.traces)),
                             refAnnotation=str(trace.refAnnotation),
                             )
        xml_add_binary_trace(item, trace.method, trace.traces)


def xml_load_binary_data(element: ET.Element):
    """Load binary data from XML element."""

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


def xml_load_binary_surface(element: ET.Element):
    """Load binary Surface object from XML element."""

    numVerts = element.get('numVertices')
    numTris = element.get('numTriangles')

    verts = [x for x in element.findall('DataArray')
             if x.get('name') == 'vertices'][0]
    _, vertices = xml_load_binary_data(verts)

    tris = [x for x in element.findall('DataArray')
            if x.get('name') == 'triangulation'][0]
    _, triangulation = xml_load_binary_data(tris)

    labels = []
    for label in element.find('SurfaceLabels').iter('SurfaceLabel'):
        l_name, data = xml_load_binary_data(label.find('DataArray'))
        labels.append(SurfaceLabel(l_name,
                                   data,
                                   label.get('location'),
                                   label.get('description')
                                   )
                      )
    s_maps = []
    for s_map in element.find('SignalMaps').iter('SignalMap'):
        m_name, data = xml_load_binary_data(s_map.find('DataArray'))
        s_maps.append(SurfaceSignalMap(m_name,
                                       data,
                                       s_map.get('location'),
                                       s_map.get('description'))
                      )

    return Surface(vertices, triangulation,
                   labels=labels,
                   signal_maps=s_maps
                   )


def xml_load_binary_trace(element: ET.Element):
    """Load binary Surface object from XML element."""

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

    if len(traces) > 1:
        traces = [[row[i] for row in traces]
                  for i in range(max(len(r) for r in traces))]
    else:
        traces = traces[0]

    return trace_name, traces


def xml_load_binary_bsecg(element: ET.Element):
    """Load binary Surface object from XML element."""

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

    return bsecg


def xml_load_binary_lesion(element: ET.Element):
    """Load binary Surface object from XML element."""

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
            Lesion(X=l_data['points'][i],
                   diameter=l_data['diameter'][i],
                   RFIndex=rfi
                   )
        )

    return lesions
