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

import os
import logging
from typing import Optional, Union, List, TypeVar
import zipfile
import py7zr
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import scipy.spatial.distance as sp_distance
from itertools import compress

from pyceps.fileio.pathtools import Repository
from pyceps.study import EPStudy, EPMap, EPPoint
from pyceps.datatypes import Surface, Mesh

from pyceps.fileio import FileWriter
from pyceps.fileio.xmlio import (
    xml_add_binary_numpy,
    xml_load_binary_data,
    xml_load_binary_trace,
    xml_load_binary_bsecg,
)
from pyceps.fileio.cartoio import (
    read_mesh_file,
    read_ecg_file_header, read_ecg_file,
    read_force_file,
    read_visitag_file,
    read_paso_config, read_paso_correlations,
    read_electrode_pos_file
)
from pyceps.datatypes.carto.cartotypes import (
    CartoUnits, Coloring, ColoringRange,
    SurfaceErrorTable,
    CFAEColoringTable, Tag,
    CartoMappingParameters,
    RefAnnotationConfig, PointImpedance,
    RFAblationParameters, RFForce, MapRF,
)
from pyceps.datatypes.carto.visitag import (
    Visitag, VisitagAblationSite, VisitagGridPoint, VisitagAblationGrid
)
from pyceps.datatypes.carto.paso import PaSo, PasoTable, PaSoTemplate
from pyceps.datatypes.lesions import Lesions, RFIndex
from pyceps.datatypes.signals import Trace, BodySurfaceECG
from pyceps.datatypes.exceptions import (MapAttributeError,
                                         MeshFileNotFoundError
                                         )
from pyceps.utils import console_progressbar, get_col_idx_from_header


TCartoStudy = TypeVar('TCartoStudy', bound='CartoStudy')  # workaround to type hint self


log = logging.getLogger(__name__)


class CartoPoint(EPPoint):
    """
    Class representing Carto3 point.

    Attributes:
        name : str
            identifier for this recording point
        pointFile : str
            name of the points XML file <map_name>_<point_ID>_Point_Export.xml
        parent : CartoMap
            parent mapping procedure this point belongs to
        recX : ndarray (3, )
            coordinates at which this point was recorded
        prjX : ndarray (3, )
            coordinates of the closest anatomical shell vertex
        prjDistance : float
            distance between recording location and closest shell vertex
        refAnnotation : int
            annotation for reference detection in samples
        latAnnotation : int
            annotation for local activation time in samples
        woi : ndarray (2, 1)
            start and end timestamps of the WOI in samples
        uniVoltage : float
            peak-to-peak voltage in unipolar EGM
        bipVoltage : float
            peak-to-peak voltage in bipolar EGM
        egmBip : Trace
            bipolar EGM trace
        egmUni : Trace
            unipolar EGm trace(s). If supported by the mapping system,
            two unipolar traces are stored
        egmRef : Trace
            reference trace
        impedance : float
        force : float
        barDirection : ndarray (3, 1)
            surface normal of the closest surface point
        tags : list of str
            tags assigned to this point, i.e. 'Full_name' in study's TagsTable
        ecgFile : str
            name of the points ECG file <map_name>_<point_name>_ECG_Export.txt
        uniX : ndarray (3, )
            cartesian coordinates of the second unipolar recording electrode
            NOTE: coordinates of second unipolar electrode are same as recX if
            position cannot be determined
        forceFile : str
            name of the points contact force file
            <map_name>_<point_name>_Contact_Force.txt
        forceData : PointForce
            full contact force data for this point
        impedanceData : PointImpedance
            full impedance data for this point

    Methods:
        is_valid()
            check if point has LAT annotation within WOI
        load(egm_names_from_pos=False)
            load all data associated with this point
        import_ecg(channel_names)
            import ECG data for this point
    """

    def __init__(
            self,
            name: str,
            coordinates: np.ndarray = np.full(3, np.nan, dtype=np.float32),
            tags: Optional[List[str]] = None,
            parent: Optional['CartoMap'] = None
    ) -> None:
        """
        Constructor.

        Parameters:
             name : str
                name / identifier for this point
            coordinates : ndarray(3, )
                cartesian coordinates of recording position
            tags: list of str (optional)
                tags assigned to this point, i.e. 'Full_name' in study's
                TagsTable
            parent : CartoMap (optional)
                the map this point belongs to

        Returns:
            None

        """

        super().__init__(name, coordinates=coordinates, parent=parent)

        # add Carto3 specific attributes
        self.pointFile = ''
        self.barDirection = None
        self.tags = tags
        self.ecgFile = ''
        self.forceFile = ''
        self.forceData = None
        self.impedanceData = None

    def import_point(
            self,
            point_file: str,
            egm_names_from_pos: bool = False
    ) -> None:
        """
        Load data associated with this point.

        Parameters:
            point_file : str
                name of this points XML file
                <map_name>_<point_name>_Point_Export.xml
            egm_names_from_pos : boolean
                If True, EGM electrode names are extracted from positions file.
                This also returns name and coordinates of the second unipolar
                channel.

        Raises:
            FileNotFoundError : if point's XML is not found

        Returns:
            None
        """

        log.debug('Loading point data for point {}'.format(self.name))

        file_loc = self.parent.parent.repository.join(point_file)
        if not self.parent.parent.repository.is_file(file_loc):
            log.info('Points export file {} does not exist'
                     .format(point_file))
            raise FileNotFoundError
        self.pointFile = point_file

        # read annotation data
        point_file = self.parent.parent.repository.join(self.pointFile)
        with self.parent.parent.repository.open(point_file) as fid:
            root = ET.parse(fid).getroot()

        annotation_item = root.find('Annotations')
        self.refAnnotation = int(annotation_item.get('Reference_Annotation'))
        self.latAnnotation = int(annotation_item.get('Map_Annotation'))
        woi_item = root.find('WOI')
        self.woi = np.asarray([woi_item.get('From'),
                               woi_item.get('To')]
                              ).astype(int)
        voltages_item = root.find('Voltages')
        self.uniVoltage = float(voltages_item.get('Unipolar'))
        self.bipVoltage = float(voltages_item.get('Bipolar'))

        # read impedance data
        impedance_item = root.find('Impedances')
        n_impedance_values = int(impedance_item.get('Number'))
        if n_impedance_values > 0:
            impedance_value = np.empty(n_impedance_values, dtype=np.float32)
            impedance_time = np.empty(n_impedance_values, dtype=np.int32)
            for i, x in enumerate(impedance_item.findall('Impedance')):
                impedance_time[i] = x.get('Time')
                impedance_value[i] = x.get('Value')

            self.impedanceData = PointImpedance(time=impedance_time,
                                                value=impedance_value)
            # update base class impedance value with the one closest to LAT
            self.impedance = impedance_value[
                np.nanargmin(np.abs(impedance_time - self.latAnnotation))
            ]

        self.ecgFile = root.find('ECG').get('FileName')

        # get egm names from ECG file
        ecg_file = self.parent.parent.repository.join(self.ecgFile)
        with self.parent.parent.repository.open(ecg_file) as fid:
            ecg_file_header = read_ecg_file_header(
                fid,
                encoding=self.parent.parent.encoding
            )
        if ecg_file_header['version'] == '4.1':
            # channel names are given in pointFile for version 4.1+
            ecg_file_header['name_bip'] = root.find('ECG').get(
                'BipolarMappingChannel')
            ecg_file_header['name_uni'] = root.find('ECG').get(
                'UnipolarMappingChannel')
            ecg_file_header['name_ref'] = root.find('ECG').get(
                'ReferenceChannel')
        egm_names = self._channel_names_from_ecg_header(ecg_file_header)

        # get coordinates of second unipolar channel
        self.uniX = self._get_2nd_uni_x(encoding=self.parent.parent.encoding)

        if egm_names_from_pos:
            egm_names, uniCoordinates = self._channel_names_from_pos_file(
                egm_names,
                encoding=self.parent.parent.encoding
            )
            self.uniX = uniCoordinates

        # now we can import the electrograms for this point
        egm_data = self.load_ecg([egm_names['bip'],
                                  egm_names['uni1'],
                                  egm_names['uni2'],
                                  egm_names['ref']])
        # build egm traces
        self.egmBip = [t for t in egm_data if t.name == egm_names['bip']][0]
        egmUni = [
            [t for t in egm_data if t.name == egm_names['uni1']][0],
            [t for t in egm_data if t.name == egm_names['uni2']][0]
        ]
        self.egmUni = egmUni
        self.egmRef = [t for t in egm_data if t.name == egm_names['ref']][0]

        # get the closest surface vertex for this point
        if self.parent.surface.has_points():
            closest, distance, direct = self.parent.surface.get_closest_vertex(
                [self.recX]
            )
            if closest.shape[0] != 1:
                log.warning('found no or multiple surface vertices closest to '
                            'to point {}: {}'
                            .format(self.name, closest))
            self.prjX = np.array(closest[0], dtype=np.float32)
            self.prjDistance = distance[0]
            self.barDirection = direct[0]

        # now get the force data for this point
        log.debug('reading force file for point {}'.format(self.name))
        try:
            self.forceFile = root.find('ContactForce').get('FileName')
            force_file = self.parent.parent.repository.join(self.forceFile)
            if self.parent.parent.repository.is_file(force_file):
                with self.parent.parent.repository.open(force_file) as fid:
                    self.forceData = read_force_file(
                        fid, encoding=self.parent.parent.encoding
                    )
                if np.isnan(self.forceData.force):
                    # update base class force value with the one closest to LAT
                    self.force = self.forceData.timeForce[
                        np.nanargmin(np.abs(self.forceData.timeForce
                                            - self.latAnnotation)
                                     )
                    ]
                else:
                    self.force = self.forceData.force

            else:
                log.debug('No force file found for point {}'.format(self.name))
        except AttributeError:
            log.debug('No force data saved for point {}'.format(self.name))

    def is_valid(
            self
    ) -> bool:
        """
        Check if LAT annotation is within the WOI.

        Raises:
            ValueError : If no WOI or reference annotation is found or there
                no LAT annotation for this point

        Returns:
            True if the points map annotation is within the WOI, else False

        """

        if not self.latAnnotation:
            log.warning('no activation annotation found for {}!'
                        .format(self.name)
                        )
            raise ValueError('Parameter mapAnnotation missing!')
        if self.woi.size == 0 or not self.refAnnotation:
            log.warning('no woi and/or reference annotation found for {}!'
                        .format(self.name)
                        )
            raise ValueError('Parameters WOI and/or refAnnotation missing!')

        woi = self.woi + self.refAnnotation

        return woi[0] < self.latAnnotation < woi[1]

    def load_ecg(
            self,
            channel_names: Optional[Union[str, List[str]]] = None,
            reload: bool = False,
            *args, **kwargs
    ) -> Optional[List[Trace]]:
        """
        Load ECG data for this point.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Parameters:
            channel_names : string or list of string
                channel names to read
            reload : bool
                reload data if already present

        Raises:
            KeyError : If a channel name is not found in ECG file

        Returns:
             list of Trace
        """

        if not self.ecgFile:
            log.warning('No ECG file found for point {}'.format(self.name))
            return None

        ecg_file = self.parent.parent.repository.join(self.ecgFile)

        if isinstance(channel_names, str):
            channel_names = [channel_names]

        with self.parent.parent.repository.open(ecg_file) as fid:
            ecg_header = read_ecg_file_header(fid)
        ecg_channels = ecg_header['ecg_names']

        not_found = []
        if not channel_names:
            # read all channels
            channel_names = ecg_channels
        else:
            # check if all names are valid
            not_found = [item for item in channel_names
                         if not any([channel.startswith(item+'(')
                                     for channel in ecg_channels])]
        if not_found:
            raise KeyError('channel(s) {} not found for point {}'
                           .format(not_found, self.name))

        if not reload:
            # check which data is already loaded
            channel_names = [n for n in channel_names
                             if n not in self.get_ecg_names()
                             ]
            if not channel_names:
                # all data already loaded, skip rest
                return []

        # get index of required channels in file
        cols = [ecg_channels.index(x) for channel in channel_names
                for x in ecg_channels if x.startswith(channel+'(')]

        with self.parent.parent.repository.open(ecg_file) as fid:
            ecg_data = read_ecg_file(fid,
                                     column_indices=cols,
                                     skip_rows=ecg_header['header_lines']
                                     )
        ecg_data *= ecg_header['gain']

        try:
            ecg_data.shape[1]
        except IndexError:
            # array has shape (2500, ) but (2500, 1) is needed
            ecg_data = np.expand_dims(ecg_data, axis=1)

        # build Traces
        traces = []
        for i, name in enumerate(channel_names):
            traces.append(
                Trace(name=name,
                      data=ecg_data[:, i].astype(np.float32),
                      fs=1000.0)
            )

        return traces

    def _channel_names_from_ecg_header(
            self,
            ecg_header: dict
    ) -> dict:
        """
        Get channel names for BIP, UNI and REF traces from file header.

        This function also tries to extract the name of the second unipolar
        channel from the bipolar channel name.

        Parameters:
            ecg_header : dict
                header info returned from read_ecg_file_header()

        Returns:
            dict : channel names
                keys: 'bip', 'uni1', 'uni2', 'ref'
        """

        log.debug('extracting channel names from ECG header for point {}'
                  .format(self.name))

        # MEC connector names have different naming convention
        if ecg_header['name_bip'].startswith('MCC'):
            log.warning('point {} was recorded with MEC connector, unipolar '
                        'channel names might be wrong!'
                        .format(self.name))
            uni_name1 = ecg_header['name_uni']
            # TODO: fix second unipolar channel name for MCC Ablation
            uni_name2 = uni_name1

        else:
            # get unipolar names from bipolar electrode names
            try:
                connector, channels = ecg_header['name_bip'].split('_')
                channel_num = channels.split('-')
                uni_name1 = connector + '_' + channel_num[0]
                uni_name2 = connector + '_' + channel_num[1]
            except ValueError:
                # some connectors don't add the connector name at beginning
                channel_names = ecg_header['name_bip'].split('-')
                uni_name1 = channel_names[0]
                uni_name2 = channel_names[1]

        # compare extracted names with header info
        if not uni_name1 == ecg_header['name_uni']:
            log.warning('extracted unipolar EGM channel name does not match '
                        'ECG header info! Using header info!')
            uni_name1 = ecg_header['name_uni']
            uni_name2 = uni_name1

        return {'bip': ecg_header['name_bip'],
                'uni1': uni_name1,
                'uni2': uni_name2,
                'ref': ecg_header['name_ref']
                }

    def _get_2nd_uni_x(
            self,
            encoding: str = 'cp1252'
    ) -> np.ndarray:
        """
        Get coordinates for 2nd unipolar channel from position file(s).

        Searches for recording coordinates in position file(s) and extracts
        coordinates of the subsequent channel. This should be the correct
        second unipolar channel for bipolar recordings.

        NOTE: If coordinates cannot be determined, the recording position of
        the first unipolar channel is used!

        NOTE: Method _channel_names_from_pos_file is more elaborate but
        fails often for missing channel positions in position files!

        Parameters:
            encoding : str (optional)
                file encoding

        Returns:
            ndarray(3, )
                coordinates of the second unipolar channel
        """

        xyz_2 = np.full(3, np.nan, dtype=np.float32)

        log.debug('get position of 2nd unipolar channel')

        # read points XML file
        point_file = self.parent.parent.repository.join(self.pointFile)
        with self.parent.parent.repository.open(point_file) as fid:
            root = ET.parse(fid).getroot()

        # get position files
        position_files = []
        for connector in root.find('Positions').findall('Connector'):
            connector_file = list(connector.attrib.values())[0]
            if 'ectrode_positions_onannotation' in connector_file.lower():
                position_files.append(connector_file)

        for filename in position_files:
            pos_file = self.parent.parent.repository.join(filename)
            if not self.parent.parent.repository.is_file(pos_file):
                log.warning('position file {} for point {} not found'
                            .format(filename, self.name))
                continue

            with self.parent.parent.repository.open(pos_file) as fid:
                idx, time, xyz = read_electrode_pos_file(fid, encoding=encoding)

            # find electrode with the closest distance
            dist = sp_distance.cdist(xyz, np.expand_dims(self.recX, axis=1).T).flatten()
            idx_closest = np.argwhere(dist == np.amin(dist)).flatten()
            if idx_closest.size != 1:
                log.debug(
                    'found no or multiple electrodes with same minimum '
                    'distance in file {}. Trying next file...'
                    .format(filename))
                continue
            idx_closest = idx_closest[0]
            if dist[idx_closest] > 0:
                # position must be exact match
                continue

            try:
                xyz_2 = xyz[idx_closest + 1, :]
            except IndexError:
                log.debug('unable to get position of 2nd uni channel for '
                          'point {}'.format(self.name))

        if np.isnan(xyz_2).all():
            log.warning('coordinates for 2nd unipolar channel not found for '
                        'point {}, using recording position'
                        .format(self.name))
            xyz_2 = self.recX

        return xyz_2

    def _channel_names_from_pos_file(
            self,
            egm_names: dict,
            encoding: str = 'cp1252'
    ) -> tuple[dict, np.ndarray]:
        """
        Get channel names for BIP, UNI and REF traces from electrode positions.

        Coordinates of the 2nd unipolar channel are determined and returned.

        Extracted names are compared to EGM names in CARTO point ECG file. If
        discrepancies are found, the names from the ECG files are used.

        Parameters:
            egm_names : dict
                names extracted from ECG file for comparison
            encoding : str (optional)
                file encoding

        Returns:
            dict : channel names
                keys: 'bip', 'uni1', 'uni2', 'ref'
            uniCoordinates : ndarray
                coordinates of the 2nd unipolar channel
        """

        log.debug('extracting channel names from position files for point {}'
                  .format(self.name))

        # read points XML file
        point_file = self.parent.parent.repository.join(self.pointFile)
        with self.parent.parent.repository.open(point_file) as fid:
            root = ET.parse(fid).getroot()

        # get position files
        position_files = []
        for connector in root.find('Positions').findall('Connector'):
            connector_file = list(connector.attrib.values())[0]
            if 'ectrode_positions_onannotation' in connector_file.lower():
                position_files.append(connector_file)

        bipName, uniName, xyz_2 = self._find_electrode_at_pos(
            self.recX,
            position_files,
            encoding=encoding
        )

        # now check the name of the electrode identified above by
        # comparing with the ECG_Export file
        if not egm_names['bip'] == bipName:
            log.warning('Conflict: bipolar electrode name {} from position '
                        'file does not match electrode name {} in ECG file!\n'
                        'Using name from ECG file for point {}.'
                        .format(bipName, egm_names['bip'], self.name)
                        )
            bipName = egm_names['bip']

        if not egm_names['uni1'] == uniName[0]:
            log.warning('Conflict: unipolar electrode name {} from position '
                        'file does not match electrode name {} in ECG file!\n'
                        'Using name from ECG file for point {}.'
                        .format(uniName[0], egm_names['uni1'], self.name)
                        )
            uniName[0] = egm_names['uni1']
            uniName[1] = egm_names['uni2']

        names = {'bip': bipName,
                 'uni1': uniName[0],
                 'uni2': uniName[1],
                 'ref': egm_names['uni1']
                 }

        return names, xyz_2

    def _find_electrode_at_pos(
            self,
            point_xyz: np.ndarray,
            position_files: List[str],
            encoding: str = 'cp1252'
    ) -> tuple[str, List[str], np.ndarray]:
        """
        Find electrode that recorded Point at xyz.

        This function also tries to identify the name and coordinates of the
        second unipolar channel which made the bipolar recording.

        Parameters:
            point_xyz : ndarray (3, 1)
                coordinates of the point
            position_files : list of string
                path to the position files
            encoding : str (optional)
                file encoding

        Returns:
             egm_name_bip : string
                name of the bipolar channel
            egm_name_uni : list of string
                names of the unipolar channels
            xyz_2 : ndarray (3, )
                coordinates of the second unipolar channel
        """

        log.debug('find electrode that recorded point at {}'
                  .format(point_xyz))

        egm_name_bip = ''
        egm_name_uni = ['', '']
        xyz_2 = np.full(3, np.nan, dtype=np.float32)

        channel_number = np.nan

        for filename in position_files:
            pos_file = self.parent.parent.repository.join(filename)
            if not self.parent.parent.repository.is_file(pos_file):
                log.warning('position file {} for point {} not found'
                            .format(filename, self.name))
                continue

            with self.parent.parent.repository.open(pos_file) as fid:
                idx, time, xyz = read_electrode_pos_file(fid)

            # calculate range of electrode positions and add last index
            lim = np.append(np.where(idx[:-1] != idx[1:])[0], len(idx) - 1)

            # find electrode with the closest distance
            dist = sp_distance.cdist(xyz, np.array([point_xyz])).flatten()
            idx_closest = np.argwhere(dist == np.amin(dist)).flatten()
            if idx_closest.size != 1:
                log.debug('found no or multiple electrodes with same minimum '
                          'distance in file {}. Trying next file...'
                          .format(filename))
                continue
            idx_closest = idx_closest[0]
            if dist[idx_closest] > 0:
                # position must be exact match
                continue

            # get channel list and index for reduced positions
            electrode_idx = lim.searchsorted(idx_closest, 'left')
            channel_list = idx[lim]

            # find second unipolar channel recorded at same time in next
            # position block
            time_closest = time[idx_closest]
            # get block limits
            try:
                block_start = lim[electrode_idx] + 1
                idx_end = lim[electrode_idx+1] + 1
            except IndexError:
                log.warning('point was recorded with last electrode, unable '
                            'to get second channel!')
                continue

            block_end = idx_end if (idx_end < time.shape[0]) else None
            idx_time = np.argwhere(time[block_start:block_end] == time_closest).flatten()
            if idx_time.size != 1:
                log.debug('found no matching time stamp for second unipolar '
                          'channel in file {}'
                          .format(filename))
            else:
                xyz_2 = xyz[idx_time[0] + lim[electrode_idx] + 1, :]

            # translate connector index to channel number
            try:
                egm_name_bip, egm_name_uni = self._translate_connector_index(
                    channel_list,
                    int(electrode_idx),
                    filename
                )
                # if no error, update minimum distance and file
                min_dist = dist[idx_closest]
            except IndexError:
                # channel indexing does not match used connector, sometimes
                # channels are missing in position on annotation file

                if '_OnAnnotation' not in filename:
                    # already working on extended files
                    log.debug('unable to get channel name from extended '
                              'position file {}'.format(filename))
                    break

                log.warning('channel indexing does not match connector! '
                            'Trying to find in extensive position file(s)')

                # get filename of extended position file
                ext_filename = filename.replace('_OnAnnotation', '')
                egm_name_bip, egm_name_uni, xyz_2 = (
                    self._find_electrode_at_pos(self.recX,
                                                [ext_filename],
                                                encoding=encoding)
                )

        if not egm_name_bip or not any(egm_name_uni):
            log.warning('unable to find which electrode recorded '
                        'point {} at {}!'
                        .format(self.name, point_xyz))
            return ('',
                    ['', ''],
                    np.full(3, np.nan, dtype=np.float32)
                    )

        return egm_name_bip, egm_name_uni, xyz_2

    @staticmethod
    def _translate_connector_index(
            channel_list: np.ndarray,
            electrode_index: int,
            filename: str
    ) -> tuple[str, List[str]]:
        """
        Translate connector index in electrode position file to channel name
        in ecg file.

        Parameters:
            channel_list : ndarray of type int
                channel numbers from position file
            electrode_index : int
                index of the recording electrode
            filename : str
                position file to evaluate channel naming convention

        Raises:
            IndexError : If channel indexing does not match a known connector.

        Returns:
             egm_name_bip : string
                name of the bipolar channel
            egm_name_uni : list of string
                names of the unipolar channels

        """

        egm_name_bip = ''
        egm_name_uni = ['', '']

        LASSO_INDEXING = [1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                          1, 2, 3, 4]
        PENTA_INDEXING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22]
        # TODO: implement correct indexing for CS catheter
        CS_INDEXING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # now we have to translate the filename into the egm name that gives us
        # the correct egm in the ECG_Export file.
        identifier = ['CS_CONNECTOR',
                      'MAGNETIC_20_POLE_A_CONNECTOR',
                      'MAGNETIC_20_POLE_B_CONNECTOR',
                      'NAVISTAR_CONNECTOR',
                      'MEC']
        translation = ['CS',
                       '20A_',
                       '20B_',
                       'M',
                       'MCC Abl BiPolar']

        idx_identifier = [identifier.index(x) for x in identifier
                          if x in filename][0]
        egm_name = translation[idx_identifier]

        if egm_name == 'MCC Abl BiPolar':
            electrode_number = channel_list[electrode_index]
            egm_name_bip = egm_name + ' {}'.format(electrode_number)
            egm_name_uni[0] = 'M{}'.format(electrode_number)
            # TODO: what is the second unipolar channel for this?
            egm_name_uni[1] = egm_name_uni[0]

        elif egm_name == '20A_' or egm_name == '20B_':
            # check which catheter was used
            if np.array_equal(channel_list, LASSO_INDEXING):
                # two electrodes offset and 1-based numbering
                electrode_number = electrode_index - 2 + 1
            elif np.array_equal(channel_list, PENTA_INDEXING):
                electrode_number = channel_list[electrode_index]
            else:
                raise IndexError('channel indexing does not match specified '
                                 'connector!')

            # build channel name
            egm_name_bip = '{}{}-{}'.format(egm_name,
                                            electrode_number,
                                            electrode_number + 1)
            egm_name_uni[0] = '{}{}'.format(egm_name, electrode_number)
            egm_name_uni[1] = '{}{}'.format(egm_name, electrode_number + 1)
            # TODO: why is this done??
            if egm_name_bip == '20B_7-8':
                egm_name_bip = '20B_9-8'

        else:
            log.warning('unknown connector! Trying best guess for channel '
                        'name!')
            electrode_number = channel_list[electrode_index]
            egm_name_bip = '{}{}-{}{}'.format(egm_name,
                                              electrode_number,
                                              egm_name,
                                              electrode_number + 1)
            egm_name_uni[0] = '{}{}'.format(egm_name, electrode_number)
            egm_name_uni[1] = '{}{}'.format(egm_name, electrode_number + 1)

        return egm_name_bip, egm_name_uni


class CartoMap(EPMap):
    """
    Class representing Carto3 map.

    Attributes:
        name : str
            name of the mapping procedure
        studyXML : str
            filename of the study's XML file (same as in parent)
        parent : subclass of EPStudy
            the parent study for this map
        index : int
            index of the map assigned by the Carto system
        visible : str
            boolean string ('true' or 'false') if map was visible in Carto
        type : str
            type of the map
        volume : float
            volume of the mesh, calculated by Carto
        RefAnnotationConfig : RefAnnotationConfig object
            algorithm and connector used as reference
        coloringRangeTable : list of ColoringRange
            color ranges used by Carto
        surfaceFile : str
            filename of file containing the anatomical shell data
        surface : Surface
            triangulated anatomical shell
        points : list of subclass EPPoints
            the mapping points recorded during mapping procedure
        bsecg : list of BodySurfaceECG
            body surface ECG data for the mapping procedure
        lesions : list of Lesion
            ablation data for this mapping procedure
        rf : MapRF object
            force and ablation data of the mapping procedure

    Methods:
        load_mesh()
            load triangulated anatomical shell
        load_points(study_tags=None, egm_names_from_pos=False)
            load EGM points
        import_lesions(directory=None)
            import lesion data for this mapping procedure (for consistency
            only)
        get_map_ecg(ecg_names=None, method=None)
            build representative body surface ECGs
        export_point_ecg(basename='', which=None, points=None)
            export ECG data for points in IGB format
        import_rf_data()
            import RF and force data
        visitag_to_lesion(visitag_sites)
            convert VisiTag ablation sites to BaseClass Lesion
    """

    def __init__(
            self,
            name: str,
            study_xml: str,
            parent: Optional['CartoStudy'] = None
    ) -> None:
        """
        Constructor.

        Parameters:
            name : str
                name of the mapping procedure
            study_xml : str
                name of the study's XML file (same as in parent)
            parent : CartoStudy (optional)
                study this map belongs to

        Returns:
            None
        """

        super().__init__(name, parent=parent)

        # add Carto3 specific attributes
        self.studyXML = study_xml
        self.index = np.nan
        self.visible = None
        self.type = None
        self.volume = np.nan
        self.RefAnnotationConfig = None
        self.coloringRangeTable = []
        self.rf = None

    def import_map(
            self,
            egm_names_from_pos: bool
    ) -> None:
        """
        Load all relevant information for this mapping procedure, import EGM
        recording points, interpolate standard surface parameter maps from
        point data (bip voltage, uni voltage, LAT), and build representative
        body surface ECGs.

        Parameters:
            egm_names_from_pos : bool
                If True, EGM electrode names are extracted from positions file.
                This also returns name and coordinates of the second unipolar
                channel.

        Raises:
            MapAttributeError : If unable to retrieve map attributes from XML
            MeshFileNotFoundError: If mesh file is not found in repository

        Returns:
            None
        """

        self._import_attributes()
        self.surface = self.load_mesh()

        # check if parent study was imported or loaded
        # if it was loaded, some attributes are missing
        if not self.parent.mappingParams:
            log.info('study was probably loaded from file, need to re-import '
                     'basic study information')
            self.parent.import_study()

        # load points
        self.points = self.load_points(
            study_tags=self.parent.mappingParams.TagsTable,
            egm_names_from_pos=egm_names_from_pos)

        # build surface maps
        self.interpolate_data('lat')
        self.interpolate_data('bip')
        self.interpolate_data('uni')
        self.interpolate_data('imp')
        self.interpolate_data('frc')

        # build map BSECGs
        self.bsecg = self.build_map_ecg(method=['median', 'mse', 'ccf'])

    def load_mesh(
            self
    ) -> Surface:
        """
        Load a Carto3 triangulated anatomical shell from file.

        Raises:
            MeshFileNotFoundError : if mesh file not found

        Returns:
            Surface
        """

        log.info('reading Carto3 mesh {}'.format(self.surfaceFile))

        mesh_file = self.parent.repository.join(self.surfaceFile)
        if not self.parent.repository.is_file(mesh_file):
            raise MeshFileNotFoundError(filename=self.surfaceFile)

        with self.parent.repository.open(mesh_file, mode='rb') as fid:
            return read_mesh_file(fid, encoding=self.parent.encoding)

    def load_points(
            self,
            study_tags: Optional[List[Tag]] = None,
            egm_names_from_pos: bool = False
    ) -> List[CartoPoint]:
        """
        Load points for Carto3 map.

        EGM names for recording points can be identified by evaluating the
        recording position to get the name of the electrode and comparing it
        to the name found in the points ECG file. Otherwise, the EGM name
        stored in a points ECG file is used.

        Parameters:
            study_tags : list of Tag objects (optional)
                to transform points tag ID to label name
            egm_names_from_pos : boolean (optional)
                Get EGM names from recording positions. (default is False)

        Returns:
            list of CartoPoints objects
        """

        log.info('import EGM points')

        if not study_tags:
            log.warning('no tag names provided for study {}: cannot '
                        'convert tag ID to tag name'.format(self.name))

        points = []

        xml_file = self.parent.repository.join(self.studyXML)
        with self.parent.repository.open(xml_file, mode='rb') as fid:
            root = ET.parse(fid).getroot()

        map_item = [x for x in root.find('Maps').findall('Map')
                    if x.get('Name') == self.name]
        if not map_item:
            log.warning('no map with name {} found in study XML'
                        .format(self.name))
            return []

        if len(map_item) > 1:
            log.warning('multiple maps with name {} found in study XML'
                        .format(self.name))
            return []

        map_item = map_item[0]

        all_points_file = self.parent.repository.join(
            self.name + '_Points_Export.xml'
        )
        if not self.parent.repository.is_file(all_points_file):
            log.warning('unable to find export overview of all points {}'
                        .format(all_points_file))
            return []

        with self.parent.repository.open(all_points_file, mode='rb') as fid:
            root = ET.parse(fid).getroot()

        if not root.get('Map_Name') == self.name:
            log.warning('map name {} in export file {} does not match map '
                        'name {} for import'
                        .format(root.get('Map_Name'),
                                self.name,
                                all_points_file)
                        )
            return []

        point_files = {}
        for i, point in enumerate(root.findall('Point')):
            point_files[point.get('ID')] = point.get('File_Name')

        # TODO: read field "Anatomical_Tags"

        # get points in this map from study XML
        n_points = int(map_item.find('CartoPoints').get('Count'))
        if not len(point_files) == n_points:
            log.warning('number of points is not equal number of points files')
            return []

        log.info('loading {} points'.format(n_points))
        for i, point in enumerate(
                map_item.find('CartoPoints').findall('Point')):

            point_name = 'P' + point.get('Id')
            # update progress bar
            console_progressbar(
                i+1, n_points,
                suffix='Loading point {}'.format(point_name)
            )

            xyz = np.array(point.get('Position3D').split()).astype(np.float32)

            # get tags for this point
            tag_names = []
            tags = point.find('Tags')
            if tags is not None and study_tags:
                n_tags = int(point.find('Tags').get('Count'))
                tag_ids = [int(x) for x in point.find('Tags').text.split()]
                if len(tag_ids) != n_tags:
                    log.warning('number of tags does not match number of '
                                'tag IDs for point {}'
                                .format(point_name))
                else:
                    tag_names = [x.FullName for x in study_tags
                                 for tid in tag_ids
                                 if int(x.ID) == tid]

            # get files associated with this point
            try:
                point_file = point_files[point.get('Id')]
            except KeyError:
                log.info('No Point Export file found for point {}'
                         .format(point_name))
                point_file = None

            log.debug('adding point {} to map {}'.format(point_name,
                                                         self.name))
            new_point = CartoPoint(point_name,
                                   coordinates=xyz,
                                   tags=tag_names,
                                   parent=self)
            new_point.import_point(point_file,
                                   egm_names_from_pos=egm_names_from_pos
                                   )
            points.append(new_point)

        return points

    def import_lesions(
            self,
            directory: str = ''
    ) -> None:
        """
        Import VisiTag lesion data.

        Note: More than one RF index can be stored per ablation site.

        Parameters:
            directory : str (optional)
                path to VisiTag data. If None, standard location
                ../<studyRepository>/VisiTagExport is used

        Returns:
            None
        """

        # VisiTag data is stored study-wise, so check parent for data.
        if not self.parent.visitag.sites:
            self.parent.import_visitag_sites(directory=directory)

        # check if lesion data was loaded
        if not self.parent.visitag.sites:
            log.info('no VisiTag data found in study')
            return

        self.lesions = self.parent.visitag.to_lesions()

    def build_map_ecg(
            self,
            ecg_names: Optional[Union[str, List[str]]] = None,
            method: Optional[Union[str, List[str]]] = None,
            reload_data: bool = False,
            *args, **kwargs
    ) -> List[BodySurfaceECG]:
        """Get a mean surface ECG trace.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        CARTO points are recorded sequentially. Therefore, ECG traces
        recorded at each point (i.e. at a time during procedure) vary. This
        function calculates a representative ECG.

        Building ECG traces with multiple method is most efficient when
        specifying the methods in a single call, since data has to be read
        only once.

        Parameters:
            ecg_names : str, list of str
                ECG names to build. If not specified, 12-lead ECG is used
            method : str, list of str (optional)
                Method to use. Options are ['median', 'ccf', 'mse']
                'median': time-wise median value of all ECGs
                'ccf': recorded ECG with highest cross-correlation to mean ecg
                'mse': recorded ECG with lowest MSE to mean ecg
                If not specified, all methods are used
            reload_data : bool
                reload ECG data or use if already loaded before

        Returns
            list of BodySurfaceECG
        """

        if not ecg_names:
            ecg_names = ['I', 'II', 'III',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                         'aVL', 'aVR', 'aVF'
                         ]
        elif isinstance(ecg_names, str):
            ecg_names = [ecg_names]

        if not method:
            method = ['median', 'mse', 'ccf']

        if isinstance(method, str):
            method = [method]

        log.info('map {}: building representative ECGs: {}'
                 .format(self.name, ecg_names))

        points = self.get_valid_points()
        if not points:
            log.info('no points found in WOI or no points in map, aborting...')
            return []

        log.debug('found {} points in WOI'.format(len(points)))

        # check if data is required
        missing_data = [p.is_ecg_data_required(ecg_names) for p in points]

        if any(missing_data) and not self.parent.is_root_valid():
            log.warning('valid study root is required to load ECG data!')
            return []

        if any(missing_data):
            log.info('missing ECG data, loading...')
            missing_points = list(compress(points, missing_data))
            for i, point in enumerate(missing_points):
                # update progress bar
                console_progressbar(
                    i + 1, len(missing_points),
                    suffix='Loading ECG(s) for point {}'.format(point.name)
                )

                point.ecg.extend(point.load_ecg(ecg_names, reload=reload_data))

        # data is available now, begin build
        data = np.full((len(points), 2500, len(ecg_names)),
                       np.nan,
                       dtype=np.float32)
        ref = points[0].refAnnotation
        woi = points[0].woi

        # get ECG traces for each mapping point
        for i, point in enumerate(points):
            # update progress bar
            console_progressbar(
                i+1, len(points),
                suffix='Processing ECG(s) for point {}'.format(point.name)
            )

            # append point ECG data
            point_data = np.array(
                [t.data for t in point.ecg
                 for chn in ecg_names if t.name == chn]
            )
            data[i, :, :] = point_data.T

            # check WOI and RefAnnotation
            if not point.woi[0] == woi[0] or not point.woi[1] == woi[1]:
                log.warning('WOI changed in point {}'.format(point.name))
                # make this WOI the new one
                woi = point.woi
            if not point.refAnnotation == ref:
                log.warning('REF annotation changed in point {}'
                            .format(point.name))
                # make this the new ref
                ref = point.refAnnotation

        # build representative bsecg trace
        repr_ecg = []

        for meth in method:
            if meth.lower() == 'median':
                ecg = np.median(data, axis=0)
            elif meth.lower() == 'mse':
                mean_ecg = np.median(data, axis=0)
                # get WOI indices
                idx_start = ref + woi[0]
                idx_end = ref + woi[1]
                idx_match = np.full((mean_ecg.shape[1], 2),
                                    np.iinfo(int).min,
                                    dtype=int
                                    )
                for i in range(mean_ecg.shape[1]):
                    mse = (np.square(data[:, idx_start:idx_end, i]
                                     - mean_ecg[idx_start:idx_end, i])
                           ).mean(axis=1)
                    idx_match[i, :] = [np.argmin(mse).astype(int), i]
                ecg = data[idx_match[:, 0], :, idx_match[:, 1]]
                ecg = ecg.T
            elif meth.lower() == 'ccf':
                # compare mean, median might result in all zeroes when WOI
                # is outside QRS
                mean_ecg = np.mean(data, axis=0)
                # get WOI indices
                idx_start = ref + woi[0]
                idx_end = ref + woi[1]
                # compute cross-correlation and select best match
                idx_match = np.full((mean_ecg.shape[1], 2),
                                    np.iinfo(int).min,
                                    dtype=int
                                    )
                corr = np.full(data.shape[0], np.nan, dtype=float)
                for i in range(mean_ecg.shape[1]):
                    for k in range(data.shape[0]):
                        mean_ecg_norm = np.linalg.norm(mean_ecg[idx_start:idx_end, i])
                        data_norm = np.linalg.norm(data[k, idx_start:idx_end, i])
                        corr[k] = np.correlate(
                            data[k, idx_start:idx_end, i] / data_norm,
                            mean_ecg[idx_start:idx_end, i] / mean_ecg_norm
                        )
                    idx_match[i, :] = [np.argmax(corr).astype(int), i]
                ecg = data[idx_match[:, 0], :, idx_match[:, 1]]
                ecg = ecg.T
            else:
                raise KeyError

            # build ECG traces
            traces = []
            for i, name in enumerate(ecg_names):
                traces.append(Trace(name=name, data=ecg[:, i], fs=1000.0))
            repr_ecg.append(BodySurfaceECG(method=meth,
                                           refAnnotation=ref,
                                           traces=traces))

        return repr_ecg

    def export_point_info(
            self,
            output_folder: str = '',
            points: Optional[List[CartoPoint]] = None
    ) -> None:
        """
        Export additional recording point info in DAT format.

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".

        Following data can is exported:
            NAME : point identifier
            REF : reference annotation
            WOI_START : window of interest, relative to REF
            WOI_END : window of interest, relative to REF

        By default, data from all valid points is exported, but also a
        list of EPPoints to use can be given.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.ptdata.<parameter>.pc.dat

        Parameters:
            output_folder : str (optional)
                path of the exported files
           points : list of CartoPoints (optional)
                EGM points to export

        Returns:
            None
        """

        log.info('exporting additional EGM point data')

        if not points:
            points = self.get_valid_points()
        if not len(points) > 0:
            log.warning('no points found in map {}. Nothing to export...'
                        .format(self.name))
            return

        # export point cloud first
        self.export_point_cloud(points=points, output_folder=output_folder)

        output_folder = self.resolve_export_folder(output_folder)
        basename = os.path.join(output_folder, self.name)

        # export data
        writer = FileWriter()

        data = np.array([point.name for point in points])
        dat_file = '{}.ptdata.NAME.pc.dat'.format(basename)
        f = writer.dump(dat_file, data)
        log.info('exported point names to {}'.format(f))

        data = np.array([point.refAnnotation for point in points])
        dat_file = '{}.ptdata.REF.pc.dat'.format(basename)
        f = writer.dump(dat_file, data)
        log.info('exported point reference annotation to {}'.format(f))

        data = np.array([point.woi[0] for point in points])
        dat_file = '{}.ptdata.WOI_START.pc.dat'.format(basename)
        f = writer.dump(dat_file, data)
        log.info('exported point WOI (start) to {}'.format(f))

        data = np.array([point.woi[1] for point in points])
        dat_file = '{}.ptdata.WOI_END.pc.dat'.format(basename)
        f = writer.dump(dat_file, data)
        log.info('exported point WOI (end) to {}'.format(f))

        return

    def export_point_ecg(
            self,
            output_folder: str = '',
            which: Optional[Union[str, List[str]]] = None,
            points: Optional[List[CartoPoint]] = None,
            reload_data: bool = False
    ) -> None:
        """
        Export surface ECG traces in IGB format. Overrides BaseClass method.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".

        By default, ECGs for all valid points are exported, but also a
        list of EPPoints to use can be given.

        If no ECG names are specified, 12-lead ECGs are exported

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.ecg.<trace>.pc.igb

        Parameters:
            output_folder : str (optional)
                path of the exported files
            which : string or list of strings
                ECG name(s) to include in IGB file.
            points : list of CartoPoints (optional)
                EGM points to export
            reload_data : bool
                reload ECG data if already loaded

        Returns:
            None
        """

        log.debug('preparing exporting point ECG data')

        if not points:
            points = self.get_valid_points()
        if not len(points) > 0:
            log.warning('no points found in map {}. Nothing to export...'
                        .format(self.name))
            return

        if not which:
            which = ['I', 'II', 'III',
                     'aVR', 'aVL', 'aVF',
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if isinstance(which, str):
            which = [which]

        # check if data is required
        missing_data = [p.is_ecg_data_required(which) for p in points]

        if any(missing_data) and not self.parent.is_root_valid():
            log.warning('valid study root is required to load ECG data!')
            return

        if any(missing_data):
            log.info('missing ECG data, loading...')
            missing_points = list(compress(points, missing_data))
            for i, point in enumerate(missing_points):
                # update progress bar
                console_progressbar(
                    i + 1, len(missing_points),
                    suffix='Loading ECG(s) for point {}'.format(point.name)
                )

                point.ecg.extend(point.load_ecg(which, reload=reload_data))

        # everything was imported, ready to save
        super().export_point_ecg(output_folder=output_folder,
                                 which=which,
                                 points=points)

        return

    def load_rf_data(
            self
    ) -> MapRF:
        """
        Load map associated RF data and RF contact forces.

        Returns:
            RF data : MapRF
        """

        log.info('loading RF and RF contact force data for map {}'
                 .format(self.name))

        # read RF data
        rf_abl = RFAblationParameters()
        rf_files = self.parent.repository.list_dir(
            self.parent.repository.join(''),
            regex='RF_' + self.name + '*'
        )

        if rf_files:
            rf_files = CartoMap._sort_rf_filenames(rf_files)
            rf_files = [self.parent.repository.join(f) for f in rf_files]
            log.debug('found {} RF files'.format(len(rf_files)))

            rf_columns = []
            rf_data = np.array([])
            for file in rf_files:
                with self.parent.repository.open(file, mode='rb') as f:
                    header = f.readline().decode(encoding=self.parent.encoding)
                    header = re.split(r'\t+', header.rstrip('\t\r\n'))
                    if not rf_columns:
                        # this is the first file
                        rf_columns = header
                        log.debug('found RF columns: {}'.format(rf_columns))
                        rf_data = np.empty((0, len(rf_columns)), dtype=np.int32)
                    if not header == rf_columns:
                        log.info('RF file header changed in file {}'
                                 .format(file))
                        continue

                    data = np.loadtxt(f, dtype=np.int32, skiprows=0)

                try:
                    data.shape[1]
                except IndexError:
                    # only 1 row of data in file
                    data = np.expand_dims(data, axis=0)

                rf_data = np.append(rf_data,
                                    data,
                                    axis=0)

            log.debug('read {} lines of RF data'.format(rf_data.shape[0]))
            if rf_data.size > 0:
                rf_abl = RFAblationParameters(time=rf_data[:, 0],
                                              irrigation=rf_data[:, 1],
                                              power_mode=rf_data[:, 2],
                                              abl_time=rf_data[:, 3],
                                              power=rf_data[:, 4],
                                              impedance=rf_data[:, 5],
                                              distal_temp=rf_data[:, 6],
                                              proximal_temp=rf_data[:, 7],
                                              )

        # read contact force in RF
        rf_force = RFForce()
        contact_force_rf_files = self.parent.repository.list_dir(
            self.parent.repository.join(''),
            regex='ContactForceInRF_' + self.name + '*'
        )

        contact_f_in_rf_columns = []
        contact_f_in_rf_data = np.array([])
        if contact_force_rf_files:
            contact_force_rf_files = CartoMap._sort_rf_filenames(
                contact_force_rf_files)
            contact_force_rf_files = [self.parent.repository.join(f)
                                      for f in contact_force_rf_files]
            log.debug('found {} RF contact force files'
                      .format(len(contact_force_rf_files)))

            # specify value converter for np.loadtxt that handles "," or "."
            conv = {
                1: lambda x: float(x.replace(',', '.'))
            }

            for file in contact_force_rf_files:
                with self.parent.repository.open(file, mode='rb') as f:
                    header = f.readline().decode(encoding=self.parent.encoding)
                    header = re.split(r'\t+', header.rstrip('\t\r\n'))
                    if not contact_f_in_rf_columns:
                        # this is the first file
                        contact_f_in_rf_columns = header
                        log.debug('found RF contact force columns: {}'
                                  .format(contact_f_in_rf_columns))
                        contact_f_in_rf_data = np.empty(
                            (0,
                             len(contact_f_in_rf_columns)),
                            dtype=np.int32
                        )
                    if not header == contact_f_in_rf_columns:
                        log.info('RF contact force file header changed in '
                                 'file {}'.format(file))
                        continue

                    data = np.loadtxt(f, skiprows=1, ndmin=2,
                                      converters=conv, encoding=None)

                contact_f_in_rf_data = np.append(contact_f_in_rf_data,
                                                 data,
                                                 axis=0)
            log.debug('read {} lines of RF contact force data'
                      .format(contact_f_in_rf_data.shape[0]))

        if contact_f_in_rf_data.size > 0:
            rf_force = RFForce(
                time=contact_f_in_rf_data[:, 0].astype(np.int32),
                force=contact_f_in_rf_data[:, 1].astype(np.float32),
                axial_angle=contact_f_in_rf_data[:, 2].astype(np.float32),
                lateral_angle=contact_f_in_rf_data[:, 3].astype(np.float32),
                abl_point=np.full(contact_f_in_rf_data.shape[0],
                                  -1,
                                  dtype=np.int32
                                  ),
                position=np.full((contact_f_in_rf_data.shape[0], 3),
                                 np.nan,
                                 dtype=np.float32
                                 )
            )

        # update rf data with point force data
        if self.points:
            log.info('updating RF dataset with EGM coordinates')
            for point in self.points:
                # add indices to the parent maps RF datasets referring back to
                # the ID and egmSurfX of this point
                if 'Ablation' in point.tags:
                    # this is an ablation point
                    point_id = int(point.name[1:])
                    point_coord = point.prjX

                    idx_min = np.argmin(np.abs(point.forceData.time))
                    acq_time = point.forceData.systemTime[idx_min]

                    sys_time = rf_force.time
                    idx_min = np.argmin(np.abs(sys_time - acq_time))
                    rf_force.ablationPoint[idx_min] = point_id
                    rf_force.position[idx_min, :] = point_coord
        else:
            log.info('no points found in map, cannot update RF dataset with '
                     'EGM coordinates!')

        return MapRF(force=rf_force, ablation_parameters=rf_abl)

    def _import_attributes(
            self
    ) -> None:
        """
        Load info and file(s) associated with this map from study XML.

        Returns:
            None
        """

        xml_file = self.parent.repository.join(self.studyXML)
        with self.parent.repository.open(xml_file) as fid:
            root = ET.parse(fid).getroot()

        map_item = [x for x in root.find('Maps').findall('Map')
                    if x.get('Name') == self.name]
        if not map_item:
            raise MapAttributeError('no map with name {} found in study XML'
                                    .format(self.name))
        if len(map_item) > 1:
            raise MapAttributeError('multiple maps with name {} found in '
                                    'study XML'
                                    .format(self.name))
        map_item = map_item[0]

        log.debug('reading map attributes')

        self.index = int(map_item.get('Index'))
        self.visible = map_item.get('Visible')
        self.type = map_item.get('Type')
        num_files = int(map_item.get('NumFiles'))
        if num_files == 0:
            raise MapAttributeError('no mesh file specified for map {}'
                                    .format(self.name))

        filenames = map_item.get('FileNames')
        if num_files > 1 or not filenames.lower().endswith('.mesh'):
            # TODO: handle filenames if more than one file
            raise MapAttributeError('Mesh file for map {} cannot be extracted '
                                    'from study XML'
                                    .format(self.name))
        self.surfaceFile = filenames
        self.volume = float(map_item.get('Volume'))
        self.RefAnnotationConfig = RefAnnotationConfig(
            algorithm=int(
                map_item.find('RefAnnotationConfig').get('Algorithm')
            ),
            connector=int(
                map_item.find('RefAnnotationConfig').get('Connector')
            )
        )

        # get map coloring range table
        colorRangeItem = map_item.find('ColoringRangeTable')
        colorRangeTable = []
        for colorRange in colorRangeItem.findall('ColoringRange'):
            colorRangeTable.append(
                ColoringRange(Id=int(colorRange.get('Id')),
                              min=float(colorRange.get('Min')),
                              max=float(colorRange.get('Max'))
                              )
            )
        self.coloringRangeTable = colorRangeTable

        return

    @staticmethod
    def _sort_rf_filenames(
            filenames: List[str],
            order: str = 'ascending'
    ) -> List[str]:
        """Sort a list of filenames."""

        names = [x.lower() for x in filenames]
        idx_sorted = np.arange(len(filenames))

        # determine the kind of files
        if any(names[0].startswith(x) for x in ['contactforceinrf_', 'rf_']):
            num_list = [int(re.sub(r'[^0-9]*', "", name.split('_')[-1]))
                        for name in names]
            idx_sorted = np.argsort(num_list)

        if order.lower() == 'descending':
            idx_sorted = np.flip(idx_sorted)

        return [filenames[i] for i in idx_sorted]


class CartoStudy(EPStudy):
    """
    Class representing a Carto3 study.

    Attributes:
        name : string
            name of the study given by Carto.
        studyXML : str
            filename of top-level XML describing the study
        mapNames : list of str
            names of the mapping procedures contained in data set
        mapPoints : list of int
            number of points recorded during mapping procedure
        maps : dict
            mapping procedures performed during study. Dictionary keys are
            the mapping procedure names (subset of mapNames attribute)
        meshes : list of Surface objects (optional)
            additional meshes from e.g. CT data
        units : CartoUnits
            units for distance and angle measures
        mappingParams : CartoMappingParameters
            mapping system settings, i.e. color tables, tag names, etc.
        visitag : Visitag
            ablation sites and ablation grid information
        paso : PaSo
            VT template matching data
        environment : Not Implemented
        externalObjects : Not Implemented

    Methods:
        import_study()
            import basic information about study
        import_maps(map_names=None, egm_names_from_pos=False)
            import mapping procedures by name
        import_visitag_sites(directory=None)
            import VisiTag ablation sites data
        load_visitag_grid(directory)
            load complete VisiTag ablation grid
        load()
            load pickled version of study
        export_additional_meshes(filename='')
            export additional meshes (e.g. CT) within study to VTK
        rfi_from_visitag_grid()
            (re)calculate ablation index from ablation grid
        is_root_valid(root_dir=None)
            check if repository is valid root. If root_dir is None,
            the current repository is checked.
        set_root(root_dir)
            check directory and set repository to given directory if valid

    """

    def __init__(
            self,
            study_repo: str,
            pwd: str = '',
            encoding: str = 'cp1252'
    ) -> None:
        """
        Constructor.

        Parameters:
            study_repo : str
                location of the study data, can be folder or ZIP archive
            pwd : bytes (optional)
                password for protected ZIP archives
            encoding : str (optional)
                file encoding used (all files are in binary mode).
                Default: cp1252

        Raises:
            FileNotFoundError : if study XML can not be located

        Returns:
            None

        """

        super().__init__(system='carto3',
                         study_repo=study_repo,
                         pwd=pwd,
                         encoding=encoding)

        self.studyXML = ''

        self.units = None
        self.environment = None  # TODO: is this relevant info?
        self.externalObjects = None  # TODO: is this relevant info?
        self.mappingParams = None

        # visitag data
        self.visitag = Visitag()
        self.paso = None

    def import_study(
            self
    ) -> None:
        """
        Load study details and basic information from study XML.

        Returns:
            None
        """

        # locate study XML
        log.info('Locating study XML in {}...'.format(self.repository))
        study_info = self.locate_study_xml(self.repository, pwd=self.pwd,
                                           encoding=self.encoding)
        if not study_info:
            log.warning('cannot locate study XML!')
            return

        log.info('found study XML at {}'.format(self.repository.root))
        self.studyXML = study_info['xml']
        self.name = study_info['name']

        log.info('accessing study XML: {}'.format(self.studyXML))
        log.info('gathering study information...')

        xml_path = self.repository.join(self.studyXML)
        with self.repository.open(xml_path) as fid:
            root = ET.parse(fid).getroot()

        log.debug('reading study units')
        study_units = root.find('Units')
        self.units = CartoUnits(Distance=study_units.get('Distance'),
                                Angle=study_units.get('Angle'))

        log.debug('reading study coloring table')
        study_parameters = root.find('Maps')
        item = study_parameters.find('ColoringTable')
        coloring_table = []
        for color in item.findall('Coloring'):
            coloring_table.append(
                Coloring(Id=int(color.get('Id')),
                         Name=color.get('Name'),
                         TextureInvert=int(color.get('TextureInvert')),
                         Propagation=int(color.get('Propagation')),
                         Units=color.get('Units')
                         )
            )

        log.debug('reading study surface error table')
        item = study_parameters.find('SurfaceErrorTable')
        surface_error = SurfaceErrorTable(
            BadErrorColor=[
                float(x) for x in item.get('BadErrorColor').split()
            ],
            MedErrorColor=[
                float(x) for x in item.get('MedErrorColor').split()
            ],
            GoodErrorColor=[
                float(x) for x in item.get('GoodErrorColor').split()
            ],
            BadErrorThreshold=float(item.get('BadErrorThreshold')),
            MedErrorThreshold=float(item.get('MedErrorThreshold')),
            GoodErrorThreshold=float(item.get('GoodErrorThreshold')))

        log.debug('reading study paso table')
        item = study_parameters.find('PasoTable')
        paso_table = PasoTable(ISName=item.get('ISName'))

        log.debug('reading study CFAE coloring table')
        item = study_parameters.find('CFAEColoringTable')
        cfae_coloring_table = CFAEColoringTable(
            IgnoreBelowColor=[
                float(x) for x in item.get('IgnoreBelowColor').split()
            ],
            IclMediumColor=[
                float(x) for x in item.get('IclMediumColor').split()
            ],
            IclHighColor=[
                float(x) for x in item.get('IclHighColor').split()
            ],
            IgnoreBelowThreshold=float(item.get('IgnoreBelowThreshold')),
            IclMediumThreshold=float(item.get('IclMediumThreshold')),
            IclHighThreshold=float(item.get('IclHighThreshold'))
        )

        log.debug('reading study tags table')
        item = study_parameters.find('TagsTable')
        tags_table = []
        for tag in item.findall('Tag'):
            tags_table.append(
                Tag(ID=int(tag.get('ID')),
                    ShortName=tag.get('Short_Name'),
                    FullName=tag.get('Full_Name'),
                    Color=[float(x) for x in tag.get('Color').split()],
                    Radius=float(tag.get('Radius'))
                    )
            )

        self.mappingParams = CartoMappingParameters(
            ColoringTable=coloring_table,
            SurfaceErrorTable=surface_error,
            PasoTable=paso_table,
            CFAEColoringTable=cfae_coloring_table,
            TagsTable=tags_table
        )

        log.debug('reading additional meshes')
        item = root.find('Meshes')
        if item:
            matrix = np.asarray(item.find('RegistrationMatrix').text.split(),
                                dtype=np.float32)
            meshes = []
            for mesh in item.findall('Mesh'):
                meshes.append(mesh.get('FileName'))
            self.meshes = Mesh(registrationMatrix=matrix, fileNames=meshes)
            log.info('found {} additional meshes in study'.format(len(meshes)))

        log.debug('reading study maps info')
        map_names = []
        map_points = []
        for item in root.iter('Map'):
            map_names.append(item.get('Name'))
            map_points.append(int(item.find('CartoPoints').get('Count')))
            log.debug('found map {} with {} mapping points'
                      .format(map_names[-1], map_points[-1]))

        self.mapNames = map_names
        self.mapPoints = map_points

    def import_maps(
            self,
            map_names: Optional[Union[str, List[str]]] = None,
            egm_names_from_pos: bool = False,
            *args, **kwargs
    ) -> None:
        """
        Import a Carto map. Extends BaseClass method.

        The mesh file is imported along with all recording points. Only
        meshes with mesh data are added to the study.

        If a map was already imported before and is part of the study,
        user interaction to reload is required.

        EGM names for recording points can be identified by evaluating the
        recording position to get the name of the electrode and comparing it
        to the name found in the points ECG file. Otherwise, the EGM name
        stored in a points ECG file is used.

        Parameters:
            map_names : list of str (optional)
                name or list of map names to import. If no name is
                specified, all maps are loaded (default).
            egm_names_from_pos : boolean (optional)
                Get EGM names from recording positions. (default is False)

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            None
        """

        # do some pre-import checks
        map_names = super().import_maps(map_names)

        # now load the maps
        for map_name in map_names:
            try:
                log.info('importing map {}:'.format(map_name))
                new_map = CartoMap(map_name, self.studyXML, parent=self)
                new_map.import_map(egm_names_from_pos=egm_names_from_pos)
                self.maps[map_name] = new_map
            except Exception as err:
                log.warning('failed to import map {}: {}'
                            .format(map_name, err))
                continue

        return

    def import_visitag_sites(
            self,
            directory: str = ''
    ) -> None:
        """
        Load VisiTag ablation sites data.

        If sites are already loaded and part of the study, user interaction
        is required.

        Parameters:
            directory : str (optional)
                path to folder containing VisiTag data
                (default is <study_root>/VisiTagExport)

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            None
        """

        visi_dir = directory if directory else 'VisiTagExport'
        if not self.repository.is_folder(self.repository.join(visi_dir)):
            log.warning('VisiTag folder {} not found'.format(visi_dir))
            return

        if self.visitag.sites:
            user_input = input('Visitag sites already loaded, reload? [Y/N] ')
            # input validation
            if user_input.lower() in ('y', 'yes'):
                log.debug('reloading Visitag sites')
            elif user_input.lower() in ('n', 'no'):
                log.debug('reload canceled ')
                return
            else:
                # ... error handling ...
                log.warning('Error: Input {} unrecognised.'.format(user_input))
                raise ValueError

        log.info('importing visitag ablation sites...')

        sites = []

        # import ablation sites from Sites.txt
        file = self.repository.join(visi_dir + '/' + 'Sites.txt')
        if not self.repository.is_file(file):
            log.warning('VisiTag Sites.txt not found')
        else:
            with self.repository.open(file, mode='rb') as fid:
                sites_data, sites_hdr = read_visitag_file(
                    fid, encoding=self.encoding)

            if not sites_data.size > 0:
                log.info('no ablation sites found in Sites.txt, '
                         'trying QMODE+...'
                         )
            else:
                for site in sites_data:
                    sites.append(
                        VisitagAblationSite(
                            int(site[sites_hdr.index('SiteIndex')]),
                            session_index=int(site[sites_hdr.index('Session')]),
                            channel_id=int(site[sites_hdr.index('ChannelID')]),
                            tag_index_status=int(site[sites_hdr.index('TagIndexStatus')]),
                            coordinates=np.array(
                                [site[sites_hdr.index('X')],
                                 site[sites_hdr.index('Y')],
                                 site[sites_hdr.index('Z')]]
                            ).astype(np.float32),
                            avg_force=site[sites_hdr.index('AverageForce')],
                            fti=site[sites_hdr.index('FTI')],
                            max_power=site[sites_hdr.index('MaxPower')],
                            max_temp=site[sites_hdr.index('MaxTemperature')],
                            duration=site[sites_hdr.index('DurationTime')],
                            base_impedance=site[sites_hdr.index('BaseImpedance')],
                            impedance_drop=site[sites_hdr.index('ImpedanceDrop')],
                            rf_index=RFIndex(
                                name='VisitagRFI',
                                value=site[sites_hdr.index('RFIndex')]
                            )
                        )
                    )

        # import ablation sites from QMODE+
        file = self.repository.join(visi_dir + '/' + 'Sites_QMODE+.txt')
        if not self.repository.is_file(file):
            log.warning('VisiTag Sites_QMODE+.txt not found')
        else:
            with self.repository.open(file, mode='rb') as fid:
                q_sites_data, q_sites_hdr = read_visitag_file(
                    fid, encoding=self.encoding)
            if not q_sites_data.size > 0:
                log.info('no ablation sites found in Sites_QMODE+.txt')
            else:
                for site in q_sites_data:
                    sites.append(
                        VisitagAblationSite(
                            int(site[q_sites_hdr.index('SiteIndex')]),
                            session_index=int(site[q_sites_hdr.index('Session')]),
                            channel_id=int(site[q_sites_hdr.index('ChannelID')]),
                            coordinates=np.array(
                                [site[q_sites_hdr.index('X')],
                                 site[q_sites_hdr.index('Y')],
                                 site[q_sites_hdr.index('Z')]]
                            ).astype(np.float32),
                            avg_force=site[q_sites_hdr.index('AverageForce')],
                            fti=site[q_sites_hdr.index('FTI')],
                            max_power=site[q_sites_hdr.index('MaxPower')],
                            max_temp=site[q_sites_hdr.index('MaxTemperature')],
                            duration=site[q_sites_hdr.index('DurationTime')],
                            base_impedance=site[q_sites_hdr.index('BaseImpedance')],
                            impedance_drop=site[q_sites_hdr.index('ImpedanceDrop')],
                            rf_index=RFIndex(
                                name='VisitagFTI',
                                value=site[q_sites_hdr.index('FTI')]
                            )
                        )
                    )

        # check if any data was loaded
        if not len(sites) > 0:
            log.warning('no visitag data found in files! Aborting...')

        self.visitag.sites = sites

    def import_paso(
            self,
            directory: str = ''
    ) -> None:
        """
        Load PaSo tables.

        Parameters:
            directory : str (optional)
                path to folder containing PaSo data
                (default is <study_root>/PaSoExport)

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            None
        """

        paso_dir = directory if directory else 'PaSoExport'
        if not self.repository.is_folder(self.repository.join(paso_dir)):
            log.warning('PaSo folder {} not found'.format(paso_dir))
            return

        if self.paso:
            user_input = input('PaSo data already loaded, reload? [Y/N] ')
            # input validation
            if user_input.lower() in ('y', 'yes'):
                log.debug('reloading PaSo data')
            elif user_input.lower() in ('n', 'no'):
                log.debug('reload canceled ')
                return
            else:
                # ... error handling ...
                log.warning('Error: Input {} unrecognised.'.format(user_input))
                raise ValueError

        log.info('importing PaSo templates...')

        # read configuration
        file = self.repository.join(paso_dir + '/' + 'ConfigurationData.txt')
        if not self.repository.is_file(file):
            log.warning('PaSo ConfigurationData.txt not found! Aborting...')
            return

        with self.repository.open(file, mode='rb') as fid:
            paso_config = read_paso_config(fid, encoding=self.encoding)

        # read templates
        templates = []
        is_files = [f for f in self.repository.list_dir(self.repository.join(paso_dir))
                    if (f.startswith(paso_config.isDefaultPrefix)
                        or f.startswith(paso_config.pmDefaultPrefix))
                    and 'Correlations' not in f
                    ]
        log.info('found {} PaSo templates'.format(len(is_files)))

        for i, filename in enumerate(is_files):
            # update progress bar
            console_progressbar(
                i + 1, len(is_files),
                suffix='Loading PaSo templates {}'.format(filename)
            )

            new_template = PaSoTemplate()
            file = self.repository.join(paso_dir + '/' + filename)
            with self.repository.open(file, mode='rb') as fid:
                new_template.load(fid, encoding=self.encoding)
            templates.append(new_template)

        # read correlation files
        correlations = []
        corr_files = [f for f in self.repository.list_dir(self.repository.join(paso_dir))
                      if 'Correlations' in f
                      ]
        log.info('found {} PaSo correlations'.format(len(corr_files)))
        for i, filename in enumerate(corr_files):
            # update progress bar
            console_progressbar(
                i + 1, len(corr_files),
                suffix='Loading PaSo correlations {}'.format(filename)
            )

            file = self.repository.join(paso_dir + '/' + filename)
            with self.repository.open(file, mode='rb') as fid:
                correlations.extend(
                    read_paso_correlations(fid, encoding=self.encoding)
                )

        self.paso = PaSo(
            configuration=paso_config,
            templates=templates,
            correlations=correlations
        )

    def import_visitag_grid(
            self,
            directory: str = ''
    ) -> None:
        """
        Import VisiTag ablation grid data.

        If sites are already loaded and part of the study, user interaction
        is required.

        Parameters:
            directory : string (optional)
                path to folder containing VisiTag data
                (default is <study_root>/VisiTagExport)

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            None
        """

        visi_dir = directory if directory else 'VisiTagExport'
        if not self.repository.is_folder(self.repository.join(visi_dir)):
            log.warning('VisiTag folder {} not found'.format(visi_dir))
            return

        if self.visitag.grid:
            user_input = input('Visitag grid already loaded, reload? [Y/N] ')
            # input validation
            if user_input.lower() in ('y', 'yes'):
                log.info('reloading Visitag grid')
            elif user_input.lower() in ('n', 'no'):
                log.info('reload canceled ')
                return
            else:
                # ... error handling ...
                log.warning('Error: Input {} unrecognised.'.format(user_input))
                raise ValueError

        log.info('importing visitag grid data...')

        # get grid data
        # first get ablation sites
        file = self.repository.join(visi_dir + '/' + 'AblationSites.txt')

        if not self.repository.is_file(file):
            log.warning('No VisiTag grid data found in {}'.format(visi_dir))
            return

        with self.repository.open(file, mode='rb') as fid:
            abl_site_data, abl_site_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )
        if not abl_site_data.size > 0:
            # TODO: implement visitag tag grid import from QMODE+
            log.warning('no grid data found! Probably QMODE+ was used, '
                        'not implemented yet...')
            return
        n_sites = abl_site_data.shape[0]
        cols = get_col_idx_from_header(abl_site_hdr, 'SiteIndex')
        site_index = abl_site_data[:, cols].astype(int).ravel()
        cols = get_col_idx_from_header(abl_site_hdr, 'Session')
        session = abl_site_data[:, cols].astype(int).ravel().tolist()
        cols = get_col_idx_from_header(abl_site_hdr, 'FirstPosTimeStamp')
        first_pos_tstamp = abl_site_data[:, cols].astype(int).ravel().tolist()
        cols = get_col_idx_from_header(abl_site_hdr, 'FirstPosPassedFilterTimeStamp')
        first_pos_passed_tstamp = abl_site_data[:, cols].astype(int).ravel().tolist()
        cols = get_col_idx_from_header(abl_site_hdr, 'LastPosTimeStamp')
        last_pos_tstamp = abl_site_data[:, cols].astype(int).ravel().tolist()

        log.info('found {} ablation sites'.format(n_sites))

        # load grid data
        log.info('load grid data. This might take a while...')

        file = self.repository.join(visi_dir + '/' + 'PositionsData.txt')
        if not self.repository.is_file(file):
            log.warning('No VisiTag positions data found in {}'
                        .format(visi_dir)
                        )
            return
        with self.repository.open(file, mode='rb') as fid:
            pos_data, pos_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )

        file = self.repository.join(visi_dir + '/' + 'ContactForceData.txt')
        if not self.repository.is_file(file):
            log.warning('No VisiTag contact force data found in {}'
                        .format(visi_dir)
                        )
            return
        with self.repository.open(file, mode='rb') as fid:
            force_data, force_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )

        file = self.repository.join(visi_dir + '/' + 'AllPositionInGrids.txt')
        if not self.repository.is_file(file):
            log.warning('No VisiTag grid positions data found in {}'
                        .format(visi_dir)
                        )
            return
        with self.repository.open(file, mode='rb') as fid:
            grid_pos_data, grid_pos_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )

        file = self.repository.join(visi_dir + '/' + 'Grids.txt')
        if not self.repository.is_file(file):
            log.warning('No VisiTag grids found in {}'
                        .format(visi_dir)
                        )
            return
        with self.repository.open(file, mode='rb') as fid:
            grid_data, grid_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )

        # extract parameters
        cols = get_col_idx_from_header(grid_pos_hdr, 'SiteIndex')
        pos_site_index = grid_pos_data[:, cols].astype(int).ravel()
        # sanity check, if SiteIndex is same in PositionsData and AblationSites
        if not np.array_equal(site_index, np.unique(pos_site_index)):
            log.info('SiteIndex is different in files! Aborting...')
            return

        # get all unique IDs from Grids.txt, needed later to extract
        # coordinates for unique IDs
        cols = get_col_idx_from_header(grid_hdr, 'UniqID')
        grid_uid = grid_data[:, cols].astype(int).ravel()
        # get all timestamps from ContactForceData.txt, needed later to extract
        # force data for unique IDs
        cols = get_col_idx_from_header(force_hdr, 'Time')
        force_tstamp = force_data[:, cols].astype(int).ravel()

        grid_sites = []
        for i, site in enumerate(site_index.tolist()):
            # update progress bar
            console_progressbar(
                i+1, n_sites,
                suffix='Processing Visitag site {}'.format(site)
            )
            # instantiate for this site
            grid = VisitagAblationGrid(
                site,
                session=session[i],
                first_pos_time_stamp=first_pos_tstamp[i],
                first_pos_passed_filter_time_stamp=first_pos_passed_tstamp[i],
                last_pos_time_stamp=last_pos_tstamp[i]
            )
            # get rows where data for this ablation site is referenced
            rows = np.asarray(pos_site_index == site).nonzero()[0]
            # now get the data for this ablation site
            cols = get_col_idx_from_header(grid_pos_hdr, 'PosTimeStamp')
            pos_tstamp = grid_pos_data[rows, cols].astype(int).ravel()
            cols = get_col_idx_from_header(grid_pos_hdr, 'UniqID')
            pos_uid = grid_pos_data[rows, cols].astype(int).ravel()
            unique_u_id = np.unique(pos_uid)

            # for example there might be 41 unique IDs for site 1
            # we need to work with every unique ID
            grid_points = []
            for this_id in unique_u_id:
                # first get a list of timestamps for this unique index
                rows = np.asarray(pos_uid == this_id).nonzero()[0]
                uid_tstamp = pos_tstamp[rows]

                # now use the timestamps to get data from other files
                # locate timestamps in PositionData.txt
                cols = get_col_idx_from_header(pos_hdr, 'TimeStamp')
                rows = np.argwhere(np.in1d(pos_data[:, cols], uid_tstamp))
                # get the data from PositionsData.txt
                cols = get_col_idx_from_header(pos_hdr, 'TimeStamp')
                gtime = pos_data[rows, cols].astype(np.int32)
                # gindex_name = 'AblationIndex'
                cols = get_col_idx_from_header(pos_hdr, 'RFIndex')
                uid_rf_index = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'Impedance')
                uid_imp = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'ImpedanceDrop')
                uid_imp_drop = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'Temperature')
                uid_temp = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'Power')
                uid_power = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'Force')
                uid_force = pos_data[rows, cols].ravel()
                cols = get_col_idx_from_header(pos_hdr, 'Passed')
                uid_passed = pos_data[rows, cols].astype(int).ravel()

                # get force time stamp and get force data from
                # ContactForceData.txt
                cols = get_col_idx_from_header(pos_hdr, 'ForceTimeStamp')
                uid_force_tstamp = pos_data[rows, cols]
                f_rows = np.asarray(force_tstamp == uid_force_tstamp).nonzero()[0]
                cols = get_col_idx_from_header(force_hdr, 'AxialAngle')
                uid_axial_angle = force_data[f_rows, cols].ravel()
                cols = get_col_idx_from_header(force_hdr, 'LateralAngle')
                uid_lateral_angle = force_data[f_rows, cols].ravel()

                # use the UniqID to find coordinates in Grids.txt
                g_rows = np.asarray(grid_uid == this_id).nonzero()[0]
                cols = get_col_idx_from_header(grid_hdr, ['X', 'Y', 'Z'])
                uid_X = grid_data[g_rows[:, None], cols]

                grid_points.append(
                    VisitagGridPoint(coordinates=uid_X,
                                     time=gtime,
                                     temperature=uid_temp,
                                     power=uid_power,
                                     force=uid_force,
                                     axial_angle=uid_axial_angle,
                                     lateral_angle=uid_lateral_angle,
                                     base_impedance=uid_imp,
                                     impedance_drop=uid_imp_drop,
                                     rf_index=RFIndex(
                                         name='CartoAblationIndex',
                                         value=uid_rf_index),
                                     passed=uid_passed
                                     )
                )

            # add all grid points to visitag grid
            grid.add_points(grid_points)
            # add this grid to list of grids
            grid_sites.append(grid)

        self.visitag.grid = grid_sites

    @classmethod
    def load(
            cls,
            file: str,
            repository_path: str = '',
            password: str = ''
    ) -> TCartoStudy:
        """
        Load study from file. Overrides BaseClass method.

        A previously saved version of a CartoStudy object can be
        loaded. The objects <study_root> is set to the one stored in the
        file if valid. If not, the folder of the PKL is set as root
        directory.
        The path to the Carto files can also be specified explicitly.

        Parameters:
            file : str
                location of .pyceps file
            repository_path : str
                set repository root to this location
            password : str

        Raises:
            TypeError : If file is not Carto3

        Returns:
            CartoStudy
        """

        log.debug('loading study')

        with open(file) as fid:
            root = ET.parse(fid).getroot()

        # check if file was generated from Carto3 data
        system = root.get('system')
        if not system.lower() == "carto3":
            raise TypeError('expected Carto3 system file, found {}'
                            .format(system))

        # create empty class instance
        repo = root.find('Repository')
        base_path = repo.get('base')
        if not os.path.exists(base_path) and not os.path.isfile(base_path):
            log.warning('repository path save in pyCEPS file can not be '
                        'reached!\n'
                        'Trying to initialize with file location...')
            base_path = file

        study = cls(base_path,
                    pwd=password,
                    encoding=repo.get('encoding'))

        # load basic info
        study.name = root.get('name')
        study.studyXML = root.get('studyXML')
        units = root.find('Units')
        study.units = CartoUnits(units.get('distance'), units.get('angle'))

        VALID_ROOT = False
        if repository_path:
            # try to set root if explicitly given
            log.info('trying to set study root to {}'.format(repository_path))

            if study.set_repository(os.path.abspath(repository_path)):
                log.info('setting study root to {}'.format(repository_path))
                VALID_ROOT = True
            else:
                log.info('cannot set study root to {}\n'
                         'Trying to use root information from file'
                         .format(repository_path)
                         )

        if not VALID_ROOT:
            # try to re-set previous study root
            base_path = root.find('Repository').get('root')
            log.info('trying to set study root to root from file: {}'
                     .format(base_path)
                     )

            if study.set_repository(base_path):
                log.info('previous study root is still valid ({})'
                         .format(study.repository.root)
                         )
                VALID_ROOT = True

        if not VALID_ROOT:
            # try to search for studyXML in current location or at folder above
            cur_dir = os.path.dirname(file)
            log.info('no valid study root found so far, trying to search for '
                     'repository at file location {}'
                     .format(cur_dir))

            # search in current pyCEPS file folder
            filenames = [f for f in os.listdir(cur_dir)
                         if (os.path.isfile(os.path.join(cur_dir, f))
                             and (zipfile.is_zipfile(os.path.join(cur_dir, f))
                                  or py7zr.is_7zfile(os.path.join(cur_dir, f))
                                  )
                             )
                         or os.path.isdir(os.path.join(cur_dir, f))
                         ]
            for file in filenames:
                try:
                    if study.set_repository(os.path.join(cur_dir, file)):
                        VALID_ROOT = True
                        break
                except:
                    # some error occurred, don't care what exactly,
                    # just continue
                    continue

            if not VALID_ROOT:
                # search in folder above
                log.info('searching in folder above file location...')
                cur_dir = os.path.abspath(os.path.join(cur_dir, '..'))
                filenames = [f for f in os.listdir(cur_dir)
                             if (os.path.isfile(os.path.join(cur_dir, f))
                                 and (zipfile.is_zipfile(os.path.join(cur_dir, f))
                                      or py7zr.is_7zfile(
                                        os.path.join(cur_dir, f))
                                      )
                                 )
                             or os.path.isdir(os.path.join(cur_dir, f))
                             ]
                for file in filenames:
                    try:
                        if study.set_repository(os.path.join(cur_dir, file)):
                            VALID_ROOT = True
                            break
                    except:
                        # some error occurred, don't care what exactly,
                        # just continue
                        continue

        if not VALID_ROOT:
            # no valid root found so far, set to pkl directory
            log.warning(
                'no valid study root found. Using file location!'.upper()
            )
            study.repository.base = os.path.abspath(file)
            study.repository.root = os.path.dirname(os.path.abspath(file))

        # load mapping procedures
        proc_item = root.find('Procedures')
        num_procedures = proc_item.get('count')
        sep = chr(int(proc_item.get('sep')))
        study.mapNames = proc_item.get('names').split(sep)
        study.mapPoints = [int(x) for x in proc_item.get('points').split(sep)]

        for proc in proc_item.iter('Procedure'):
            name = proc.get('name')

            new_map = CartoMap(name, study.studyXML, parent=study)
            new_map.surfaceFile = proc.get('meshFile')
            new_map.volume = float(proc.get('volume'))

            # load mesh
            mesh_item = proc.find('Mesh')
            if mesh_item:
                new_map.surface = Surface.load_from_xml(mesh_item)
            else:
                log.warning('no surface data found in XML!')

            # load BSECGs
            new_map.bsecg = xml_load_binary_bsecg(proc.find('BSECGS'))

            # load lesions
            lesions_item = proc.find('Lesions')
            if mesh_item:
                new_map.lesions = Lesions.load_from_xml(lesions_item)
            else:
                log.info('no lesion data found in XML')

            # load EGM points
            p_data = {}
            points_item = proc.find('Points')
            num_points = int(points_item.get('count'))

            if num_points > 0:
                for arr in points_item.findall('DataArray'):
                    d_name, data = xml_load_binary_data(arr)
                    p_data[d_name] = data
                for arr in points_item.findall('Traces'):
                    d_name, data = xml_load_binary_trace(arr)
                    p_data[d_name] = data

                points = []
                for i in range(num_points):
                    new_point = CartoPoint('dummy', parent=new_map)
                    for key, value in p_data.items():
                        if hasattr(new_point, key):
                            setattr(new_point, key, value[i])
                        else:
                            log.warning('cannot set attribute "{}" '
                                        'for CartoPoint'
                                        .format(key)
                                        )
                    points.append(new_point)
                new_map.points = points

            # now we can add the procedure to the study
            study.maps[name] = new_map

        # load additional meshes
        mesh_item = root.find('AdditionalMeshes')
        if mesh_item:
            _, reg_matrix = xml_load_binary_data(
                [x for x in mesh_item.findall('DataArray')
                 if x.get('name') == 'registrationMatrix'][0]
            )
            _, file_names = xml_load_binary_data(
                [x for x in mesh_item.findall('DataArray')
                 if x.get('name') == 'fileNames'][0]
            )
            study.meshes = Mesh(registrationMatrix=reg_matrix,
                                fileNames=file_names
                                )

        # load PaSo
        paso_item = root.find('PaSo')
        if paso_item:
            study.paso = PaSo.load_from_xml(paso_item)

        return study

    def save(
            self,
            filepath: str = '',
            keep_ecg: bool = False
    ) -> str:
        """
        Save study object as .pyceps archive.
        Note: File is only created if at least one map was imported!

        By default, the filename is the study's name, but can also be
        specified by the user.
        If the file already exists, user interaction is required to either
        overwrite file or specify a new file name.

        Parameters:
            filepath : string (optional)
                custom path for the output file
            keep_ecg : bool
                export point ECG data

        Raises:
            ValueError : If user input is not recognised

        Returns:
            str : file path .pyceps was saved to
        """

        # first check if all ECG data is loaded and load if needed
        if keep_ecg:
            log.info('ECG export requested, performing some checks first...')
            # all point ECGs must contain the same channels for export
            ecg_names = ['I', 'II', 'III',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                         'aVL', 'aVR', 'aVF'
                         ]
            for cmap in self.maps.values():
                # check if data is required
                points = cmap.points
                missing_data = [p.is_ecg_data_required(ecg_names)
                                for p in points
                                ]

                if any(missing_data) and not self.is_root_valid():
                    log.warning('valid study root is required to load ECG '
                                'data!\n'
                                '')
                    keep_ecg = False
                    break

                if any(missing_data):
                    log.info('missing ECG data, loading...')
                    missing_points = list(compress(points, missing_data))
                    for i, point in enumerate(missing_points):
                        # update progress bar
                        console_progressbar(
                            i + 1, len(missing_points),
                            suffix='Loading ECG(s) for point {}'.format(point.name)
                        )

                        point.ecg.extend(
                            point.load_ecg(ecg_names, reload=False)
                        )

        # add basic information to XML
        root, filepath = super().save(filepath, keep_ecg=keep_ecg)

        if not root:
            # no base info was created (no maps imported), nothing to add
            return filepath

        # add Carto specific data
        root.set('studyXML', self.studyXML)
        ET.SubElement(root, 'Units',
                      distance=self.units.Distance,
                      angle=self.units.Angle
                      )

        if self.paso:
            self.paso.add_to_xml(root)

        for key, cmap in self.maps.items():
            map_item = [p for p in root.iter('Procedure')
                        if p.get('name') == key][0]

            # add additional procedure info
            map_item.set('meshFile', cmap.surfaceFile)
            map_item.set('volume', str(cmap.volume))

            # add additional point info
            point_item = map_item.find('Points')
            to_add = ['pointFile', 'ecgFile', 'forceFile', 'uniX']
            for name in to_add:
                data = [getattr(p, name) for p in cmap.points]
                xml_add_binary_numpy(point_item, name, np.array(data))

        # make XML pretty
        dom = minidom.parseString(ET.tostring(root))
        xml_string = dom.toprettyxml(encoding='utf-8')

        # write XML
        with open(filepath, 'wb') as fid:
            fid.write(xml_string)

        log.info('saved study to {}'.format(filepath))
        return filepath

    def export_additional_meshes(
            self,
            output_folder: str = ''
    ) -> None:
        """
        Export additional meshes and registration matrix in study to VTK.
        Overrides BaseClass method.

        If additional meshes, e.g. from CT, are part of the study, they are
        exported in VTK format along with the registration matrix as .YAML

        The name of the mesh is preserved in the VTK filename except for
        meshes with no name (.mesh) which are exported as noname.VTK.
        The registration matrix is exported as RegistrationMatrix.yaml

        If meshes were exported before, they are not exported again. If
        anything changed in the meshes, and you want to export it, delete old
        files first.

        If no filename is specified, export all meshes to the folder above
        the study_root.

        Parameters:
            output_folder : str (optional)
                path to export file, export to default location if not given

        Returns:
            None
        """

        if not self.is_root_valid():
            log.warning('a valid study root is necessary to dump additional '
                        'meshes!')
            return

        if not self.meshes:
            log.info('no additional meshes found in study, nothing to export')
            return

        basename = self.resolve_export_folder(
            os.path.join(output_folder, 'additionalMeshes')
        )

        # export registration matrix
        log.debug('exporting registration matrix')
        matrix_str = ['{:.7f}'.format(v)
                      for v in self.meshes.registrationMatrix
                      ]
        with open(os.path.join(basename, 'RegistrationMatrix.yaml'), 'w') as f:
            f.write('affine transform:\n  ' + ' '.join(matrix_str))

        # check if meshes were already exported
        mesh_names = [f if not f == '.mesh' else 'noname.mesh'
                      for f in self.meshes.fileNames
                      ]
        filenames = [os.path.join(basename, f.split('.mesh')[0] + '.vtk')
                     for f in mesh_names]
        export_files = [os.path.basename(f)
                        if not os.path.basename(f) == 'noname.vtk' else '.vtk'
                        for f in filenames
                        if not os.path.isfile(f)]

        # export meshes
        log.debug('found {} unsaved meshes in study, exporting as VTK'
                  .format(len(export_files)))

        for file in export_files:
            f_loc = self.repository.join(file.split('.vtk')[0] + '.mesh')
            # for ZIP roots and meshes with no name (i.e. ".mesh") path is
            # incorrect, so fix
            if isinstance(f_loc, zipfile.Path) and f_loc.at.endswith('/'):
                # files without name (extension only) are interpreted as
                # folders, remove trailing "/"
                f_loc.at = f_loc.at[:-1]

            with self.repository.open(f_loc, mode='rb') as fid:
                surface = read_mesh_file(fid)

            export_file = os.path.join(basename, file)
            # treat meshes with no name, i.e. ".mesh"
            _, ext = os.path.splitext(export_file)
            if not ext:
                export_file = os.path.join(basename, 'noname' + file)

            # now we can export the mesh
            f = surface.dump_mesh_carp(os.path.splitext(export_file)[0])
            log.info('exported anatomical shell to {}'
                     .format(f + ' (.pts, .elem)'))
            surf_maps = surface.get_map_names()
            surf_labels = surface.get_label_names()
            surface.dump_mesh_vtk(export_file,
                                  maps_to_add=surf_maps,
                                  labels_to_add=surf_labels
                                  )

    def rfi_from_visitag_grid(
            self
    ) -> None:
        """
        Calculate RF index for VisiTag sites from VisiTag grid data.

        RFI values are added to the study's VisiTag sites data with name
        "CustomRFI".

        Returns:
            None
        """

        log.info('(re)calculating RF index...')

        if not self.visitag.sites:
            log.warning('no VisiTag sites found, import first!')
        if not self.visitag.grid:
            log.warning('no VisiTag grid data found, import first!')

        for i, site in enumerate(self.visitag.sites):
            # update progress bar
            console_progressbar(
                i+1, len(self.visitag.sites),
                suffix='Processing site {}'.format(site.siteIndex)
            )

            # get VisiTag grid data for this site
            grid = [g for g in self.visitag.grid
                    if g.siteIndex == site.siteIndex]

            if len(grid) == 0:
                log.warning('no grid data found for VisiTag site {}'
                            .format(site.siteIndex))
            elif len(grid) > 1:
                log.warning('found multiple grids with same index for VisiTag '
                            'site {}'.format(site.siteIndex))
            else:
                site.add_rf_index(grid[0].calc_rfi())

    def is_root_valid(
            self,
            root_dir: str = '',
            pwd: str = ''
    ) -> bool:
        """
        Check if study root is valid. Overrides BaseClass method.

        Parameters:
            root_dir : str (optional)
                path to check. If not specified, the current study root
                is checked.
            pwd : bytes

        Returns:
            bool : valid or not
        """

        log.info('checking if study root{}is valid'
                 .format(' ' + root_dir + ' ' if root_dir else ' '))

        studyXML = self.repository.join(self.studyXML)
        if not root_dir and self.repository.is_file(studyXML):
            # root saved in study is valid, nothing to do
            return True
        elif root_dir:
            try:
                tmp_root = Repository(root_dir, pwd=pwd)
            except FileNotFoundError:
                # repository can not be found, so it's invalid
                return False

            if not tmp_root.root:
                # dummy repo was not initialized properly, so root is invalid
                return False
            return self.locate_study_xml(tmp_root, pwd=pwd) is not None

        return False

    def set_repository(
            self,
            root_dir: str
    ) -> bool:
        """
        Change path to root directory. Overrides BaseClass method.
        If new root directory is invalid, it is not changed.

        Parameters:
            root_dir : string
                new root directory

        Returns:
            bool : successful or not
        """

        log.info('setting study root to new directory {}'.format(root_dir))

        study_root = os.path.abspath(root_dir)
        if not self.is_root_valid(study_root):
            log.warning('root directory is invalid: {}'.format(study_root))
            return False

        # study XML was found, check if it is the same study
        root = Repository(root_dir)
        study_info = self.locate_study_xml(root, pwd=self.pwd,
                                           encoding=self.encoding)
        if not study_info:
            # should never happen...
            raise FileNotFoundError

        if not self.studyXML == study_info['xml']:
            log.warning('name of study XML differs, will not change root!')
            return False
        if not self.name == study_info['name']:
            log.warning('name of study differs, will not change root!')
            return False

        # change study root
        self.repository = root
        log.info('found study XML at {}'.format(self.repository.root))

        return True

    @staticmethod
    def locate_study_xml(
            repository: Repository,
            pwd: str = '',
            regex: str = r'^((?!Export).)*.xml$',
            encoding: str = 'cp1252'
    ) -> Optional[dict]:
        """
        Locate study XML in Carto repository. A file is considered valid if
        it starts with '<Study name='.

        Parameters:
            repository : Repository
                This is searched recursively
            pwd : bytes (optional)
                password for protected ZIP archives.
            regex: str literal (optional)
                regular expression used for search
            encoding : str (optional)

        Raises:
            TypeError : if study repository is not of type Repository

        Returns:
            dict
                'xml' : filepath to XML file
                'name' : study name retrieved from XML
        """

        log.debug('searching for Study XML in: {}'.format(repository))

        if not isinstance(repository, Repository):
            raise TypeError

        # search base folder
        file_matches = repository.list_dir(repository.join(''), regex=regex)
        log.debug('found matches: {}'.format(file_matches))

        for f in file_matches:
            with repository.open(repository.join(f), mode='rb') as fid:
                line = fid.readline().decode(encoding=encoding)
                if line.startswith('<Study name='):
                    # found XML, return info
                    return {'xml': f,
                            'name': re.search('<Study name="(.*)">',
                                              line).group(1)
                            }

        # study xml not found, try subdirectories
        folders = [f for f in repository.list_dir(repository.join(''))
                   if repository.is_folder(repository.join(f))
                   or repository.is_archive(repository.join(f))
                   ]
        log.debug('found subdirectories: {}'.format(folders))

        for folder in folders:
            # update root location and start new search there
            repository.update_root(repository.join(folder))
            return CartoStudy.locate_study_xml(repository)

        # XML was nowhere to be found
        return None
