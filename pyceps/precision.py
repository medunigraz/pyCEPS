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
import copy
from typing import Optional, Union, List, TypeVar
import zipfile
import py7zr
from xml.etree import ElementTree as ET
from xml.dom import minidom
import numpy as np
import re

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
from pyceps.datatypes.precision.precisiontypes import DetectionAlgorithm
from pyceps.datatypes.signals import Trace, BodySurfaceECG
from pyceps.datatypes.lesions import Lesions, RFIndex, AblationSite
from pyceps.fileio.precisionio import (
    read_landmark_geo, load_dxl_data,
    load_ecg_data, load_lesion_data
)
from pyceps.datatypes.exceptions import MeshFileNotFoundError
from pyceps.utils import console_progressbar


# workaround to type hint self
TPrecisionStudy = TypeVar('TPrecisionStudy', bound='PrecisionStudy')


log = logging.getLogger(__name__)


class PrecisionPoint(EPPoint):
    """
    Class representing a Precision mapping point.

    Attributes:
        name : str
            identifier for this recording point
        parent : PrecisionMap
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
        uniX : ndarray (3, )
            cartesian coordinates of the second unipolar recording electrode
            NOTE: coordinates of second unipolar electrode are same as recX if
            position cannot be determined
        egmRef : Trace
            reference trace
        ecg : list of Trace
            ecg traces for this point
        impedance : float
        force : float
        catheterName : str
            name of the catheter used to record this point
        electrodeName :
            name of the recording electrode
        displayed : bool
            flag if point was displayed in Precision
        utilized : bool
            flag if point was used to create map
        exportedSeconds : float
            exported data length in (msec)
        endTime : float
        pNegativeVoltage : float
            peak negative voltage in (mV)
        CFEMean : float
            CFE mean measures in (msec)
        CFEstd : float
            CFE standard deviation measures in (msec)
        CFEDetection : CFEDetection
            information about CFE detection
        algorithm : dict
            'rov' : DetectionAlgorithm for the roving trace
            'ref' : DetectionAlgorithm for the reference trace
    """

    def __init__(
            self,
            name: str,
            coordinates: np.ndarray = np.full((3, 1), np.nan, dtype=float),
            parent: Optional['PrecisionMap'] = None
    ) -> None:
        """
        Constructor.

        Parameters:
             name : str
                name / identifier for this point
            coordinates : ndarray(3, )
                cartesian coordinates of recording position
            parent : PrecisionMap (optional)
                the map this point belongs to

        Raises:
            TypeError : if parent is not of type PrecisionMap

        Returns:
            None
        """

        super().__init__(name, coordinates=coordinates, parent=parent)
        # explicitly set parent for correct type hinting
        if parent is not None and not isinstance(parent, PrecisionMap):
            raise TypeError('Cannot set parent for PrecisionPoint of type {}'
                            .format(type(parent))
                            )
        self.parent = parent

        # add Precision specific attributes
        self.catheterName = ''
        self.electrodeName = ''

        self.displayed = False
        self.utilized = False
        self.exportedSeconds = np.nan
        self.endTime = np.nan
        self.pNegativeVoltage = np.nan
        self.CFEMean = np.nan
        self.CFEstd = np.nan
        self.CFEDetection = None
        self.algorithm = {}

    def is_valid(
            self
    ) -> bool:
        """Check if point was used for map generation."""

        return self.utilized


class PrecisionMap(EPMap):
    """
    Class representing Precision map.

    Attributes:
            name : str
                name of the mapping procedure
            parent : subclass of EPStudy
                the parent study for this map
            surfaceFile : str
                filename of file containing the anatomical shell data
            location : str
                path to map data within repository
            surface : Surface
                triangulated anatomical shell
            points : list of subclass EPPoints
                the mapping points recorded during mapping procedure
            bsecg : list of BodySurfaceECG
                body surface ECG data for the mapping procedure
            lesions : Lesions
                ablation data for this mapping procedure
            ablationSites = list of PrecisionLesion
                ablation data imported from study
    """

    def __init__(
            self,
            name: str,
            location: str = '',
            parent: Optional['PrecisionStudy'] = None
    ) -> None:
        """
        Constructor.

        Parameters:
            name : str
                name of the mapping procedure
            location : str
                path to map data within repository
            parent : PrecisionStudy (optional)
                study this map belongs to

        Raises:
            TypeError : if parent is not of type CartoStudy

        Returns:
            None
        """

        super().__init__(name, parent=None)
        # explicitly set parent for correct type hinting
        if parent is not None and not isinstance(parent, PrecisionStudy):
            raise TypeError('Cannot set parent for PrecisionMap of type {}'
                            .format(type(parent))
                            )
        self.parent = parent

        # add Precision specific attributes
        self.surfaceFile = 'DxLandmarkGeo.xml'
        self.location = location
        self.ablationSites = []

    def import_map(
            self,
    ) -> None:
        """
        Load all relevant information for this mapping procedure, import EGM
        recording points, interpolate standard surface parameter maps from
        point data (bip voltage, uni voltage, LAT), and build representative
        body surface ECGs.

        Raises:
            MeshFileNotFoundError: If mesh file is not found in repository

        Returns:
            None
        """

        self.surface = self.load_mesh()
        self.points = self.load_points()
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
        Load a Precision mesh from file.

        Raises:
            MeshFileNotFoundError : if mesh file is not found

        Returns:
            Surface
        """

        log.info('reading Precision mesh {}'.format(self.surfaceFile))

        mesh_file = self.parent.repository.join(
            self.location + '/' + self.surfaceFile
        )
        if not self.parent.repository.is_file(mesh_file):
            raise MeshFileNotFoundError(filename=self.surfaceFile)

        with self.parent.repository.open(mesh_file, mode='rb') as fid:
            return read_landmark_geo(fid, encoding=self.parent.encoding)

    def load_points(
            self
    ) -> List[PrecisionPoint]:
        """
        Load points for Precision map.

        Point information is found in "DxL_#.csv" files.

        Returns:
            list of PrecisionPoint
        """

        log.info('import EGM points')

        points = []

        # get all DxL files in root folder
        dxl_regex = re.compile('DxL.*csv')
        dxl_files = self.parent.repository.list_dir(
            self.parent.repository.join(self.location),
            regex=dxl_regex
        )
        dxl_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # work through files and get data
        point_id = 1
        for n, filename in enumerate(dxl_files):
            # update progress bar
            console_progressbar(
                n + 1, len(dxl_files),
                suffix='Loading point data from file {}'.format(filename)
            )

            # load file data
            file = self.parent.repository.join(self.location + '/' + filename)
            with self.parent.repository.open(file) as fid:
                header, point_data, ecg_data, cfe_data = load_dxl_data(
                    fid,
                    encoding=self.parent.encoding
                )

            # check if files are in correct order
            if not n == header.fileNumber[0] - 1:
                log.warning('files are not read in consecutive order! '
                            'Reading file {}, expected file {}!'
                            .format(header.fileNumber[0], n)
                            )
            # check if all files are listed
            if not header.fileNumber[1] == len(dxl_files):
                log.warning('not all data files are listed! Expected {} '
                            'files, but only {} are listed'
                            .format(header.fileNumber[1], len(dxl_files))
                            )

            # build PrecisionPoints
            for i in range(header.nPoints):
                # get coordinates first
                coordinates = np.array([float(point_data['roving x'][i]),
                                        float(point_data['roving y'][i]),
                                        float(point_data['roving z'][i])
                                        ]
                                       )
                surf_coordinates = np.array([float(point_data['surfPt x'][i]),
                                             float(point_data['surfPt y'][i]),
                                             float(point_data['surfPt z'][i])
                                             ]
                                            )

                # initialize point
                point = PrecisionPoint(
                    'P{}'.format(point_id),
                    coordinates=coordinates,
                    parent=self
                )

                # calculate annotation time in (samples) relative to trace
                # window
                exported_secs = header.exportedSeconds
                end_time = float(point_data['end time'][i])
                ref_lat = float(point_data['ref LAT'][i])
                egm_lat = float(point_data['rov LAT'][i])
                ref_annotation = exported_secs - (end_time - ref_lat)
                lat_annotation = exported_secs - (end_time - egm_lat)

                # add base attributes
                point.prjX = surf_coordinates
                point.refAnnotation = ref_annotation * header.sampleRate
                point.latAnnotation = lat_annotation * header.sampleRate
                point.bipVoltage = float(point_data['peak2peak'][i])
                point.egmBip = Trace(name=ecg_data['rov']['names'][i],
                                     data=ecg_data['rov']['values'][:, i],
                                     fs=header.sampleRate
                                     )
                # TODO: get unipolar recordings for Precision
                uni_names = ecg_data['rov']['names'][i].split()[-1].split('-')
                point.egmUni = [
                    Trace(name=uni_names[0],
                          data=np.full(point.egmBip.data.shape, np.nan),
                          fs=header.sampleRate),
                    Trace(name=uni_names[1],
                          data=np.full(point.egmBip.data.shape, np.nan),
                          fs=header.sampleRate)
                ]
                point.egmRef = Trace(name=ecg_data['ref']['names'][i],
                                     data=ecg_data['ref']['values'][:, i],
                                     fs=header.sampleRate
                                     )

                # add Precision-specific attributes
                point.utilized = bool(int(point_data['utilized'][i]))
                point.displayed = bool(int(point_data['displayed'][i]))
                point.exportedSeconds = exported_secs
                point.endTime = end_time
                point.pNegativeVoltage = float(point_data['peak neg'][i])
                point.CFEMean = float(point_data['CFE mean'][i])
                point.CFEstd = float(point_data['CFE stddev'][i])
                point.CFEDetection = cfe_data[i]
                point.algorithm = {
                    'rov': DetectionAlgorithm(
                        code=point_data['rov detect'][i],
                        parameter=float(point_data['rov param'][i])),
                    'ref': DetectionAlgorithm(
                        code=point_data['ref detect'][i],
                        parameter=float(point_data['ref param'][i])),
                }

                # add point to map
                points.append(point)
                point_id += 1

        log.info('loaded {} points to map {}'
                 .format(len(points), self.name)
                 )

        return points

    def import_lesions(
            self,
            *args, **kwargs
    ) -> None:
        """
        Import Precision lesion data.

        Note: More than one RF index can be stored per ablation site.

        Returns:
            None
        """

        log.info('loading lesion data for map {}'.format(self.name))

        lesion_file = self.parent.repository.join(
            self.location + '/' + 'Lesions.csv'
        )
        if not self.parent.repository.is_file(lesion_file):
            log.warning('no lesion data found ({})'.format(lesion_file))
            return

        with self.parent.repository.open(lesion_file) as fid:
            self.ablationSites = load_lesion_data(fid,
                                                  encoding=self.parent.encoding
                                                  )

        # convert ablation sites data to base class lesions
        self.lesions = self.ablation_sites_to_lesion(self.ablationSites)

    def build_map_ecg(
            self,
            ecg_names: Optional[Union[str, List[str]]] = None,
            method: Optional[Union[str, List[str]]] = None,
            *args, **kwargs
    ) -> List[BodySurfaceECG]:
        """
        Get a mean surface ECG trace.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Parameters:
            method : str, list of str (optional)
                not used here, just for compatibility. Precision ECGs are
                saved per map anyway.
            ecg_names : list of str
                ECG names to build. If note specified, 12-lead ECG is used

        Returns
            list of BodySurfaceECG : representative ECG traces
        """

        log.info('loading ECGs for map {}'.format(self.name))

        ecg_file = self.parent.repository.join(
            self.location + '/' + 'ECG_RAW.csv')
        if not self.parent.repository.is_file(ecg_file):
            log.warning('no ECG data found ({})'.format(ecg_file))
            return []

        if not ecg_names:
            ecg_names = ['I', 'II', 'III',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                         'aVL', 'aVR', 'aVF'
                         ]
        elif isinstance(ecg_names, str):
            ecg_names = [ecg_names]

        with self.parent.repository.open(ecg_file, mode='rb') as fid:
            traces = load_ecg_data(fid, encoding=self.parent.encoding)

        # check if requested ECG signals were loaded
        trace_names = [t.name for t in traces]
        load_names = []
        for name in ecg_names:
            if name not in trace_names:
                log.warning('ECG {} not found in data!'.format(name))
            else:
                load_names.append(name)

        return [BodySurfaceECG(method='recording',
                               refAnnotation=np.nan,
                               traces=[t for t in traces
                                       if t.name in load_names
                                       ]
                               )
                ]

    def export_point_info(
            self,
            output_folder: str = '',
            points: Optional[List[EPPoint]] = None
    ) -> None:
        """
        Export additional recording point info in DAT format.

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".

        Following data can is exported:
            NAME : point identifier
            REF : reference annotation

        By default, data from all valid points is exported, but also a
        list of EPPoints to use can be given.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.ptdata.<parameter>.pc.dat

        Parameters:
            output_folder : str (optional)
                path of the exported files
           points : list of PrecisionPoints (optional)
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

        return

    def export_point_ecg(
            self,
            output_folder: str = '',
            which: Optional[Union[str, List[str]]] = None,
            points: Optional[List[EPPoint]] = None,
            reload_data: bool = False
    ) -> None:
        """
        Export surface ECG traces in IGB format.

        Not implemented yet!

        Parameters:
            output_folder : str (optional)
                path of the exported files
            which : string or list of strings
                ECG name(s) to include in IGB file.
            points : list of PrecisionPoints (optional)
                EGM points to export
            reload_data : bool
                reload ECG data if already loaded

        Returns:
            None
        """

        log.info('cannot export point ECG data for Precision studies!')
        return

    @staticmethod
    def ablation_sites_to_lesion(
            sites
    ) -> Lesions:
        """
        Convert ablation sites data to base class lesions.

        Parameters:
            sites : list of PrecisionLesion

        Returns:
            list of AblationSites
        """

        lesions = []
        for site in sites:
            # Precision lesions only have color information, convert to
            # numeric value as RFI
            rfi_value = (site.color[0]
                         + site.color[1] * 256
                         + site.color[2] * 256**2
                         )
            rfi = RFIndex(name='precision', value=rfi_value)
            lesions.append(AblationSite(X=site.X,
                                        diameter=site.diameter,
                                        RFIndex=[rfi]
                                        )
                           )

        return Lesions(lesions)


class PrecisionStudy(EPStudy):
    """
    Class representing a Precision study.

    Attributes:
        system : str
            name of the EAM system used
        repository : Repository
            path object pointing to study data repository. Can be a folder or
            a ZIP archive
        pwd : bytes
            password to access encrypted ZIP archives
        encoding : str
            file encoding used to read files. (default: cp1252)
        name : str
            name of the study
        mapNames : list of str
            names of the mapping procedures contained in data set
        mapPoints : list of int
            number of points recorded during mapping procedure
        maps : dict
            mapping procedures performed during study. Dictionary keys are
            the mapping procedure names (subset of mapNames attribute)
        version : str
            file version used in repository
        mapLocations : list of str
            path(s) to map data within the repository
        meshes : list of Surface objects (optional)
            additional meshes from e.g. CT data
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
            pwd : str (optional)
                password for protected ZIP archives
            encoding : str (optional)
                file encoding used (all files are in binary mode).
                Default: cp1252

        Raises:
            TypeError : if system name is unknown (or not yet implemented)
            TypeError : if study_repo is not of type string
            FileExistsError : if study file or folder does not exist

        Returns:
            None
        """

        super().__init__(system='precision',
                         study_repo=study_repo,
                         pwd=pwd,
                         encoding=encoding)

        self.version = '0.0'  # system version creating the data
        self.mapLocations = []  # location of map data within repository

    def import_study(
            self
    ) -> None:
        """
        Load study details and basic information.

        Returns:
            None
        """

        # evaluate study name
        study_info = self.get_study_info()

        self.name = study_info['name']
        self.version = study_info['version']
        self.mapNames = [m[0] for m in study_info['maps']]
        self.mapLocations = [m[1] for m in study_info['maps']]
        # number of points is undetermined for now...
        self.mapPoints = [np.nan] * len(self.mapNames)

    def get_study_info(
            self
    ) -> Optional[dict]:
        """
        Load basic info about study from repository.

        Returns:
            dict
                'name' : study name
                'version' : version with which this was created
                'maps' : list of tuple
                    names and path of mapping procedures
        """

        log.debug('searching for data in {}'.format(self.repository))

        def check_folder(
                path: Repository,
                structure: List[tuple[str, str, str, str]]
        ) -> Optional[List[tuple[str, str, str, str]]]:
            """
            Search this folder tree for Precision data.

            Returns:
                List of tuple or empty list
                    name : str
                    map_name : str
                    version : str
                    loc : str (data path relative to repository root)
            """

            file_matches = path.list_dir(path.join(''),
                                         regex=r'Model(.*)Groups.xml'
                                         )
            if file_matches:
                map_name = os.path.basename(path.get_root_string())
                map_loc = ''
                version = '0.0'
                study_name = ''

                # get version info from timeline .csv
                csv_file = path.list_dir(path.join(''),
                                         regex=r'NotebookByTime.csv'
                                         )
                if not csv_file:
                    log.warning('unable to find NotebookByTime.csv, cannot '
                                'retrieve study info!')
                else:
                    with path.open(path.join(csv_file[0])) as fid:
                        # read first 10 lines, should contain all info needed
                        header = [next(fid).decode(encoding=self.encoding).rstrip()
                                  for _ in range(10)]
                        for line in header:
                            if 'File Revision' in line:
                                version = line.split(':')[1].strip()
                            if 'Export from Study' in line:
                                study_name = line.split(':')[1].strip()
                    map_loc = os.path.relpath(path.get_root_string(),
                                              path.get_base_string()
                                              )

                return [
                    (study_name, map_name, version, map_loc)
                ]

            # no matches found, continue search
            folders = [f for f in path.list_dir(path.join(''))
                       if path.is_folder(path.join(f))
                       or path.is_archive(path.join(f))
                       ]
            log.debug('found subdirectories: {}'.format(folders))

            for folder in folders:
                # update root location and start new search there
                temp_repo = copy.copy(path)
                temp_repo.update_root(path.join(folder))
                structure += check_folder(temp_repo, structure)

            # no data found
            return []

        # search directory tree
        study_structure = []
        study_structure += check_folder(self.repository, study_structure)

        if not study_structure:
            return None

        # build study info from tree structure
        # check if all maps are from same study
        if not all([x[0] == study_structure[0][0] for x in study_structure]):
            raise TypeError('data comes from different studies!')
        # check if file version is same for all maps
        if not all([x[2] == study_structure[0][2] for x in study_structure]):
            raise TypeError('data contains different file formats!')
        # build map names and location
        map_info = []
        for pmap in study_structure:
            name = pmap[1]
            loc = pmap[3]
            map_info.append((name, loc))

        study_info = dict()
        study_info['name'] = study_structure[0][0]
        study_info['version'] = study_structure[0][2]
        study_info['maps'] = map_info

        # no data was found
        return study_info

    def import_maps(
            self,
            map_names: Optional[Union[str, List[str]]] = None,
            *args, **kwargs
    ) -> None:
        """
        Import Precision maps.

        If a map was already imported before and is part of the study,
        user interaction to reload is required.

        Parameters:
            map_names : list of str (optional)
                name or list of map names to import. If no name is
                specified, all maps are loaded (default).

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            None
        """

        # do some pre-import checks
        map_names = super().import_maps()

        # now import maps
        for i, map_name in enumerate(map_names):
            try:
                map_location = self.mapLocations[i]
                log.info('importing map {} from {}:'
                         .format(map_name, map_location))
                new_map = PrecisionMap(map_name, map_location, parent=self)
                new_map.import_map()
                self.maps[map_name] = new_map
                self.mapPoints[i] = len(new_map.points)
            except Exception as err:
                log.warning('failed to import map {}: {}'
                            .format(map_name, err))
                continue

        return

    def export_additional_meshes(
            self,
            output_folder: str = '',
            *args, **kwargs
    ) -> None:
        """Export any additional meshes."""

        log.warning('export of additional meshes is not yet supported for '
                    'Precision!')
        return

    def is_root_valid(
            self,
            root_dir: str = '',
            pwd: str = ''
    ) -> bool:
        """
        Check if study root is valid.

        Parameters:
            root_dir : string (optional)
                path to check. If not specified, the current study root
                is checked.
            pwd : str
                password used for protected archives

        Returns:
            bool : valid or not
        """

        log.info('checking if study root{}is valid'
                 .format(' ' + root_dir + ' ' if root_dir else ' '))

        if not root_dir:
            for folder in self.mapLocations:
                path = self.repository.join(folder)
                if not self.repository.is_folder(path):
                    log.warning('cannot find {} in repository!'.format(folder))
                    # map location does not exist
                    return False
                # all map folders found, valid
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

            study_info = self.get_study_info()
            if study_info is None:
                log.warning('no data found in {}'.format(root_dir))
                return False

            # if study name is correct, assume valid repository
            if study_info['name'] == self.name:
                return True

            log.warning('study name in repository ({}) does not match!'
                        .format(study_info['name'])
                        )

        # at this point root is definitely invalid
        return False

    def set_repository(
            self,
            root_dir: str
    ) -> bool:
        """
        Change path to root directory.

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

        # set repository root to new location
        self.repository.update_root(root_dir)
        return True

    @classmethod
    def load(
            cls,
            file: str,
            repository_path: str = '',
            password: str = ''
    ) -> TPrecisionStudy:
        """
        Load study from file. Overrides BaseClass method.

        A previously saved version of a PrecisionStudy object can be
        loaded. The objects <study_root> is set to the one stored in the
        file if valid. If not, the folder of the file is set as root
        directory.
        The path to the Precision files can also be specified explicitly.

        Parameters:
            file : str
                location of .pyceps file
            repository_path : str
                set repository root to this location
            password : str

        Raises:
            TypeError : If file is not Precision

        Returns:
            PrecisionStudy
        """

        log.debug('loading study')

        with open(file) as fid:
            root = ET.parse(fid).getroot()

        # check if file was generated from Carto3 data
        system = root.get('system')
        if not system.lower() == "precision":
            raise TypeError('expected Precision system file, found {}'
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
        study.version = root.get('file_version')

        # try to set study root
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
        sep = chr(int(proc_item.get('sep')))
        study.mapNames = proc_item.get('names').split(sep)
        study.mapPoints = [int(x) if x != 'nan' else np.iinfo(int).min
                           for x in proc_item.get('points').split(sep)
                           ]

        for proc in proc_item.iter('Procedure'):
            name = proc.get('name')
            location = proc.get('location')
            # add location to study
            study.mapLocations.append(location)

            new_map = PrecisionMap(name, location, parent=study)

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
            if lesions_item:
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
                    new_point = PrecisionPoint('dummy', parent=new_map)
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

        if keep_ecg:
            log.warning('saving point ECGs is not supported for Precision!')
            keep_ecg = False

        # add basic information to XML
        root, filepath = super().save(filepath, keep_ecg=keep_ecg)
        if not root:
            # no base info was created (no maps imported), nothing to add
            return filepath

        # add Precision specific data
        root.set('file_version', self.version)

        for key, cmap in self.maps.items():
            map_item = [p for p in root.iter('Procedure')
                        if p.get('name') == key][0]

            # add additional procedure info
            map_item.set('location', cmap.location)

            # add additional point info
            point_item = map_item.find('Points')
            to_add = ['displayed', 'utilized', 'exportedSeconds', 'endTime',
                      'pNegativeVoltage', 'CFEMean', 'CFEstd'
                      ]
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
