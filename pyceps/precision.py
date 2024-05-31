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
import pickle
import gzip
import numpy as np
import re

from pyceps.study import EPStudy, EPMap, EPPoint
from pyceps.datatypes.precision.precisiontypes import DetectionAlgorithm
from pyceps.datatypes.signals import Trace, BodySurfaceECG
from pyceps.datatypes.lesions import RFIndex
from pyceps.fileio.precisionio import (read_landmark_geo, load_dxl_data,
                                       load_ecg_data, load_lesion_data
                                       )
from pyceps.datatypes.exceptions import MeshFileNotFoundError
from pyceps.utils import console_progressbar


log = logging.getLogger(__name__)


class PrecisionStudy(EPStudy):
    """
    Class representing a Precision study.
    """

    def __init__(self, study_repo, pwd='', encoding='cp1252'):
        """Constructor."""

        super().__init__(system='precision',
                         study_repo=study_repo,
                         pwd=pwd,
                         encoding=encoding)

        if not os.path.isdir(study_repo):
            log.warning('Study folder not found!')
            raise FileNotFoundError

        # Precision study name is the name of export folder
        self.name = os.path.basename(study_repo)
        self.studyRoot = study_repo

        self.import_study()

    def import_study(self):
        """Load study details and basic information."""

        # get map names, i.e. sub-folder names of study root
        self.mapNames = self._get_immediate_subdir(self.studyRoot)
        # number of points is undetermined for now...
        self.mapPoints = [np.nan] * len(self.mapNames)

    def import_maps(self, map_names=None, *args, **kwargs):
        """
        Load a Precision map

        Returns:
            None
        """

        # do some pre-import checks
        map_names = super().import_maps()

        # now import maps
        for map_name in map_names:
            try:
                new_map = PrecisionMap(map_name, parent=self)
                self.maps[map_name] = new_map
            except Exception as err:
                log.warning('failed to import map {}: {}'
                            .format(map_name, err))
                continue

        return

    def export_additional_meshes(self, *args, **kwargs):
        raise NotImplementedError

    def is_root_valid(self, root_dir=None):
        """
        Check if study root is valid.

        Parameters:
            root_dir : string (optional)
                path to check. If not specified, the current study root
                is checked.

        Returns:
            bool : valid or not
        """

        if not root_dir:
            folder_list = self._get_immediate_subdir(self.studyRoot)
        else:
            folder_list = self._get_immediate_subdir(root_dir)

        # check if study root contains folders with same name as maps
        if not all(n in folder_list for n in self.mapNames):
            # root saved in study is invalid
            return False

        # specified root directory is valid
        return True

    def set_repository(self, root_dir):
        """
        Change path to root directory.

        If new root directory is invalid, it is not changed.

        Parameters:
            root_dir : string
                new root directory

        Returns:
            bool : successful or not
        """

        study_root = os.path.abspath(root_dir)
        if not self.is_root_valid(study_root):
            log.warning('root directory is invalid: {}'.format(study_root))
            return False

        # set proper study root, i.e string or zipfile.Path
        self.studyRoot = root_dir
        return True

    @classmethod
    def load(cls, file: str, repository_path: str = '', password: str = ''):
        """
        Load pickled version of a study.

        A previously saved pickled version of a CartoStudy object can be
        loaded. The objects <study_root> is set to the one stored in the
        PKL file if valid. If not, the folder of the PKL is set as root
        directory.
        The path to the Carto files can also be specified explicitly.

        Note that loading to a string with pickle.loads is about 10% faster
        but probably consumes a lot more memory so we'll skip that for now.

        Parameters:
            filename : string
                path to the .PKL or .GZ study file
            root : string (optional)
                path to the root directory

        Raises:
            FileNotFoundError : if pickled file cannot be found

        Returns:
            CartoStudy object
        """

        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()

        # try to set root if explicitly given
        if root:
            if obj.set_repository(os.path.abspath(root)):
                log.info('setting study root to {}'.format(root))
                return obj
            else:
                log.info('cannot set study root to {}\n'
                         'Trying to use root information from PKL'
                         .format(root))

        # check if repository is valid and accessible
        if obj.is_root_valid():
            log.info('previous study root is still valid ({})'
                     .format(obj.studyRoot))
            obj.set_repository(os.path.abspath(obj.studyRoot))
            return obj

        # no valid root found so far, set to pkl directory
        log.warning('no valid study root found. Using .pkl location!'.upper())
        obj.studyRoot = os.path.abspath(filename)

        return obj

    @staticmethod
    def _get_immediate_subdir(parent_dir):
        """Get all directory names in parent folder."""

        return [name for name in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, name))
                ]


class PrecisionMap(EPMap):
    """
    Class representing Precision map.
    """

    def __init__(self, name, parent=None):
        """Constructor."""

        # Note: map name is folder name for now, needs to be extracted from
        # DxL data files and set later (load_points())!
        super().__init__(name, parent=parent)

        self.surfaceFile = 'DxLandmarkGeo.xml'

        # add Precision specific attributes
        self.rootDir = os.path.join(self.parent.studyRoot, name)
        self.files = [x for x in os.listdir(self.rootDir)
                      if os.path.isfile(os.path.join(self.rootDir, x))]
        self.ablationSites = []

        # load data
        self.surface = self.import_mesh()
        self.points = self.load_points()
        # build surface maps
        self.interpolate_data('act')
        self.interpolate_data('bip')
        # self.interpolate_data('uni')
        self.ecg = self.build_map_ecg()

    def import_mesh(self, *args, **kwargs):
        """
        Load a Precision mesh from file.

        Raises:
            MeshFileNotFoundError

        Returns:
            Surface object
        """

        mesh_file = os.path.join(self.rootDir, self.surfaceFile)
        log.info('reading Precision mesh {}'.format(mesh_file))

        if not os.path.isfile(mesh_file):
            raise MeshFileNotFoundError(filename=mesh_file)

        return read_landmark_geo(mesh_file)

    def load_points(self, *args, **kwargs):
        """
        Load points for Carto3 map.

        Point information is found in "DxL_#.csv" files.

        Returns:
            points : list of PrecisionPoint objects

        """

        log.info('import EGM points')

        points = []

        # get all DxL files in root folder
        dxl_regex = re.compile('DxL.*csv')
        dxl_files = [os.path.join(self.rootDir, f)
                     for f in self.files if re.match(dxl_regex, f)]
        dxl_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # work through files and get data
        point_id = 1
        for n, file in enumerate(dxl_files):
            # update progress bar
            console_progressbar(
                n + 1, len(dxl_files),
                suffix='Loading point data from file {}'.format(file)
            )

            # load file data
            header, point_data, ecg_data, cfe_data = load_dxl_data(file)

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

    def import_lesions(self, *args, **kwargs):
        """
        Import Precision lesion data.

        Note: More than one RF index can be stored per ablation site.

        Returns:
            None
        """

        log.info('loading lesion data for map {}'.format(self.name))

        lesion_file = 'Lesions.csv'
        if lesion_file not in self.files:
            log.warning('no lesion data found ({})'.format(lesion_file))
            return

        self.ablationSites = load_lesion_data(os.path.join(self.rootDir,
                                                           lesion_file)
                                              )

        # convert ablation sites data to base class lesions
        self.lesions = self.ablation_sites_to_lesion()

    def build_map_ecg(self, ecg_names=None, method=None, *args, **kwargs):
        """Get a mean surface ECG trace.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Parameters:
            method:
            ecg_names : list of str
                ECG names to build. If note specified, 12-lead ECG is used

        Returns
            list of BodySurfaceECG : representative ECG traces
        """

        log.info('loading ECGs for map {}'.format(self.name))

        ecg_file = 'ECG_RAW.csv'
        if ecg_file not in self.files:
            log.warning('no ECG data found ({})'.format(ecg_file))
            return []

        if not ecg_names:
            ecg_names = ['I', 'II', 'III',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                         'aVL', 'aVR', 'aVF'
                         ]
        elif isinstance(ecg_names, str):
            ecg_names = [ecg_names]

        traces = load_ecg_data(os.path.join(self.rootDir, ecg_file))

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

    def ablation_sites_to_lesion(self):
        """Convert ablation sites data to base class lesions."""

        lesions = []
        for site in self.ablationSites:
            # Precision lesions only have color information, convert to
            # numeric value as RFI
            rfi_value = (site.color[0]
                         + site.color[1] * 256
                         + site.color[2] * 256**2
                         )
            rfi = RFIndex(name='precision', value=rfi_value)
            lesions.append(Lesion(X=site.X,
                                  diameter=site.diameter,
                                  RFIndex=[rfi]
                                  )
                           )

        return lesions


class PrecisionPoint(EPPoint):
    """
    Class representing a Precision mapping point.
    """

    def __init__(self, name,
                 coordinates=np.full((3, 1), np.nan, dtype=float),
                 parent=None):
        """Constructor."""

        super().__init__(name, coordinates=coordinates, parent=parent)

        # add Carto3 specific attributes
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

    def is_valid(self):
        return self.utilized
