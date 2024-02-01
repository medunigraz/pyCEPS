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
import zipfile
import re
import xml.etree.ElementTree as xml
import numpy as np
import gzip
import pickle

from src.fileio.pathtools import Repository
from src.datatypes import EPStudy, EPMap, EPPoint, Mesh
from src.fileio import FileWriter
from src.fileio.cartoutils import (# open_carto_file, join_carto_path,
                                       # list_carto_dir, carto_isfile,
    # carto_isdir,
                                       read_mesh_file,
                                       read_ecg_file_header, read_ecg_file,
                                       channel_names_from_ecg_header,
                                       channel_names_from_pos_file,
                                       read_force_file,
                                       read_visitag_file)
from src.datatypes.cartotypes import (CartoUnits, Coloring, ColoringRange,
                                      SurfaceErrorTable,
                                      PasoTable, CFAEColoringTable, Tag,
                                      CartoMappingParameters,
                                      RefAnnotationConfig, PointImpedance,
                                      RFAblationParameters, RFForce, MapRF,
                                      Visitag, VisitagAblationSite,
                                      VisitagGridPoint, VisitagAblationGrid,
                                      VisitagRFIndex
                                      )
from src.datatypes.lesions import Lesion, RFIndex
from src.datatypes.signals import Trace, BodySurfaceECG
from src.exceptions import MapAttributeError, MeshFileNotFoundError
from src.utils import console_progressbar, get_col_idx_from_header


log = logging.getLogger(__name__)


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
        visitag : VisiTag
            ablation sites and ablation grid information
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

    def __init__(self, study_repo, pwd='', encoding='cp1252'):
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

        # locate study XML
        log.info('Locating study XML in {}...'.format(self.repository))
        study_info = self._locate_study_xml(self.repository,
                                            pwd=self.pwd,
                                            encoding=self.encoding)
        if not study_info:
            raise FileNotFoundError

        log.info('found study XML at {}'.format(self.repository.root))
        self.studyXML = study_info['xml']
        self.name = study_info['name']

        self.units = None
        self.environment = None  # TODO: is this relevant info?
        self.externalObjects = None  # TODO: is this relevant info?
        self.mappingParams = None

        # visitag data
        self.visitag = Visitag()

        self.import_study()

    def import_study(self):
        """
        Load study details and basic information from study XML.
        Overrides BaseClass method.
        """

        log.info('accessing study XML: {}'.format(self.studyXML))
        log.info('gathering study information...')

        xml_path = self.repository.join(self.studyXML)
        with self.repository.open(xml_path) as fid:
            root = xml.parse(fid).getroot()

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
            BadErrorColor=[int(x) for x in item.get('BadErrorColor').split()],
            MedErrorColor=[int(x) for x in item.get('MedErrorColor').split()],
            GoodErrorColor=[int(x) for x in item.get('GoodErrorColor').split()],
            BadErrorThreshold=int(item.get('BadErrorThreshold')),
            MedErrorThreshold=int(item.get('MedErrorThreshold')),
            GoodErrorThreshold=int(item.get('GoodErrorThreshold')))

        log.debug('reading study paso table')
        item = study_parameters.find('PasoTable')
        paso_table = PasoTable(ISName=item.get('ISName'))

        log.debug('reading study CFAE coloring table')
        item = study_parameters.find('CFAEColoringTable')
        cfae_coloring_table = CFAEColoringTable(
            IgnoreBelowColor=[float(x) for x in item.get('IgnoreBelowColor').split()],
            IclMediumColor=[float(x) for x in item.get('IclMediumColor').split()],
            IclHighColor=[float(x) for x in item.get('IclHighColor').split()],
            IgnoreBelowThreshold=item.get('IgnoreBelowThreshold'),
            IclMediumThreshold=item.get('IclMediumThreshold'),
            IclHighThreshold=item.get('IclHighThreshold'))

        log.debug('reading study tags table')
        item = study_parameters.find('TagsTable')
        tags_table = []
        for tag in item.findall('Tag'):
            tags_table.append(
                Tag(ID=tag.get('ID'),
                    Short_Name=tag.get('Short_Name'),
                    Full_Name=tag.get('Full_Name'),
                    Color=[float(x) for x in tag.get('Color').split()],
                    Radius=int(tag.get('Radius'))
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
                                dtype=float)
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
            map_points.append(item.find('CartoPoints').get('Count'))
            log.debug('found map {} with {} mapping points'
                      .format(map_names[-1], map_points[-1]))

        self.mapNames = map_names
        self.mapPoints = map_points

    def import_maps(self, map_names=None, egm_names_from_pos=False,
                    *args, **kwargs):
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
            map_names : string or list (optional)
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
                new_map = CartoMap(map_name, self.studyXML,
                                   parent=self,
                                   egm_names_from_pos=egm_names_from_pos)
                self.maps[map_name] = new_map
            except Exception as err:
                log.warning('failed to import map {}: {}'
                            .format(map_name, err))
                continue

        return

    def import_visitag_sites(self, directory=None):
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
            list of VisitagAblationSites objects

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
                return self.visitag.sites
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
                log.info('no ablation sites found in Sites.txt, trying QMODE+...')
            else:
                for site in sites_data:
                    sites.append(
                        VisitagAblationSite(
                            int(site[sites_hdr.index('SiteIndex')]),
                            session_index=int(site[sites_hdr.index('Session')]),
                            channel_id=int(site[sites_hdr.index('ChannelID')]),
                            tag_index_status=int(site[sites_hdr.index('TagIndexStatus')]),
                            coordinates=[site[sites_hdr.index('X')],
                                         site[sites_hdr.index('Y')],
                                         site[sites_hdr.index('Z')]],
                            avg_force=site[sites_hdr.index('AverageForce')],
                            fti=site[sites_hdr.index('FTI')],
                            max_power=site[sites_hdr.index('MaxPower')],
                            max_temp=site[sites_hdr.index('MaxTemperature')],
                            duration=site[sites_hdr.index('DurationTime')],
                            base_impedance=site[sites_hdr.index('BaseImpedance')],
                            impedance_drop=site[sites_hdr.index('ImpedanceDrop')],
                            rf_index=VisitagRFIndex(
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
                            coordinates=[site[q_sites_hdr.index('X')],
                                         site[q_sites_hdr.index('Y')],
                                         site[q_sites_hdr.index('Z')]],
                            avg_force=site[q_sites_hdr.index('AverageForce')],
                            fti=site[q_sites_hdr.index('FTI')],
                            max_power=site[q_sites_hdr.index('MaxPower')],
                            max_temp=site[q_sites_hdr.index('MaxTemperature')],
                            duration=site[q_sites_hdr.index('DurationTime')],
                            base_impedance=site[q_sites_hdr.index('BaseImpedance')],
                            impedance_drop=site[q_sites_hdr.index('ImpedanceDrop')],
                            rf_index=VisitagRFIndex(
                                name='VisitagFTI',
                                value=site[q_sites_hdr.index('FTI')]
                            )
                        )
                    )

        # check if any data was loaded
        if not len(sites) > 0:
            log.warning('no visitag data found in files! Aborting...')

        self.visitag.sites = sites

    def load_visitag_grid(self, directory=None):
        """
        Load VisiTag ablation grid data.

        If sites are already loaded and part of the study, user interaction
        is required.

        Parameters:
            directory : string (optional)
                path to folder containing VisiTag data
                (default is <study_root>/VisiTagExport)

        Raises:
            ValueError : If user entered character other than Y/N/y/n

        Returns:
            list of VisitagAblationGrid objects.
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
        session = abl_site_data[:, cols].astype(int).ravel()
        cols = get_col_idx_from_header(abl_site_hdr, 'FirstPosTimeStamp')
        first_pos_tstamp = abl_site_data[:, cols].astype(int).ravel()
        cols = get_col_idx_from_header(abl_site_hdr, 'FirstPosPassedFilterTimeStamp')
        first_pos_passed_tstamp = abl_site_data[:, cols].astype(int).ravel()
        cols = get_col_idx_from_header(abl_site_hdr, 'LastPosTimeStamp')
        last_pos_tstamp = abl_site_data[:, cols].astype(int).ravel()

        log.info('found {} ablation sites'.format(n_sites))

        # load grid data
        log.info('load grid data. This might take a while...')
        file = self.repository.join(visi_dir + '/' + 'PositionsData.txt')
        with self.repository.open(file, mode='rb') as fid:
            pos_data, pos_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )
        file = self.repository.join(visi_dir + '/' + 'ContactForceData.txt')
        with self.repository.open(file, mode='rb') as fid:
            force_data, force_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )
        file = self.repository.join(visi_dir + '/' + 'AllPositionInGrids.txt')
        with self.repository.open(file, mode='rb') as fid:
            grid_pos_data, grid_pos_hdr = read_visitag_file(
                fid, encoding=self.encoding
            )
        file = self.repository.join(visi_dir + '/' + 'Grids.txt')
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
        for i, site in enumerate(site_index):
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
                uid_X = grid_data[g_rows, cols]

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
                                     rf_index=VisitagRFIndex(
                                         name='CartoAblationIndex',
                                         value=uid_rf_index),
                                     passed=uid_passed
                                     )
                )

            # add all grid points to visitag grid
            grid.add_points(grid_points)
            # add this grid to list of grids
            grid_sites.append(grid)

        return grid_sites

    @classmethod
    def load(cls, filename, root=None):
        """
        Load pickled version of a study. Overrides BaseClass method.

        A previously saved pickled version of a CartoStudy object can be
        loaded. The objects <study_root> is set to the one stored in the
        PKL file if valid. If not, the folder of the PKL is set as root
        directory.
        The path to the Carto files can also be specified explicitly.

        Note that loading to a string with pickle.loads() is about 10% faster
        but probably consumes a lot more memory, so we'll skip that for now.

        Parameters:
            filename : string
                path to the .PKL or .GZ study file
            root : string (optional)
                set study root to this directory

        Raises:
            FileNotFoundError : if pickled file cannot be found

        Returns:
            CartoStudy

        """

        log.info('loading study from {}'.format(filename))

        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()

        # try to set root if explicitly given
        if root:
            if obj.set_root(os.path.abspath(root)):
                log.info('setting study root to {}'.format(root))
                return obj
            else:
                log.info('cannot set study root to {}\n'
                         'Trying to use root information from PKL'
                         .format(root))

        # try to re-set previous study root
        if obj.set_root(obj.repository.base):
            log.info('previous study root is still valid ({})'
                     .format(obj.repository.root))
            return obj

        # no valid root found so far, set to pkl directory
        log.warning('no valid study root found. Using .pkl location!'.upper())
        obj.repository.update_root(os.path.dirname(os.path.abspath(filename)))
        return obj

    def export_additional_meshes(self, filename=''):
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
            filename : str (optional)
                path to export file, export to default location if not given

        Returns:
            None
        """

        if not self.meshes:
            log.info('no additional meshes found in study, nothing to export')
            return

        if not filename:
            basename = self.build_export_basename('additionalMeshes')
        else:
            basename = os.path.abspath(filename)

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

    def rfi_from_visitag_grid(self):
        """
        Calculate RF index for VisiTag sites from VisiTag grid data.

        RFI values are added to the study's VisiTag sites data

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

    def is_root_valid(self, root_dir=None, pwd=''):
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
            tmp_root = Repository(root_dir, pwd=pwd)
            if not tmp_root.root:
                # dummy repo was not initialized properly, so root is invalid
                return False
            return self._locate_study_xml(tmp_root, pwd=pwd) is not None

        return False

    def set_root(self, root_dir):
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
        study_info = self._locate_study_xml(root,
                                            pwd=self.pwd,
                                            encoding=self.encoding)
        if not study_info:
            # should never happen...
            raise FileNotFoundError

        log.info('found study XML at {}'.format(self.repository.root))
        if not self.studyXML == study_info['xml']:
            log.warning('name of study XML differs, will not change root!')
            return False
        if not self.name == study_info['name']:
            log.warning('name of study differs, will not change root!')
            return False

        # change study root
        self.repository = root

        return True

    @staticmethod
    def _locate_study_xml(repository,
                          pwd='',
                          regex=r'^((?!Export).)*.xml$',
                          encoding='cp1252'):
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
            TypeError

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
            return CartoStudy._locate_study_xml(repository)

        # XML was nowhere to be found
        return None


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
        ecg : list of BodySurfaceECG
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

    def __init__(self, name, study_xml, parent=None, egm_names_from_pos=False):
        """
        Constructor.
        Load all relevant information for this mapping procedure, import EGM
        recording points, interpolate standard surface parameter maps from
        point data (bip voltage, uni voltage, LAT), and build representative
        body surface ECGs.

        Parameters:
            name : str
                name of the mapping procedure
            study_xml : str
                name of the study's XML file (same as in parent)
            parent : CartoStudy (optional)
                study this map belongs to
            egm_names_from_pos : boolean (optional)
                get names of egm traces from electrode positions

        Raises:
            MapAttributeError : If unable to retrieve map attributes from XML
            MeshFileNotFoundError: If mesh file is not found in repository

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

        self._import_attributes()
        self.surface = self.load_mesh()
        self.points = self.load_points(
            study_tags=self.parent.mappingParams.TagsTable,
            egm_names_from_pos=egm_names_from_pos)
        # build surface maps
        self.interpolate_data('lat')
        self.interpolate_data('bip')
        self.interpolate_data('uni')
        self.ecg = self.get_map_ecg(method=['median', 'mse', 'ccf'])

    def load_mesh(self):
        """
        Load a Carto3 triangulated anatomical shell from file. Overrides
        BaseClass method.

        Raises:
            MeshFileNotFoundError : if mesh file not found

        Returns:
            Surface object

        """

        log.info('reading Carto3 mesh {}'.format(self.surfaceFile))

        mesh_file = self.parent.repository.join(self.surfaceFile)
        if not self.parent.repository.is_file(mesh_file):
            raise MeshFileNotFoundError(filename=self.surfaceFile)

        with self.parent.repository.open(mesh_file, mode='rb') as fid:
            return read_mesh_file(fid, encoding=self.parent.encoding)

    def load_points(self, study_tags=None, egm_names_from_pos=False):
        """
        Load points for Carto3 map. Overrides BaseClass method.

        EGM names for recording points can be identified by evaluating the
        recording position to get the name of the electrode and comparing it
        to the name found in the points ECG file. Otherwise, the EGM name
        stored in a points ECG file is used.

        Parameters:
            study_tags : list of Tag objects
                to transform tag ID to label
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
            root = xml.parse(fid).getroot()

        map_item = [x for x in root.find('Maps').findall('Map')
                    if x.get('Name') == self.name]
        if not map_item:
            log.warning('no map with name {} found in study XML'
                        .format(self.name))
            return -1
        if len(map_item) > 1:
            log.warning('multiple maps with name {} found in study XML'
                        .format(self.name))
            return -1
        map_item = map_item[0]

        all_points_file = self.parent.repository.join(
            self.name + '_Points_Export.xml'
        )
        if not self.parent.repository.is_file(all_points_file):
            log.warning('unable to find export overview of all points {}'
                        .format(all_points_file))
            return -1

        with self.parent.repository.open(all_points_file, mode='rb') as fid:
            root = xml.parse(fid).getroot()

        if not root.get('Map_Name') == self.name:
            log.warning('map name {} in export file {} does not match map '
                        'name {} for import'
                        .format(root.get('Map_Name'),
                                self.name,
                                all_points_file)
                        )
            return -1
        point_files = {}
        for i, point in enumerate(root.findall('Point')):
            point_files[point.get('ID')] = point.get('File_Name')

        # TODO: read field "Anatomical_Tags"

        # get points in this map from study XML
        n_points = int(map_item.find('CartoPoints').get('Count'))
        if not len(point_files) == n_points:
            log.warning('number of points is not equal number of points files')
            return -1

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
                    tag_names = [x.Full_Name for x in study_tags
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
                                   point_file,
                                   coordinates=xyz,
                                   tags=tag_names,
                                   parent=self,
                                   egm_names_from_pos=egm_names_from_pos)
            points.append(new_point)

        return points

    def import_lesions(self, directory=None):
        """
        Import VisiTag lesion data.

        Note: More than one RF index can be stored per ablation site.

        Parameters:
            directory : str
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

        self.lesions = self.visitag_to_lesion(self.parent.visitag.sites)

    def get_map_ecg(self, ecg_names=None, method=None, *args, **kwargs):
        """Get a mean surface ECG trace.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        CARTO points are recorded sequentially. Therefore, ECG traces
        recorded at each point (i.e. at a time during procedure) vary. This
        function calculates a representative ECG.

        Building ECG traces with multiple method is most efficient when
        specifying the methods in a single call, since data has to be read
        only once.

        Parameters:
            ecg_names : list of str
                ECG names to build. If not specified, 12-lead ECG is used
            method : str, list of str (optional)
                Method to use. Options are ['median', 'ccf', 'mse']
                'median': time-wise median value of all ECGs
                'ccf': recorded ECG with highest cross-correlation to mean ecg
                'mse': recorded ECG with lowest MSE to mean ecg
                If not specified, all methods are used

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

        log.info('map {}: building representative ECGs: {}'
                 .format(self.name, ecg_names))

        points = self.get_valid_points()
        if not points:
            log.info('no points found in WOI or no points in map, aborting...')
            return []

        log.debug('found {} points in WOI'.format(len(points)))

        data = np.full((len(points), 2500, len(ecg_names)),
                       np.nan,
                       dtype=float)
        ref = points[0].refAnnotation
        woi = points[0].woi

        # get ECG traces for each mapping point
        for i, point in enumerate(points):
            # update progress bar
            console_progressbar(
                i+1, len(points),
                suffix='Loading ECG(s) for point {}'.format(point.name)
            )
            data[i, :, :] = point.import_ecg(channel_names=ecg_names)
            if not point.woi[0] == woi[0] or not point.woi[1] == woi[1]:
                log.warning('WOI changed in point {}'.format(point.name))
                # make this WOI the new one
                woi = point.woi
            if not point.refAnnotation == ref:
                log.warning('REF annotation changed in point {}'
                            .format(point.name))
                # make this the new ref
                ref = point.refAnnotation

        # build representative ecg trace
        if isinstance(method, str):
            method = [method]
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

    def export_point_ecg(self, basename='', which=None, points=None):
        """
        Export surface ECG traces in IGB format. Overrides BaseClass method.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".

        By default, EGMs for all valid points are exported, but also a
        list of EPPoints to use can be given.

        If no ECG names are specified, 12-lead ECGs are exported

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.ecg.<trace>.pc.igb

        Parameters:
            basename : string (optional)
                path and filename of the exported files
            which : string or list of strings
                ECG name(s) to include in IGB file.
            points : list of CartoPoints (optional)
                EGM points to export

        Returns:
            None
        """

        log.info('exporting surface ECG data')

        if not points:
            points = self.get_valid_points()
        if not len(points) > 0:
            log.warning('no points found in map {}. Nothing to export...'
                        .format(self.name))
            return

        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name)

        if not which:
            which = ['I', 'II', 'III',
                     'aVR', 'aVL', 'aVF',
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if isinstance(which, str):
            which = [which]

        # prepare data
        data = np.full((len(points), 2500, len(which)),
                       np.nan,
                       dtype=np.float32)

        # get data point-wise (fastest for multiple channels)
        for i, point in enumerate(points):
            console_progressbar(
                i + 1, len(points),
                suffix=('Loading ECG for Point {}'.format(point.name))
            )
            try:
                data[i, :, :] = point.import_ecg(channel_names=which)
            except KeyError as err:
                log.warning('{}'.format(err))
                continue

        # export point files
        self.export_point_cloud(points=points, basename=basename)

        # save data channel-wise
        writer = FileWriter()
        for i, channel in enumerate(which):
            channel_data = data[:, :, i]
            # save data to igb
            # Note: this file cannot be loaded with the CARTO mesh but rather
            #       with the exported mapped nodes
            header = {'x': channel_data.shape[0],
                      't': channel_data.shape[1],
                      'inc_t': 1}

            filename = '{}.ecg.{}.pc.igb'.format(basename, channel)
            f = writer.dump(filename, header, channel_data)
            log.info('exported ecg trace {} to {}'.format(channel, f))

        return

    def import_rf_data(self):
        """Load map associated RF data and RF contact forces."""

        log.info('loading RF and RF contact force data for map {}'
                 .format(self.name))

        # read RF data
        rf_abl = RFAblationParameters()
        rf_files = self.parent.repository.list_dir(
            '',
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

                    data = np.loadtxt(f, dtype=np.int32, skiprows=1)

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
                                              power=rf_data[:, 2],
                                              impedance=rf_data[:, 3],
                                              distal_temp=rf_data[:, 4])

        # read contact force in RF
        rf_force = RFForce()
        contact_force_rf_files = self.parent.repository.list_dir(
            '',
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

        # add extra column for indexing ablation position after data was read
        if contact_f_in_rf_data.size > 0:
            extra_col = np.full((contact_f_in_rf_data.shape[0], 1),
                                -1,
                                dtype=np.int32)
            contact_f_in_rf_data = np.c_[contact_f_in_rf_data, extra_col]

            rf_force = RFForce(time=contact_f_in_rf_data[:, 0],
                               force=contact_f_in_rf_data[:, 1],
                               axial_angle=contact_f_in_rf_data[:, 2],
                               lateral_angle=contact_f_in_rf_data[:, 3],
                               abl_point=contact_f_in_rf_data[:, -1],
                               position=np.full(
                                   (contact_f_in_rf_data.shape[0], 3),
                                   np.nan,
                                   dtype=np.float32)
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

                    idx_min = np.argmin(np.abs(point.force.time))
                    acq_time = point.force.systemTime[idx_min]

                    sys_time = rf_force.time
                    idx_min = np.argmin(np.abs(sys_time - acq_time))
                    rf_force.ablationPoint[idx_min] = point_id
                    rf_force.position[idx_min, :] = point_coord
        else:
            log.info('no points found in map, cannot update RF dataset with '
                     'EGM coordinates!')

        return MapRF(force=rf_force, ablation_parameters=rf_abl)

    @staticmethod
    def visitag_to_lesion(visitag_sites):
        """Convert VisiTags to base class lesions."""

        lesions = []
        for site in visitag_sites:
            rfi = [RFIndex(name=x.name, value=x.value) for x in site.RFIndex]
            lesions.append(Lesion(X=site.X,
                                  diameter=6.0,
                                  RFIndex=rfi
                                  )
                           )

        return lesions

    def _import_attributes(self):
        """
        Load info and file(s) associated with this map from study XML.

        Returns:
            None

        """

        xml_file = self.parent.repository.join(self.studyXML)
        with self.parent.repository.open(xml_file) as fid:
            root = xml.parse(fid).getroot()

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
            algorithm=map_item.find('RefAnnotationConfig').get('Algorithm'),
            connector=map_item.find('RefAnnotationConfig').get('Connector'))

        # get map coloring range table
        colorRangeItem = map_item.find('ColoringRangeTable')
        colorRangeTable = []
        for colorRange in colorRangeItem.findall('ColoringRange'):
            colorRangeTable.append(
                ColoringRange(Id=int(colorRange.get('Id')),
                              min=colorRange.get('Min'),
                              max=colorRange.get('Max'))
            )
        self.coloringRangeTable = colorRangeTable

        return True

    @staticmethod
    def _sort_rf_filenames(filenames, order='ascending'):
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
        recX : ndarray (3, 1)
            coordinates at which this point was recorded
        prjX : ndarray (3, 1)
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
        uniCoordinates : ndarray (3, 2)
            cartesian coordinates of the unipolar recording electrodes
            NOTE: coordinates of second unipolar electrode are NaN if
            unipolar channel names were read from ECG file only
        forceFile : str
            name of the points contact force file
            <map_name>_<point_name>_Contact_Force.txt
        force : PointForce
            contact force data for this point

    Methods:
        is_valid()
            check if point has LAT annotation within WOI
        load(egm_names_from_pos=False)
            load all data associated with this point
        import_ecg(channel_names)
            import ECG data for this point

    """

    def __init__(self, name, point_file,
                 coordinates=np.full((3, 1), np.nan, dtype=float),
                 tags=None,
                 egm_names_from_pos=False,
                 parent=None):
        """
        Constructor.

        Parameters:
             name : str
                name / identifier for this point
            point_file : str
                name of this points XML file
                <map_name>_<point_name>_Point_Export.xml
            coordinates : ndarray(3, 1)
                cartesian coordinates of recording position
            tags: list of str (optional)
                tags assigned to this point, i.e. 'Full_name' in study's
                TagsTable
            egm_names_from_pos : bool (optional)
                get names of EGM traces from electrode positions
                NOTE: second unipolar channel name and coordinates are only
                valid if this is True
            parent : CartoMap
                the map this point belongs to

        Raises:
            FileNotFoundError : if point's XML is not found

        Returns:
            None

        """

        super().__init__(name, coordinates=coordinates, parent=parent)

        file_loc = self.parent.parent.repository.join(point_file)
        if not self.parent.parent.repository.is_file(file_loc):
            log.info('Points export file {} does not exist'
                     .format(point_file))
            raise FileNotFoundError

        # add Carto3 specific attributes
        self.pointFile = point_file
        self.barDirection = None
        self.woi = np.array([])
        self.tags = tags
        self.ecgFile = None
        self.uniCoordinates = None
        self.forceFile = None
        self.forceData = None

        self.load(egm_names_from_pos=egm_names_from_pos)

    def load(self, egm_names_from_pos=False):
        """
        Load data associated with this point.

        Parameters:
            egm_names_from_pos : boolean
                If True, EGM electrode names are extracted from positions file.
                This also returns name and coordinates of the second unipolar
                channel.

        Returns:
            None

        """

        log.debug('Loading point data for point {}'.format(self.name))

        # read annotation data
        point_file = self.parent.parent.repository.join(self.pointFile)
        with self.parent.parent.repository.open(point_file) as fid:
            root = xml.parse(fid).getroot()

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
            impedance_time = np.empty(n_impedance_values, dtype=np.float32)
            for i, x in enumerate(impedance_item.findall('Impedance')):
                impedance_time[i] = x.get('Time')
                impedance_value[i] = x.get('Value')

            self.impedance = PointImpedance(time=impedance_time,
                                            value=impedance_value)

        self.ecgFile = root.find('ECG').get('FileName')

        # get egm names
        if not egm_names_from_pos:
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
            egm_names = channel_names_from_ecg_header(ecg_file_header)
        else:
            egm_names = channel_names_from_pos_file(
                self,
                study_root=self.parent.parent.repository.root,
                encoding=self.parent.parent.encoding
            )

        bipName = egm_names[0]
        uniName = egm_names[1]
        refName = egm_names[2]
        try:
            self.uniCoordinates = egm_names[3]
        except IndexError:
            log.debug('no coordinates for second unipolar channel found')
            self.uniCoordinates = np.stack((self.recX,
                                            np.full(3, np.nan, dtype=float)
                                            ),
                                           axis=-1)

        # TODO: is exporting of 2 unipolar channel names really
        #  necessary?? Why is this done??

        # now we can import the electrograms for this point
        egm_data = self.import_ecg([bipName,
                                    uniName[0],
                                    uniName[1],
                                    refName])
        # build egm traces
        self.egmBip = Trace(name=bipName,
                            data=egm_data[:, 0].astype(np.float32),
                            fs=1000.0)
        self.egmUni = [Trace(name=uniName[0],
                             data=egm_data[:, 1].astype(np.float32),
                             fs=1000.0),
                       Trace(name=uniName[1],
                             data=egm_data[:, 2].astype(np.float32),
                             fs=1000.0)
                       ]
        self.egmRef = Trace(name=refName,
                            data=egm_data[:, 3].astype(np.float32),
                            fs=1000.0)

        # get the closest surface vertex for this point
        if self.parent.surface.has_points():
            closest, distance, direct = self.parent.surface.get_closest_vertex(
                [self.recX]
            )
            if closest.shape[0] != 1:
                log.warning('found no or multiple surface vertices closest to '
                            'to point {}: {}'
                            .format(self.name, closest))
            self.prjX = np.array(closest[0], dtype=float)
            self.prjDistance = distance[0]
            self.barDirection = direct[0]

        # now get the force data for this point
        log.debug('reading force file for point {}'.format(self.name))
        self.forceFile = (self.pointFile.split('_Point_Export')[0]
                          + '_ContactForce.txt'
                          )

        if self.parent.parent.repository.is_file(self.forceFile):
            with self.parent.parent.repository.open(self.forceFile) as fid:
                self.forceData = read_force_file(
                    fid, encoding=self.parent.parent.encoding
                )
            # update base class force value
            self.force = self.forceData.force
        else:
            log.debug('No force file found for point {}'.format(self.name))

    def is_valid(self):
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

    def import_ecg(self, channel_names=None):
        """
        Load ECG data for this point.

        NOTE: THIS FUNCTION NEEDS A VALID ROOT DIRECTORY TO RETRIEVE DATA!

        Parameters:
            channel_names : string or list of string
                channel names to read

        Raises:
            KeyError : If a channel name is not found in ECG file

        Returns:
             ndarray (2500, 1)
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
            # array has shape (2500,) but (2500,1) is needed
            ecg_data = np.expand_dims(ecg_data, axis=1)

        return ecg_data


