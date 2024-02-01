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

import datetime
import logging
import os
from typing import Iterable
import numpy as np
import zipfile
import gzip
import pickle
import webbrowser

import py7zr

from src.fileio.pathtools import Repository
from src.datatypes.surface import SurfaceSignalMap
from src.fileio import FileWriter
from src.interpolation import (inverse_distance_weighting,
                               remove_redundant_points
                               )
from src.visualize import get_dash_app


log = logging.getLogger(__name__)


class EPStudy:
    """
    Base Class representing an EP study.

    Attributes:
        system : str
            name of the EAM system used
        repository : str
            path object pointing  to study data repository. Can be a folder or
            a ZIP archive
        pwd : str (optional)
            password to access encrypted ZIP archives
        encoding : str (optional)
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
        meshes : list of Surface objects (optional)
            additional meshes from e.g. CT data

    Methods:
        import_study()
            load basic information about the study, i.e. map names contained
            in data set, number of mapping points, ...
        import_maps(map_names)
            load mapping procedures
        is_root_valid(root_dir=None)
            checks if the repository points to a valid location,
            i.e. if it points to mapping data. If a root_dir is given as
            argument, this file path is checked
        set_root(root_dir)
            set the repository attribute to the given directory. The
            specified location is checked first for validity and set only if
            new root is valid
        list_maps(minimal=False)
            returns names of mapping procedures. If minimal is False only
            the names of imported maps are returned, else more detailed
            information is returned. Information is also send to logger for
            command line display
        imported_maps()
            get names of maps that were already imported. (These are the keys
            of attribute maps)
        build_export_basename(folder_name)
            build and create (if necessary) output folder for study data
        visualize()
            visualize the study in a local HTML page (dash).
            Note: This will lock the console!
        save(filename)
            save a pickled version of the study object
        load(filepath)
            load a pickled version of the study object
        export_additional_meshes()
            export any additional meshes contained in the study

    """

    EAM_SYSTEMS = ['carto3', 'precision']

    def __init__(self, system, study_repo, pwd='', encoding='cp1252'):
        """
        Constructor.

        Parameters:
            system : str
                Name of the EAM system
            study_repo : string
                path to EAM data
            pwd : str (optional)
                password used for protected ZIP files
            encoding : str
                file encoding to use (all files are read in binary mode)

        Raises:
            TypeError : if system name is unknown (or not yet implemented)
            TypeError : if study_repo is not of type string
            FileExistsError : if study file or folder does not exist

        """

        if system not in self.EAM_SYSTEMS:
            raise TypeError('Unknown EAM system {}!'
                            'Choose from: {}'
                            .format(system, self.EAM_SYSTEMS))

        self.system = system
        self.repository = Repository(study_repo)
        self.pwd = pwd.encode(encoding='UTF-8')
        self.encoding = encoding

        self.name = ''
        self.mapNames = []
        self.mapPoints = []
        self.maps = {}

        # additional meshes, i.e. from CT data
        self.meshes = []

    def import_study(self):
        raise NotImplementedError

    def import_maps(self, map_names=None, *args, **kwargs):
        """
        Pre-import checks. If a map is already part of the study user
        interaction is required to reload the study or to skip import.
        Actual import must be implemented by derived classes.

        Parameters:
            map_names : list of str
                the map names to import. Only valid map names, i.e. the
                names must be listed in mapNames attribute, are imported

        Returns:
            map_names : list
                valid names of maps to import

        Raises:
            ValueError : if command line input is invalid
        """

        if not self.mapNames:
            log.warning('No study info was found. Load study structure first!')
            return -1

        if map_names and not issubclass(type(map_names), Iterable):
            map_names = [map_names]

        if map_names is None:
            # import all available maps
            map_names = self.mapNames
        else:
            # check if map names are valid
            not_found = [x for x in map_names if x not in self.mapNames]
            if not_found:
                log.warning('map(s) {} not found in study!'.format(not_found))
            # import only valid map names
            map_names = [x for x in map_names if x in self.mapNames]

            # check if map(s) already loaded
            load_map = np.ones(len(map_names), dtype=np.bool_)
            for i, map_name in enumerate(map_names):
                if map_name in self.maps:
                    # map already loaded
                    user_input = input('Map {} already loaded, reload? [Y/N] '
                                       .format(map_name)
                                       )
                    # input validation
                    if user_input.lower() in ('y', 'yes'):
                        load_map[i] = True
                    elif user_input.lower() in ('n', 'no'):
                        load_map[i] = False
                    else:
                        # ... error handling ...
                        print(
                            'Error: Input {} unrecognised.'.format(user_input))
                        raise ValueError

            # these are the maps that actually are imported
            map_names = [x for x, y in zip(map_names, load_map) if y]

        return map_names

    def export_additional_meshes(self, *args, **kwargs):
        raise NotImplementedError

    def is_root_valid(self, root_dir=None):
        raise NotImplementedError

    def set_root(self, root_dir):
        raise NotImplementedError

    @classmethod
    def load(cls, filename, root=None):
        raise NotImplemented

    def list_maps(self, minimal=False):
        """
        Return names of maps in this study.

        Map names are also added to logger for command line display.
        Parameter minimal determines format: If True only map names are
        listed, if False more detailed info is sent to logger, i.e. all map
        names and which ones were already imported.

        Parameters:
            minimal : boolean (optional)
                Sets format of logger message

        Returns:
             list of tuples
                map names and number of egm points for all and imported maps
        """

        all_maps = ['{} ({} points)'.format(m, p)
                    for m, p in zip(self.mapNames, self.mapPoints)]
        imported_maps = ['{} ({} points)'.format(m, p)
                         for m, p in zip(self.mapNames, self.mapPoints)
                         if m in self.maps]

        if not minimal:
            log.info('Available maps: {}'
                     .format('; '.join(all_maps)))
            log.info('Imported maps:  {}'
                     .format('; '.join(imported_maps)))
        else:
            for name in self.mapNames:
                log.info('#{}'.format(name))

        return (list(zip(self.mapNames, self.mapPoints)),
                [(m, p) for m, p in zip(self.mapNames, self.mapPoints)
                 if m in self.maps]
                )

    def imported_maps(self):
        """Get names of maps that were already imported."""

        return list(self.maps.keys())

    def build_export_basename(self, folder_name):
        """Build and create (if necessary) output folder for study data.

        If study root points to the folder containing <study>.xml the export
        folder is created in folder above root.
        If study root points to a .zip or .pkl file (invalid root) the export
        folder is created in the same folder as root.
        """

        export_folder = self.repository.build_export_basename(folder_name)

        # check if export folder exists, create if necessary
        if not os.path.isdir(export_folder):
            os.mkdir(export_folder)

        return export_folder

    def save(self, filepath='', gz=False):
        """
        Save pickled version of study object.
        Note: File is only created if at least one map was imported!

        By default, the filename is the study's name, but can also be
        specified by the user.
        If the file already exists, user interaction is required to either
        overwrite file or specify a new file name.

        Using HIGHEST_PROTOCOL is almost 2X faster and creates a file that
        is ~10% smaller. Load times go down by a factor of about 3X.

        Parameters:
            filepath : string (optional)
                custom path for the output file
            gz : boolean (optional)
                use gzip compression or not

        Raises:
            ValueError : If user input is not recognised

        Returns:
            None
        """

        if not self.maps:
            log.info('No maps imported, nothing to save!')
            return

        if filepath:
            if filepath.lower().endswith('.pkl'):
                filepath = filepath[:-4]
            f_loc = os.path.abspath(filepath)
        else:
            f_loc = os.path.join(self.build_export_basename(''), self.name)

        # handle un-pickleable attributes
        study_base = self.repository.base
        study_root = self.repository.root
        if isinstance(self.repository.root, zipfile.Path):
            self.repository.root = os.path.abspath(
                self.repository.root.root.filename
            )
        if isinstance(self.repository.base, zipfile.Path):
            self.repository.base = os.path.abspath(
                self.repository.base.root.filename
            )
        if isinstance(self.repository.root, py7zr.SevenZipFile):
            self.repository.root = os.path.abspath(
                self.repository.root.filename
            )
        if isinstance(self.repository.base, py7zr.SevenZipFile):
            self.repository.base = os.path.abspath(
                self.repository.base.filename
            )

        f_loc += '.gz' if gz else '.pkl'
        if os.path.isfile(f_loc):
            user_input = input('Study object already exists, overwrite? [Y/N]')
            # input validation
            if user_input.lower() in ('y', 'yes'):
                pass
            elif user_input.lower() in ('n', 'no'):
                user_input = input('Save with suffix? [Y/N] ')
                if user_input.lower() in ('y', 'yes'):
                    suffix = ''
                    while not suffix:
                        suffix = input('Suffix: ')
                    f_loc, ext = os.path.splitext(f_loc)
                    f_loc = f_loc + '_' + suffix + ext
                elif user_input.lower() in ('n', 'no'):
                    return
            else:
                # ... error handling ...
                print('Error: Input {} unrecognised.'.format(user_input))
                raise ValueError

        if gz:
            f = gzip.open(f_loc, 'wb')
        else:
            f = open(f_loc, 'wb')

        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        # restore study root
        self.repository.root = study_root
        self.repository.base = study_base

        log.info('saved study to {}'.format(f_loc))

    def visualize(self, bgnd=None):
        """Visualize the study in dash."""

        log.info('visualizing study')
        log.warning('This will lock the console!')

        app = get_dash_app(self, bgnd=bgnd)
        webbrowser.open_new("http://127.0.0.1:8050")
        app.run_server(debug=False, use_reloader=False)


class EPMap:
    """
    Base class representing an EP mapping procedure.

    Attributes:
        name : str
            name of the mapping procedure
        parent : subclass of EPStudy
            the parent study for this map
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

    Methods:
        get_valid_points(return_invalid=False)
            returns valid mapping points recorded during procedure. If
            return_invalid is True, also invalid points are returned
        load_mesh()
            load the triangulated anatomical shell for this mapping procedure
        load_points()
            load mapping points data
        import_lesions()
            import abaltion lesions data
        get_map_ecg(ecg_names=None)
            Load/Build body surface ECGs for this mapping procedure.
            ecg_names specifies which ECG leads are imported
        interpolate_data(which=None)
            interpolate surface signal maps from mapping points data. which
            specifies the type of signal map, i.e. uni, bip, or lat
        export_mesh_vtk(basename='', maps_to_add=None, labels_to_add=None)
            export anatomical shell to VTK. maps_to_add specifies which
            surface signal maps are included (None export all available) and
            labels_to_add specifies which surface labels are included (None
            exports all available)
        export_mesh_carp(basename='')
            export anatomical shell to openCARP compatible formats
            (.pts and .elem)
        export_point_cloud(points=None, basename='')
            exports the mapping points coordinates to openCARP compatible
            formats (.pts)
        export_point_data(basename='', which=None, points=None)
            exports measurements from mapping points in openCARP compatible
            formats (.dat). which specifies the type of measurement
            (bip, uni, lat). points specifies for which points data is
            exported (None exports for all valid points)
        export_signal_maps(basename='', which=None)
            exports surface signal maps to openCARP compatible formats (
            .dat). which specifies which surface signal maps to export (None
            exports all available signal maps)
        export_point_egm(basename='', which=None, points=None)
            exports EGM traces for recording points to openCARP compatible
            formats (.igb). which specifies which traces are exported (bip,
            uni, ref). points specifies for which  points data is exported (
            None exports data for all valid points)
        export_point_ecg(basename='', which=None, points=None)
            exports surface ECG data stored along with mapping points to
            openCARP compatible formats (.igb). which specifies which traces
            are exported (any 12-lead ECG names). points specifies for which
            points data is exported (None exports data for all valid points)
        export_lesions(filename='')
            exports ablation sites data to openCARP compatible formats (.pts
            and .dat)
        get_rfi_names(return_counts=False)
            returns the names of RFI parameters stored in ablation data.

    """

    def __init__(self, name, parent=None):
        """
        Constructor.

        Parameters:
             name : str
             parent : subclass of EPStudy

        Raises:
            TypeError : if parent is not subclass of EPStudy

        """

        if not issubclass(type(parent), EPStudy):
            raise TypeError('Cannot set parent for EPMap of type {}'
                            .format(type(parent)))

        self.name = name
        self.parent = parent

        self.surfaceFile = ''
        self.surface = None
        self.points = []
        self.ecg = []
        self.lesions = []

    def get_valid_points(self, return_invalid=False):
        """
        Get valid points for this map.

        Parameters:
            return_invalid : boolean (optional)
                return invalid points as well

        Returns:
            list of CartoPoint
            tuple (valid, invalid) if return_invalid is True.
        """

        if not return_invalid:
            return [point for point in self.points if point.is_valid()]

        return ([point for point in self.points if point.is_valid()],
                [point for point in self.points if not point.is_valid()]
                )

    def load_mesh(self, *args, **kwargs):
        """Load triangulated representation of the anatomical shell."""
        raise NotImplementedError

    def load_points(self, *args, **kwargs):
        """Load mapping points recorded during mapping procedure."""
        raise NotImplementedError

    def import_lesions(self, *args, **kwargs):
        """Import ablation data for mapping procedure."""
        raise NotImplementedError

    def get_map_ecg(self, ecg_names=None, *args, **kwargs):
        """Build/Load body surface ECGs for this mapping procedure."""
        raise NotImplementedError

    def interpolate_data(self, which):
        """
        Create surface parameter maps by interpolating EGM data on mesh
        surface points.

        Surface maps are added to the surface.

        Parameters:
            which : str
                parameter to interpolate, options: ['uni', 'bip', 'act']

        Raises:
            KeyError : If parameter to interpolate is unknown

        Returns:
            None
        """

        log.info('creating surface parameter map {} for {}'
                 .format(which, self.name))

        if not self.surface:
            log.info('no mesh found, nothing to interpolate!')
            return

        if which in self.surface.get_map_names():
            log.info('surface map {} already created, nothing to do...')
            return

        mesh_points = self.surface.X
        valid_points = self.get_valid_points()
        if not len(valid_points) > 0:
            log.info('no valid points for interpolation!')
            return

        points_surf = np.stack([p.prjX for p in valid_points])
        unique_points, unique_ids = remove_redundant_points(points_surf)

        if which.lower() == 'lat':
            data = np.asarray([p.latAnnotation - p.refAnnotation
                               for p in valid_points])
        elif which.lower() == 'bip':
            data = np.asarray([p.bipVoltage for p in valid_points])
        elif which.lower() == 'uni':
            data = np.asarray([p.uniVoltage for p in valid_points])
        else:
            raise KeyError()

        interpolated = inverse_distance_weighting(unique_points,
                                                  data[unique_ids],
                                                  mesh_points,
                                                  k=7)

        # adjust array dims for compatibility
        interpolated = np.expand_dims(interpolated, 1)

        surf_map = SurfaceSignalMap(name=which.upper(),
                                    values=interpolated.astype(np.single),
                                    location='pointData',
                                    description='creationDate:{}'.format(
                                        datetime.datetime)
                                    )
        self.surface.add_signal_maps(surf_map)

    def export_mesh_vtk(self, basename='',
                        maps_to_add=None,
                        labels_to_add=None):
        """
        Save the anatomical shell as VTK file.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.<map>.surf.vtk

        Interpolated surface parameters can be added to the VTK file.

        Parameters:
            basename : string (optional)
                path and filename of the exported files
            maps_to_add : string or list of strings
                parameter maps to include in VTK file.
            labels_to_add : string or list of strings
                surface labels to include in VTK file

        Returns:
            None
        """

        log.info('exporting mesh for map {} to VTK'.format(self.name))

        if not self.surface or not self.surface.has_points():
            log.info('no anatomical shell found, nothing to export...')
            return

        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name + '.surf')

        if isinstance(maps_to_add, str):
            maps_to_add = [maps_to_add]
        if isinstance(labels_to_add, str):
            labels_to_add = [labels_to_add]

        filename = self.surface.dump_mesh_vtk(basename,
                                              maps_to_add=maps_to_add,
                                              labels_to_add=labels_to_add)

        log.info('exported anatomical shell to {}'.format(filename))

    def export_mesh_carp(self, basename=''):
        """
        Save an anatomical shell as CARP ascii files ending .pts and .elem

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.<map>.surf.<.pts or .elem>

        Parameters:
            basename : string (optional)
                path and filename of the exported files

        Returns:
            None
        """

        log.info('exporting surface mesh (CARP)')

        if not self.surface:
            log.warning('no surface mesh found for map {}'.format(self.name))
            return
        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name + '.surf')

        f = self.surface.dump_mesh_carp(basename)
        log.info('exported anatomical shell to {}'
                 .format(f + ' (.pts, .elem)'))

        return

    def export_point_cloud(self, points=None, basename=''):
        """
        Export EGM point coordinates to points file (.PTS).

        To files are exported:
            (i) <basename>.pc.pts : EGM recording positions, 'PointCloud'
            (ii) <basename>.ppc.pts : positions projected onto mesh surface,
                'InterpolatedSurface'

        By default, all valid EGM points are exported, but a list of
        EPPoints can also be specified.
        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.

        Parameters:
            points : list of CartoPoints (optional)
                points to be exported
            basename : string (optional)
                path and filename of the exported files

        Returns:
            None
        """

        if not points:
            points = self.get_valid_points()
        if not len(points) > 0:
            log.warning('no points found in map {}. Nothing to export...'
                        .format(self.name))
            return
        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name)

        writer = FileWriter()

        mesh_points = np.array([point.recX for point in points])
        # retrieve accepted mapping nodes projected onto surface
        surf_points = np.array([point.prjX for point in points])

        # create points files if necessary
        pts_file = '{}.pc.pts'.format(basename)
        log.info('exporting mapping points cloud')
        writer.dump(pts_file, mesh_points)

        pts_file = '{}.ppc.pts'.format(basename)
        log.info('exporting mapping points projected on surface')
        writer.dump(pts_file, surf_points)

        return

    def export_point_data(self, basename='', which=None, points=None):
        """
        Export recording point data in DAT format.

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".

        Following data can be exported:
            BIP : bipolar voltage
            UNI : unipolar voltage
            LAT : local activation time
            IMP : impedance
            FRC : contact force

        By default, data from all valid points is exported, but also a
        list of EPPoints to use can be given.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.ptdata.<parameter>.pc.dat

        Parameters:
            basename : string (optional)
                path and filename of the exported files
            which : list of strings
                parameter maps to include in DAT file. Options are 'LAT',
                'BIP', 'UNI', 'IMP', 'FRC'
           points : list of CartoPoints (optional)
                EGM points to export

        Returns:
            None
        """

        log.info('exporting surface map data')

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
            which = ['UNI', 'BIP', 'LAT', 'IMP', 'FRC']

        # export point files
        self.export_point_cloud(points=points, basename=basename)

        # export data
        writer = FileWriter()

        if "UNI" in which:
            data = [point.uniVoltage for point in points]
            dat_file = basename + '.ptdata.UNI.pc.dat'
            writer.dump(dat_file, data)
            log.info('exported surface map data to {}'.format(dat_file))

        if "BIP" in which:
            data = [point.bipVoltage for point in points]
            dat_file = basename + '.ptdata.BIP.pc.dat'
            writer.dump(dat_file, data)
            log.info('exported surface map data to {}'.format(dat_file))

        if "LAT" in which:
            data = [point.latAnnotation - point.refAnnotation
                    for point in points]
            dat_file = basename + '.ptdata.LAT.pc.dat'
            writer.dump(dat_file, data)
            log.info('exported surface map data to {}'.format(dat_file))

        if "IMP" in which:
            data = [point.impedance for point in points]
            dat_file = basename + '.ptdata.IMP.pc.dat'
            writer.dump(dat_file, data)
            log.info('exported surface map data to {}'.format(dat_file))

        if "FRC" in which:
            data = [point.force for point in points]
            dat_file = basename + '.ptdata.FRC.pc.dat'
            writer.dump(dat_file, data)
            log.info('exported surface map data to {}'.format(dat_file))

        return

    def export_signal_maps(self, basename='', which=None):
        """
        Export surface map data in DAT format.

        Files created can be associate data with point cloud of the
        high-resolution mesh.
        All associated signal maps are exported by default.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.map.<parameter>.dat for EGm data

        Returns:
            None
        """

        log.info('exporting surface map data for map {}'.format(self.name))

        if not self.surface:
            log.info('no surface found, nothing to export...')
            return

        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name)

        status = self.surface.dump_signal_map(basename=basename, which=which)

        log.info(status)

        return

    def export_point_egm(self, basename='', which=None, points=None):
        """
        Export mapping EGM traces in IGB format.

        Files created are labeled ".pc." and can be associated with
        recording location point cloud ".pc.pts" or with locations projected
        onto the high-resolution mesh".ppc.pts".
        Following traces can be exported:
            BIP : bipolar EGM
            UNI : unipolar EGM
            REF : reference signal
        By default EGMs for all valid points are exported, but also a
        list of EPPoints to use can be given.

        If no basename is explicitly specified, the map's name is used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.egm.<trace>.pc.igb

        Parameters:
            basename : string (optional)
                path and filename of the exported files
            which : string or list of strings (optional)
                EGM trace(s) to include in IGB file. Options are 'BIP', 'UNI',
                'REF'. If not specified, all are exported.
           points : list of CartoPoints (optional)
                EGM points to export. If not specified, data from all valid
                points in WOI is exported.

        Returns:
            None
        """

        log.info('exporting point egm data')

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
            which = ['BIP', 'UNI', 'REF']
        if isinstance(which, str):
            which = [which]

        # check requested channel names
        not_found = [item for item in which
                     if not item.upper() in ['BIP', 'UNI', 'REF']]
        if not_found:
            raise KeyError('EGM(s) {} not found for point {}'
                           .format(not_found, self.name))

        # export point files
        self.export_point_cloud(points=points, basename=basename)

        # store ecg traces as igb files
        writer = FileWriter()
        channel_data = np.array([])
        # save data channel-wise
        for channel in which:
            if channel.upper() == 'BIP':
                channel_data = np.asarray([x.egmBip.data for x in points])
            elif channel.upper() == 'UNI':
                channel_data = np.asarray([x.egmUni[0].data
                                           for x in points])
            elif channel.upper() == 'REF':
                channel_data = np.asarray([x.egmRef.data for x in points])
            if channel_data.size == 0:
                log.warning('no data found for channel {}, nothing to export!'
                            .format(channel))
            # save data to igb
            # Note: this file cannot be loaded with the CARTO mesh but rather
            #       with the exported mapped nodes
            header = {'x': channel_data.shape[0],
                      't': channel_data.shape[1],
                      'unites_t': 'ms',
                      'unites': 'mV',
                      'dim_t': channel_data.shape[0]-1, # (num_tsteps - 1) * inc_t
                      'org_t': 0,
                      'inc_t': 1}

            filename = '{}.egm.{}.pc.igb'.format(basename, channel)
            f = writer.dump(filename, header, channel_data)
            log.info('exported EGM trace {} to {}'.format(channel, f))

        return

    def export_point_ecg(self, basename='', which=None, points=None):
        """
        Export surface ECG traces in IGB format.

        ECG data for points differs on EP systems, needs to be implemented
        in specific data type.
        """
        raise NotImplementedError

    def export_map_ecg(self, basename=''):
        """Export representative body surface ECG to JSON file.

        If no basename is explicitly specified, the map's name and the
        method for generating the representative ECGs are used and
        files are saved to the directory above the study root.
        Naming convention:
            <basename>.bsecg.<method>.json

        Parameters:
            basename : string (optional)
                path and filename of the exported files
        Returns:
              None
        """

        log.info('exporting body surface ECGs for map {}'.format(self.name))

        if not self.ecg:
            log.warning('no body surface ECG data found, nothing to export...')
            return

        if not basename:
            basename = self.parent.build_export_basename(self.name)
            basename = os.path.join(basename, self.name)

        writer = FileWriter()

        for bsecg in self.ecg:
            filename = '{}.bsecg.{}.json'.format(basename, bsecg.method)

            # build timeline
            t = np.linspace(start=0.0,
                            stop=bsecg.traces[0].data.shape[0] / 1000,
                            num=bsecg.traces[0].data.shape[0]
                            )
            # create data dict
            bsecg_json = dict()
            bsecg_json['t'] = t.round(decimals=3).tolist()

            data_dict = dict()
            for signal in bsecg.traces:
                data_dict[signal.name] = signal.data.tolist()

            bsecg_json['ecg'] = data_dict

            f = writer.dump(filename, bsecg_json, indent=2)
            log.info('exported body surface ECG trace(s) {} to {}'
                     .format([t.name for t in bsecg.traces], f))

    def export_lesions(self, filename=''):
        """
        Export lesion data.

        Ablation sites are exported as .PTS and .PTS_T files along
        with the RFI values as .DAT and .DAT_T
        Naming convention:
            <study_name>.lesion.<pts or pts_t> for point coordinates
            <study_name>.lesion.<RFI name>.<dat or dat_t>

        Parameters:
             filename : string (optional)
                custom export location and file names

        Returns:
            None
        """

        if not len(self.lesions) > 0:
            log.info('no lesion data found for map {}'.format(self.name))
            return

        log.info('exporting lesion(s) data...')

        writer = FileWriter()

        if not filename:
            basename = self.parent.build_export_basename(self.name)
        else:
            basename = os.path.abspath(filename)
        export_file = os.path.join(basename, self.name + '.lesions')

        # dump points
        points = np.array([site.X for site in self.lesions])
        writer.dump(export_file + '.pts', points)
        writer.dump(export_file + '.pts_t', points)

        # dump RFI
        # get RFIndex names
        names, counts = self.get_rfi_names(return_counts=True)
        for name, count in zip(names, counts):
            # check validity first
            if not count == len(self.lesions):
                log.warning('cannot export RFI data "{}", wrong number of '
                            'data points: {}!'.format(name, count))
                continue
            rfi = [x.value for lesion in self.lesions for x in lesion.RFIndex
                   if x.name == name]
            # dump RFI data
            writer.dump(export_file + '.' + name + '.dat', np.array(rfi))
            writer.dump(export_file + '.' + name + '.dat_t', np.array(rfi))

    def get_rfi_names(self, return_counts=False):
        """Return unique RF parameter names in lesions data."""

        names = [x.name for lesion in self.lesions for x in lesion.RFIndex]
        names, counts = np.unique(names, return_counts=True)

        if return_counts:
            return names, counts

        return names


class EPPoint:
    """
    Base class representing a recording point.

    Attributes:
        name : str
            identifier for this recording point
        parent : subclass of EPMap
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

    Methods:
        is_valid()

    """

    def __init__(self, name,
                 coordinates=np.full((3, 1), np.nan, dtype=float),
                 parent=None):
        """
        Constructor.

        Parameters:
            name : string
            coordinates : ndarray (3, 1)
                the coordinates where this point was recorded
            parent : EPMap

        Raises:
            TypeError : if parent is not of type EPMap

        """

        if not issubclass(type(parent), EPMap):
            raise TypeError('Cannot set parent for EPPoint of type {}'
                            .format(type(parent)))

        self.name = name
        self.parent = parent

        # location info
        self.recX = coordinates
        self.prjX = np.full((3, 1), np.nan, dtype=float)
        self.prjDistance = np.nan

        # annotation info
        self.refAnnotation = np.nan
        self.latAnnotation = np.nan

        # voltages
        self.uniVoltage = np.nan
        self.bipVoltage = np.nan

        # signal traces
        self.egmBip = None
        self.egmUni = None
        self.egmRef = None

        self.impedance = np.nan
        self.force = np.nan

    def is_valid(self):
        """Check if this point is valid."""
        raise NotImplementedError
