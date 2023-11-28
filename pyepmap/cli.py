#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of pyEPmap.
#
# pyEPmap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyEPmap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyEPmap.  If not, see <http://www.gnu.org/licenses/>.

"""
Command line interface for pyEPmap
"""

import os
import shutil
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, Action
import logging
import tempfile

import numpy
import numpy as np
numpy.seterr(all='raise')

from pyepmap.fileio.cartoio import CartoStudy
from pyepmap.fileio.precisionio import PrecisionStudy


logger = logging.getLogger('pyepmap')


class LogFormatter(logging.Formatter):
    """A class to customize logging formats."""

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        else:
            self._style._fmt = '%(levelname)s: %(message)s'

        return super().format(record)


class OptionalListParser(Action):
    """
    Combine optional argument '?' and list arguments '+'.

    Returns:
        list of arguments or "NONE"

    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, 'NONE')
        else:
            setattr(namespace, self.dest, values)


def valid_path(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit('Could not find {}. Aborting...'.format(path))

    return path


def get_args():
    """Command line argument parser."""

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    # configure clinical mapping system from which to import
    system = parser.add_argument_group('Clinical Mapping System')
    system.add_argument('--system',
                        required=True,
                        type=str.upper,
                        choices=['CARTO', 'PRECISION'],
                        help='Specify the clinical mapping system that '
                             'recorded the study.'
                        )

    # specify location of data
    # Note: subsequent arguments are mutually exclusive. The user must enter
    #       either one!
    group = parser.add_argument_group('Import study from')
    load = group.add_mutually_exclusive_group(required=True)
    load.add_argument(
        '--study-repository',
        type=str,
        default=None,
        help='Import study from repository. Absolute path to folder or to '
             'ZIP file containing study data.'
    )
    load.add_argument(
        '--pkl-file',
        type=str,
        default=None,
        help='Load study from <abs-path-pkl-file>.pkl\n'
             'If study root saved in PKL file does not point to a valid '
             'location, it can be set with --study-root'
    )

    # study related arguments
    study = parser.add_argument_group('EP Study')
    study.add_argument(
        '--study-root',
        type=valid_path,
        default=None,
        help='Parent directory of "--study-xml" for Carto3 or folder for '
             'Precision. Can be absolute path to folder or ZIP file.'
    )
    study.add_argument(
        '--save-study',
        type=str,
        nargs='?',
        const='DEFAULT',
        help='Save study PKL file\n'
             'Default location is folder above study root,  default name is '
             'study name e.g. <study_root>/../<map>.pkl\n'
             'Custom location and file name can be given alternatively'
    )

    # map related arguments
    maps = parser.add_argument_group('Study Maps')
    maps.add_argument(
        '--import-map',
        nargs='+',
        type=str,
        default=None,
        help='Name of the map to be imported from study repository.\n'
             'Argument "all" will import all maps in the study.'
    )
    maps.add_argument(
        '--map',
        type=str,
        default=None,
        help='Specify map name to subsequently work with.'
    )

    io = parser.add_argument_group('Export')
    io.add_argument(
        '--dump-mesh',
        action='store_true',
        help='Save anatomical shell for current "--map" in openCARP and VTK '
             'format.\n'
             'For VTK files, all available surface parameter maps and '
             'surface labels are included.\n'
             'Note: surface maps can also be exported as .dat with '
             '"--dump-surface-map".\n'
             'Default: <study_root>/../<map>.[pts,elem,vtk]'
    )
    io.add_argument(
        '--dump-point-ecgs',
        type=str,
        action=OptionalListParser,
        nargs='*',
        choices=['I', 'II', 'III',
                 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        help='Export ecg traces for all valid points associated '
             'with current "--map" to IGB.\n'
             'Dimension: Nx2500 (valid recorded points x ms)\n'
             'Note: Requires valid study root!\n'
             'If no traces are specified, all surface ECGs are '
             'exported by default.\n'
             'Default: <study_root>/../<map>.ecg.V1.pc.igb'
    )
    io.add_argument(
        '--dump-point-egms',
        action='store_true',
        help='Export point EGMs for all valid points associated '
             'with current "--map".\n'
             'Dimension: Nx2500 (valid recorded nodes x ms)\n'
             'Default: e.g. <study_root>/../<map>.egm.BIP.is.igb'
    )
    io.add_argument(
        '--dump-map-ecgs',
        action='store_true',
        help='Export representative 12-lead body surface ECGs associated '
             'with current "--map" to JSON.\n'
             'Default: <study_root>/../<map>.bsecg.<method>.json'
    )
    io.add_argument(
        '--dump-surface-maps',
        action='store_true',
        help='Export surface maps associated with current "--map" to DAT.\n'
             'Default: <study_root>/../<map>.map.is.BIP.dat'
    )
    io.add_argument(
        '--dump-lesions',
        action='store_true',
        help='Export lesion data associated with current "--map".\n'
             'Default: <study_root>/../<map>.lesions.<RFI_name>.dat'
    )

    vis = parser.add_argument_group('Visualization')
    vis.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the study. This opens a local HTML page in the '
             'standard browser. NOTE: This will lock the console!'
    )

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument(
        '--logger-level',
        type=str,
        default='INFO',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        help='Define level of information sent to command line.'
    )
    misc.add_argument(
        '--encoding',
        type=str,
        default='cp1252',
        help='Set encoding for file imports. Default: "cp1252".'
    )

    return parser.parse_args()


def configure_logger(log_level):
    """
    Set logging console and file formats.

    Messages are written to a temporary file and can be saved to disk later

    Returns:
        fid : file handle to an open file
        path : absolute pathname
    """

    # create temporary file for logging
    fid, path = tempfile.mkstemp()

    # configure console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    # set a format which is simpler for console use
    console.setFormatter(LogFormatter())

    # configure file handler
    file = logging.FileHandler(path, 'w', encoding='utf-8')
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter('%(asctime)s '
                                        '%(levelname)s '
                                        '%(name)s '
                                        '%(message)s',
                                        '%Y-%m-%d %H:%M:%S'),
                      )

    logging.basicConfig(level=logging.DEBUG,
                        handlers=[console, file])

    return fid, path


def load_study(args):
    """Import study data from repository or load existing study object."""

    if args.study_repository:
        logger.info('importing {} study'.format(args.system))
        if args.system == 'CARTO':
            study = CartoStudy(args.study_repository,
                               encoding=args.encoding)
        elif args.system == 'PRECISION':
            study = PrecisionStudy(args.study_repository,
                                   encoding=args.encoding)
        else:
            raise KeyError('unknown EAM system specified!')

    else:
        study_pkg = os.path.abspath(args.pkl_file)
        logger.info('loading {} study PKL'.format(args.system))
        # read in study from a pkl file
        # Note: both inputs are mutually exclusive
        if not study_pkg.lower().endswith(('.pkl', '.pkl.gz')):
            # supposedly reading the carto folder directly
            study_pkg += '.pkl'

        if not os.path.isfile(study_pkg):
            raise FileNotFoundError('could not find {}'.format(study_pkg))

        # update argument in case of later usage
        args.study_pkg = study_pkg

        # now we can load the object
        if args.system == 'CARTO':
            study = CartoStudy.load(study_pkg, root=args.study_root)
        elif args.system == 'PRECISION':
            study = PrecisionStudy.load(study_pkg, root=args.study_root)
        else:
            raise KeyError('unknown EAM system specified!')

        # verify study root
        if study.studyRoot == os.path.splitext(study_pkg)[0]:
            # if study root was set to .pkl folder no valid root was found
            logger.warning('use --study-root to set a valid path.')

    # list maps for overview
    study.list_maps()

    return study, args


def execute_commands(args):
    # import/load study
    study, args = load_study(args)

    # update study root if selected
    if args.study_root:
        success = study.set_root(os.path.abspath(args.study_root))
        if not success:
            logger.warning('could not set study root to {}'
                           .format(args.study_root)
                           )
        logger.info('current study root set is to {}'
                    .format(study.study_root)
                    )

    # import additional map
    if args.import_map:
        if not study.is_root_valid():
            logger.warning('a valid study root is necessary to import maps!')
        elif args.import_map[0].lower() == 'all':
            logger.info('importing all maps in study')
            study.import_maps(study.mapNames)
            # import lesion data
            for ep_map in study.maps.values():
                ep_map.import_lesions(directory=None)
        else:
            logger.info('importing study map "{}"'.format(args.import_map))
            study.import_maps(args.import_map)
            # import lesion data
            study.maps[args.import_map].import_lesion(directory=None)

    # select map to work with
    study_map = None
    if args.map and args.map in study.mapNames:
        study_map = args.map
        logger.info('Selected map: {}'.format(study_map))
    elif args.map and args.map not in study.mapNames:
        logger.info('map {} not imported, use --import-map'.format(args.map))
        logger.info('find available maps in study using `--list-maps`')
    elif args.map:
        logger.info('map {} not found in study'.format(args.map))
    else:
        logger.info('no map specified, continuing...')

    # process selected map
    if study_map:
        # save carto mesh
        if args.dump_mesh:
            study.maps[study_map].export_mesh_carp()
            surf_maps = study.maps[study_map].surface.get_map_names()
            surf_labels = study.maps[study_map].surface.get_label_names()
            study.maps[study_map].export_mesh_vtk(maps_to_add=surf_maps,
                                                  labels_to_add=surf_labels)
            if study.meshes:
                logger.info('found additional meshes in study, exporting...')
                study.export_additional_meshes()

        # dump ECG traces for recording points
        if args.dump_point_ecgs:
            if not study.is_root_valid():
                logger.warning('a valid study root is necessary to dump ECG '
                               'data for recording points!')
            else:
                study.maps[study_map].export_point_ecg(
                    which=(None if args.dump_point_ecgs == 'NONE'
                           else args.dump_point_ecgs)
                )

        # dump EGM traces for recording points
        if args.dump_point_egms:
            study.maps[study_map].export_point_egm()

        # dump representative ECGs for map
        if args.dump_map_ecgs:
            study.maps[study_map].export_map_ecg()

        # dump surface signal maps to DAT
        if args.dump_surface_maps:
            study.maps[study_map].export_signal_maps()

        # export lesion data
        if args.dump_lesions:
            study.maps[study_map].export_lesions()

    # save study
    if args.save_study:
        study.save(None if args.save_study == 'DEFAULT' else args.save_study)

    return study


def run():
    # get CL arguments from parser
    cl_args = get_args()

    # initialize logger and set downstream logger to same logging level
    log_fid, log_path = configure_logger(cl_args.logger_level)

    # import the EP study
    ep_study = None
    log_file = os.path.join(os.getcwd(), 'import.log')
    try:
        ep_study = execute_commands(cl_args)
        # redirect log file
        log_file = os.path.join(ep_study.build_export_basename(''),
                                ep_study.name + '_import.log'
                                )
    except Exception as err:
        logger.error('File import finished with errors!\n{}'
                     .format(err))
    finally:
        # close handlers for logging
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        logging.shutdown()

        # save log file to disk
        os.close(log_fid)
        shutil.copy(log_path, log_file)
        print('import log saved to {}'.format(log_file))
        os.remove(log_path)

    # everything is done, visualize if requested
    # NOTE: This will lock the console
    if ep_study and cl_args.visualize:
        ep_study.visualize()


if __name__ == '__main__':
    run()
