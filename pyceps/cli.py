#!/usr/bin/env python3
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

"""
Command line interface for pyEPmap
"""
import argparse
import os
import traceback
import shutil
import sys
from argparse import ArgumentParser, Action
import logging
import tempfile
from typing import Tuple

from pyceps.carto import CartoStudy
from pyceps.precision import PrecisionStudy


logger = logging.getLogger('pyceps')


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
        list of arguments or "DEFAULT"

    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, 'DEFAULT')
        else:
            setattr(namespace, self.dest, values)


def valid_path(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit('Could not find {}. Aborting...'.format(path))

    return path


def get_args():
    """Command line argument parser."""

    parser = ArgumentParser(
        prog='pyceps',
        # add_help=False,
        # formatter_class=RawTextHelpFormatter,
    )

    # specify location of data
    # Note: subsequent arguments are mutually exclusive. The user must enter
    #       either one!
    group = parser.add_argument_group('Data location')
    load = group.add_mutually_exclusive_group(required=True)
    load.add_argument(
        '--study-repository',
        type=str,
        default=None,
        help='Import study from EAM repository.\n'
             'Specify path to folder or to ZIP file containing study data.\n'
             'IMPORTANT: use --system to specify EAM system!'
    )
    load.add_argument(
        '--study-file',
        type=str,
        default=None,
        help='Load study from previously created pyCEPS file.\n'
             'If study root saved in pyCEPS file does not point to a valid '
             'location, it can be set with --change-root'
    )

    bio = parser.add_argument_group('Export')
    bio.add_argument(
        '--convert',
        type=str,
        nargs='?',
        default=None,
        const='ALL',
        help='Convenience function to export complete EAM data set.\n'
             'When importing from EAM repository all available maps are '
             'loaded. When importing from pyCEPS file all maps currently in '
             'the file are exported or --import-map might be used to import '
             'additional map(s) before export.\n'
             'Alternatively a specific map name can be given.\n'
             'Calls all functions under Advanced Exports with default values.'
    )
    bio.add_argument(
        '--save-study',
        type=str,
        nargs='?',
        const='DEFAULT',
        help='Save study as pyCEPS file.\n'
             'Default location is folder above study root, default name is '
             'study name e.g. <study_root>/../<study_name>.pyceps\n'
             'Custom location and file name can be given alternatively. All '
             'export files are redirected to this location.'
    )
    bio.add_argument(
        '--keep-ecg',
        action='store_true',
        help='Save point ECG data in pyCEPS export file.'
    )

    vis = parser.add_argument_group('Visualization')
    vis.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the study. This opens a local HTML page in the '
             'standard browser. NOTE: This will lock the console!'
    )

    aio = parser.add_argument_group('Advanced Import/Export')
    aio.add_argument(
        '--import-map',
        nargs='+',
        type=str,
        default=None,
        help='Name of the map to be imported from study repository.\n'
             'Argument "all" will import all maps in the study.'
    )
    aio.add_argument(
        '--map',
        type=str,
        default=None,
        help='Specify map name to subsequently work with.'
    )
    aio.add_argument(
        '--dump-mesh',
        action='store_true',
        help='Save anatomical shell for current "--map" in openCARP and VTK '
             'format.\n'
             'For VTK files, all available surface parameter maps and '
             'surface labels are included.\n'
             'Note: surface maps can also be exported as .dat with '
             '"--dump-surface-map".\n'
             'Default: <study_root>/../<map>.surf.[pts,elem,vtk]'
    )
    aio.add_argument(
        '--dump-point-data',
        action='store_true',
        help='Export data for recording points for current "--map" in '
             'openCARP format.\n'
             'For each mapping point following data are exported: unipolar '
             'voltages (UNI), bipolar voltages (BIP) and local activation '
             'time (LAT), point identifier (NAME), time stamps for '
             'annotations (LAT, REF), window of interest (WOI_START, '
             'WOI_END), impedance (IMP, if available), and contact force '
             '(FRC, if available).\n'
             'Default: <study_root>/../<map>.ptdata.<parameter>.pc.dat'
    )
    aio.add_argument(
        '--dump-point-ecgs',
        type=str,
        action=OptionalListParser,
        nargs='*',
        choices=['I', 'II', 'III',
                 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        help='Export ECG traces for all valid points associated '
             'with current "--map" to IGB.\n'
             'Dimension: Nx2500 (valid recorded points x ms)\n'
             'Note: Requires valid study root!\n'
             'If no traces are specified, all surface ECGs are '
             'exported by default.\n'
             'Default: <study_root>/../<map>.ecg.<lead>.pc.igb'
    )
    aio.add_argument(
        '--dump-point-egms',
        action='store_true',
        help='Export point EGM traces for unipolar, bipolar and reference '
             'channels for all valid points associated with current "--map".\n'
             'Dimension: Nx2500 (valid recorded nodes x ms)\n'
             'Default: e.g. <study_root>/../<map>.egm.<lead>.pc.igb'
    )
    aio.add_argument(
        '--dump-map-ecgs',
        action='store_true',
        help='Export representative 12-lead body surface ECGs associated '
             'with current "--map" to JSON.\n'
             'Default: <study_root>/../<map>.bsecg.<method>.json'
    )
    aio.add_argument(
        '--dump-surface-maps',
        action='store_true',
        help='Export surface maps associated with current "--map" to DAT.\n'
             'Default: <study_root>/../<map>.map.<parameter>.dat'
    )
    aio.add_argument(
        '--dump-lesions',
        action='store_true',
        help='Export lesion data associated with current "--map".\n'
             'Default: <study_root>/../<map>.lesions.<RFI_name>.dat'
    )

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument(
        '--change-root',
        type=valid_path,
        default=None,
        help='Change location of EAM data repository.\n'
             'Parent directory of "--study-xml" for Carto3 or folder for '
             'Precision. Can be absolute path to folder or ZIP file.\n'
             'Used only when data is loaded from pyCEPS file.'
    )
    misc.add_argument(
        '--egm-from-pos',
        action='store_true',
        help='Retrieve EGM channel names from recording positions during '
             'import.\n'
             'Note: Requires valid study root!'
    )
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
    misc.add_argument('--password',
                      type=str,
                      default='',
                      help='Password for protected archives.'
                      )

    main_args, _ = parser.parse_known_args()

    # configure clinical mapping system from which to import
    conditional_parser = argparse.ArgumentParser(parents=[parser],
                                                 add_help=False
                                                 )
    if main_args.study_repository:
        system = conditional_parser.add_argument_group('Clinical Mapping System')
        system.add_argument('--system',
                            required=main_args.study_repository,
                            type=str.upper,
                            choices=['CARTO', 'PRECISION'],
                            help='Specify the clinical mapping system that '
                                 'recorded the study.'
                            )

    return conditional_parser.parse_args()


def configure_logger(log_level: str) -> Tuple[int, str]:
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
                               pwd=args.password,
                               encoding=args.encoding)
            study.import_study()
            study.import_paso()
        elif args.system == 'PRECISION':
            study = PrecisionStudy(args.study_repository,
                                   pwd=args.password,
                                   encoding=args.encoding)
        else:
            raise KeyError('unknown EAM system specified!')

    else:
        study_file = os.path.abspath(args.study_file)
        logger.info('loading study from file {}'.format(study_file))

        # read in study from a pkl file
        # Note: both inputs are mutually exclusive
        if not study_file.lower().endswith('.pyceps'):
            # supposedly reading the carto folder directly
            study_file += '.pyceps'

        if not os.path.isfile(study_file):
            raise FileNotFoundError('could not find {}'.format(study_file))

        # update argument in case of later usage
        args.study_file = study_file

        # now we can open the file and try to load the object
        # read file line-wise, not using eTree for performance reasons
        with open(study_file, 'r') as fid:
            line = fid.readline()
            if not line == '<?xml version="1.0" encoding="utf-8"?>\n':
                logger.warning('unknown file format, aborting!')
                return None, args
            line = fid.readline()
            try:
                system = line.split('system="')[1].split('"')[0]
            except IndexError:
                logger.warning('unable to determine EAM system from file.\n'
                               'key "system" not found in line {}'
                               .format(line))
                return None, args

        if system.lower() == 'carto3':
            study = CartoStudy.load(args.study_file,
                                    password=args.password,
                                    repository_path=args.change_root
                                    )
        elif args.system == 'precision':
            study = PrecisionStudy.load(args.study_file,
                                        password=args.password,
                                        repository_path=args.change_root
                                        )

        else:
            raise KeyError('unknown EAM system specified!')
        # TODO: notification if root has changed (needed to process unsaved changes)

        # verify study root
        if not study.is_root_valid():
            # if study root was set to .pkl folder no valid root was found
            logger.warning('study root invalid, use --change-root to set a '
                           'valid path.')

    # list maps for overview
    study.list_maps()

    return study, args


def export_map_data(study, map_name, args):
    """Export data for specified maps."""

    # handle explicit export location
    out_path = ''
    if args.save_study:
        out_path = '' if args.save_study == 'DEFAULT' else args.save_study
    out_path = study.resolve_export_folder(os.path.dirname(out_path))

    # save carto mesh
    if args.dump_mesh:
        study.maps[map_name].export_mesh_carp(out_path)
        surf_maps = study.maps[map_name].surface.get_map_names()
        surf_labels = study.maps[map_name].surface.get_label_names()
        study.maps[map_name].export_mesh_vtk(output_folder=out_path,
                                             maps_to_add=surf_maps,
                                             labels_to_add=surf_labels
                                             )

    # dump point data for recording points
    if args.dump_point_data:
        study.maps[map_name].export_point_data(out_path)
        study.maps[map_name].export_point_info(out_path)

    # dump ECG traces for recording points
    if args.dump_point_ecgs:
        study.maps[map_name].export_point_ecg(
            output_folder=out_path,
            which=(None if args.dump_point_ecgs == 'DEFAULT'
                   else args.dump_point_ecgs)
        )

    # dump EGM traces for recording points
    if args.dump_point_egms:
        study.maps[map_name].export_point_egm(out_path)

    # dump representative ECGs for map
    if args.dump_map_ecgs:
        study.maps[map_name].export_map_ecg(out_path)

    # dump surface signal maps to DAT
    if args.dump_surface_maps:
        study.maps[map_name].export_signal_maps(out_path)

    # export lesion data
    if args.dump_lesions:
        study.maps[map_name].export_lesions(out_path)

    # check if additional meshes are part of the study
    if study.meshes and args.dump_mesh:
        logger.info('found additional meshes in study, exporting...')
        study.export_additional_meshes(out_path)


def execute_commands(args):
    # import/load study
    study, args = load_study(args)
    data_changed = False  # indicates if data was added/reloaded

    # work out what has to be imported
    import_maps = []
    if args.convert:
        if args.convert.lower() == 'all':
            import_maps += study.mapNames
        else:
            import_maps.append(args.convert)
    if args.import_map:
        if args.import_map[0].lower() == 'all':
            import_maps += study.mapNames
        else:
            import_maps += args.import_map
    # remove duplicates
    import_maps = list(dict.fromkeys(import_maps))

    # check if all requested map names are valid
    invalid_maps = [n for n in import_maps if n not in study.mapNames]
    if invalid_maps:
        logger.warning('map(s) with name(s) {} is not part of study!'
                       .format(invalid_maps)
                       )
    # remove invalid from list
    import_maps = [e for e in import_maps if e not in invalid_maps]

    # now we can import maps from EAM repo
    if import_maps:
        logger.info('need to import map(s): {}'.format(import_maps))

        if args.study_file and not study.is_root_valid():
            logger.warning('a valid study root is necessary to import maps!')
        else:
            study.import_maps(import_maps,
                              egm_names_from_pos=args.egm_from_pos)
            # import lesion data for all loaded maps
            for map_name in study.maps.keys():
                study.maps[map_name].import_lesions(directory='')
            data_changed = True

    # work out which map(s) to process
    export_maps = []
    if args.convert and args.map:
        logger.info('using --map together with --convert is not supported!\n'
                    'will use --convert only to proceed')
        args.map = ''

    if args.convert:
        if args.convert.lower() == 'all':
            export_maps += study.maps.keys()
        else:
            export_maps.append(args.convert)

    if args.map and args.map in study.maps.keys():
        export_maps = [args.map]
    elif args.map and args.map in study.mapNames:
        logger.info('map {} not imported, use --import-map'.format(args.map))
    elif args.map:
        logger.info('map {} not found in study'.format(args.map))

    if export_maps:
        logger.info('Selected map(s) for export: {}'.format(export_maps))
    else:
        logger.info('no map specified, continuing...')

    # work out what to export
    if args.convert:
        args.dump_mesh = True
        args.dump_point_data = True
        args.dump_point_ecgs = 'DEFAULT'
        args.dump_point_egms = True
        args.dump_map_ecgs = True
        args.dump_surface_maps = True
        args.dump_lesions = True

    # process selected map(s)
    for map_name in export_maps:
        logger.info('exporting data for map {}'.format(map_name))
        export_map_data(study, map_name, args)

    # save study
    if not args.save_study and data_changed:
        logger.debug('unsaved changes found!')
        user_input = input('There are unsaved changes, save them now? [Y/N] ')
        # input validation
        if user_input.lower() in ('y', 'yes'):
            logger.debug('user selected to save changes')
            args.save_study = 'DEFAULT'
        elif user_input.lower() in ('n', 'no'):
            logger.debug('user selected to continue without saving changes')
        else:
            logger.warning('Unknown user input {}'.format(user_input))

    pyceps_loc = ''
    if args.save_study:
        pyceps_loc = study.save(None if args.save_study == 'DEFAULT' else
                                args.save_study,
                                keep_ecg=args.keep_ecg)

    # redirect log file
    if pyceps_loc:
        base_file, _ = os.path.splitext(pyceps_loc)
        log_file = base_file + '_import.log'
    else:
        log_file = os.path.join(
            study.build_export_basename(''),
            study.name + '_import.log'
        )

    return study, log_file


def run():
    # get CL arguments from parser
    cl_args = get_args()

    # initialize logger and set downstream logger to same logging level
    log_fid, log_path = configure_logger(cl_args.logger_level)

    # import the EP study
    ep_study = None
    log_file = os.path.join(os.getcwd(), 'import.log')
    try:
        ep_study, log_file = execute_commands(cl_args)
    except:
        logger.error('File import finished with errors!\n{}'
                     .format(traceback.format_exc()))
    finally:
        # close handlers for logging
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        logging.shutdown()

        # save log file to disk
        if os.access(os.path.dirname(log_file), os.W_OK):
            os.close(log_fid)
            shutil.copy(log_path, log_file)
            print('import log saved to {}'.format(log_file))
        else:
            print('cannot save log file, no write permission for {}'
                  .format(log_file))
            os.close(log_fid)
        os.remove(log_path)

    # everything is done, visualize if requested
    # NOTE: This will lock the console
    if ep_study and cl_args.visualize:
        ep_study.visualize()


if __name__ == '__main__':
    print(
        'pyCEPS  Copyright (C) 2023  Robert Arnold\n'
        'This program comes with ABSOLUTELY NO WARRANTY;\n'
        'This is free software, and you are welcome to redistribute '
        'it under certain conditions; see LICENSE.txt for details.\n'
    )
    run()
