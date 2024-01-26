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
import re
import logging
import zipfile
import py7zr


log = logging.getLogger(__name__)


class Py7zPath:
    """
    Class to wrap file paths in 7z archives for proper function selection.

    Paths in 7z archives are given as strings or list of strings. Strings
    might be mistaken as paths in folders, therefore this wrapper is used.

    """

    def __init__(self, path):
        self.path = path


class Repository:
    """
    Base class representing a study data repository.

    Paths to files and folders are always given relative to the study root.


    Attributes:
        root : path to EAM data repository
            type depends on repository
            For folders this is a string, for ZIP archives this is zipfile.Path,
            for 7z archives this is py7zr.SevenZipFile.
        base : path provided at instantiation

    """

    FUNC_ID = {str: '_path_',
               zipfile.Path: '_zip_',
               py7zr.SevenZipFile: '_7z_',
               Py7zPath: '_7z_',
               # list: '_7z_',  # folder paths in 7z are given as list
               }

    def __init__(self, basepath='', pwd=''):
        """
        Constructor.

        Parameters:
             basepath : str, zipfile.Path, py7zr.SevenZipFile
             pwd : bytes
                password for protected archives

        Raises:
            TypeError
            FileNotFoundError

        """

        if not isinstance(basepath, tuple(self.FUNC_ID.keys())):
            raise TypeError('unknown type {} provided, must be {}'
                            .format(type(basepath), self.FUNC_ID.keys())
                            )

        # base points to location given at creation time
        self.base = self.build_path(basepath)

        # root points to actual data root. This might be different to base
        # if data lies within nested archive
        self.root = self.base

    def is_file(self, filepath):
        """
        Check if path points to existing file in EAM repository.

        Parameters:
             filepath : str, zipfile.Path, list

        Returns:
            bool
        """

        func = getattr(self, self.FUNC_ID.get(type(filepath)) + 'is_file')

        return func(filepath)

    def is_folder(self, path):
        """
        Check if path points to existing folder in EAM repository.

        Parameters:
             path : str, zipfile.Path, list

        Returns:
            bool
        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'is_folder')

        return func(path)

    def is_archive(self, path):
        """
        Check if path points to existing archive (ZIP, 7z) in EAM repository.

        Parameters:
            path : str, zipfile.Path, list

        Returns:
            bool

        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'is_archive')

        return func(path)

    def list_dir(self, path, regex=''):
        """
        List contents of directory in EAM repository.

        Parameters:
             path : str, zipfile.Path, list
             regex : str

        Returns:
            list of str
        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'list_dir')

        return func(path, regex=regex)

    def join(self, path):
        """
        Build path within EAM repository.

        Parameters:
             path : str

        Returns:
            str, zipfile.Path
        """

        func = getattr(self, self.FUNC_ID.get(type(self.root)) + 'join')

        return func(path)

    def open(self, filepath, pwd=None, mode='rb'):
        """
        Open a file in EAM repository.

        Parameters:
            filepath : str, zipfile.Path, list
            pwd : bytes (optional)
            mode : str (optional)

        Returns:
            file object
        """

        func = getattr(self, self.FUNC_ID.get(type(filepath)) + 'open')

        return func(filepath, pwd=pwd, mode=mode)

    def build_export_basename(self, folder_name):
        """Build path to output folder for study data.

        If study root points to the folder containing <study>.xml the export
        folder is created in folder above root.
        If study root points to a .zip or .pkl file (invalid root) the export
        folder is created in the same folder as root.
        """

        study_root = self.base
        # convert archive paths to str
        if isinstance(study_root, zipfile.Path):
            study_root = os.path.abspath(str(study_root.root.filename))
        if isinstance(study_root, py7zr.SevenZipFile):
            study_root = os.path.abspath(study_root.filename)

        if os.path.isfile(study_root):
            # study root points to ZIP or PKL file, export to same directory
            export_folder = os.path.join(os.path.dirname(study_root),
                                         folder_name)
        else:
            # study root points to a folder, export to folder above
            export_folder = os.path.join(study_root,
                                         '../',
                                         folder_name)

        # this should make path platform-independent
        return os.path.abspath(export_folder)

    def build_path(self, path):
        """Build valid path."""

        if isinstance(path, str):
            path = os.path.abspath(path)
            if not os.path.exists(path):
                log.warning('EAM path {} not found!'.format(path))
                return ''
                # raise FileNotFoundError('EAM repository at {} not found'
                #                         .format(path))

        if isinstance(path, list):
            # if 7z folder is requested return root (py7zr.SevenZipFile)
            return self.root
        elif isinstance(path, zipfile.Path):
            # already valid
            return path
        elif zipfile.is_zipfile(path):
            # build zipfile.Path
            return zipfile.Path(path)
        elif py7zr.is_7zfile(path):
            # build py7zr.SevenZipFile object
            return py7zr.SevenZipFile(path)

        return path

    def update_root(self, path):
        """
        Set root to new location.

        Note: This does not change path initially provided upon instantiation.

        """

        func1 = getattr(self, self.FUNC_ID.get(type(path)) + 'is_folder')
        func2 = getattr(self, self.FUNC_ID.get(type(path)) + 'is_archive')
        if not func1(path) and not func2(path):
            raise FileExistsError()

        self.root = self.build_path(path)

    def _path_is_file(self, filepath):
        return os.path.isfile(filepath)

    def _zip_is_file(self, filepath):
        return filepath.exists() and filepath.is_file()

    def _7z_is_file(self, filepath):
        # TODO: check if file is in archive
        if isinstance(filepath.path, list):
            _, ext = os.path.splitext(filepath.path[1])
        else:
            _, ext = os.path.splitext(filepath.path)

        return not ext == ''

    def _path_is_folder(self, path):
        return os.path.isdir(path)

    def _zip_is_folder(self, path):
        if not path.at:
            # path points to root, treat as folder
            return True
        return path.exists() and path.is_dir()

    def _7z_is_folder(self, path):
        # TODO: check if folder is in archive
        if isinstance(path.path, list):
            _, ext = os.path.splitext(path.path[1])
        else:
            _, ext = os.path.splitext(path.path)
        return not ext

    def _path_is_archive(self, path):
        return zipfile.is_zipfile(path)

    def _zip_is_archive(self, path):
        return path.is_file() and path.filename.suffix == '.zip'

    def _7z_is_archive(self, path):
        if isinstance(path, list):
            return path.path[1].endswith == '.7z'
        else:
            return path.path.endswith == '.7z'

    def _path_list_dir(self, path, regex=''):
        return [f for f in os.listdir(path) if re.match(regex, f)]

    def _zip_list_dir(self, path, regex=''):
        return [f.name for f in path.iterdir() if re.match(regex, f.name)]

    def _7z_list_dir(self, path, regex=''):
        # TODO: support nested folders
        return [f for f in self.root.getnames() if re.match(regex, f)]

    def _path_join(self, path):
        return os.path.join(self.root, path)

    def _zip_join(self, path):
        # check if folder or file is requested
        # folder names in zipfile end with "/"
        _, ext = os.path.splitext(path)
        path = path + '/' if path and not ext else path

        # check if nested zip is requested
        if path.endswith('.zip'):
            raise NotImplementedError(
                'Nested archive files are not supported!')

        try:
            return self.root.joinpath(path)
        except FileNotFoundError:
            return self.root

    def _7z_join(self, path):
        # check if file or folder is requested
        _, ext = os.path.splitext(path)
        if not ext:
            # folder requested
            return Py7zPath([path, path])

        # check if file in subdirectory is requested
        subdir, file = os.path.split(path)
        if subdir:
            return Py7zPath([subdir, path])

        # file is requested
        return Py7zPath(path)

    def _path_open(self, filepath, pwd=None, mode='rb'):
        return open(filepath, mode=mode)

    def _zip_open(self, filepath, pwd=None, mode='rb'):
        # mode "rb" has to be given to read as binary file-like, although
        # not in documentation...?
        return filepath.open(mode='rb', pwd=pwd)

    def _7z_open(self, filepath, pwd=None, mode='rb'):
        # in case file was opened already
        self.root.reset()
        data = self.root.read(filepath.path)
        if isinstance(filepath.path, list):
            return data[filepath.path[1]]
        else:
            return data[filepath.path]

    def __str__(self):
        return format(self.root)
