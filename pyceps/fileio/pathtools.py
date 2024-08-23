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
from typing import TypeVar, Union, Optional, List, IO
import re
import logging
import sys
import zipfile
import py7zr


log = logging.getLogger(__name__)


TPy7zPath = TypeVar('TPy7zPath', bound='Py7zPath')  # workaround to type self


class Py7zPath:
    """
    Class to wrap file paths in 7z archives.

    Paths in 7z archives are given as strings or list of strings. Strings
    might be mistaken as paths to folders, therefore this wrapper is used.

    Attributes:
        archive: py7zr.SevenZipFile
        path : str
        _filenames : list of str
            filenames in archives
    """

    def __init__(
            self,
            root: Union[str, py7zr.SevenZipFile, TPy7zPath],
            path: str = ''
    ) -> None:
        """Constructor."""

        if isinstance(root, Py7zPath):
            self.archive = root.archive
        elif isinstance(root, str):
            self.archive = py7zr.SevenZipFile(root)
        elif isinstance(root, py7zr.SevenZipFile):
            self.archive = root
        else:
            raise TypeError('cannot build Py7zPath from type {}'
                            .format(type(root))
                            )

        self.path = path
        self._filenames = self.archive.getnames()

    def join(
            self,
            path: str
    ) -> TPy7zPath:
        """
        Build path within 7z archive.

        Parameters:
            path : str

        Returns:
            Py7zPath
        """

        # adjust folder separator on windows
        path = path.replace('\\', '/')

        new_path = Py7zPath(root=self.archive, path=path)
        if not new_path.exists():
            raise FileNotFoundError

        return new_path

    def open(
            self
    ) -> Optional[IO]:
        """Open file in 7z archive."""
        if not self.is_file():
            log.warning('cannot open folder!')
            return None

        # in case file was opened already
        self.archive.reset()

        data = self.archive.read([self.path])
        try:
            f_handle = data[self.path]
            # add name attribute for compatibility with io_BufferedReader
            # from other open methods
            f_handle.name = self.archive.filename + '/' + self.path
            return f_handle
        except KeyError:
            log.warning('cannot read {} in 7z archive {}'
                        .format(self.path, self.archive.filename))
            return None

    def list(
            self
    ) -> List[str]:
        """List names of current directory."""
        if not self.is_folder():
            # path points to file
            log.warning('7z path points to file, cannot list contents!')
            return []

        # get folder names
        folders = [f for f in self._filenames
                   if not os.path.splitext(f)[1]
                   ]

        if not self.path:
            # path points to root, list everything except fil in subfolders
            filelist = [f for f in self._filenames
                        if not f.startswith(tuple(folders))
                        ]
            return folders + filelist

        filelist = [os.path.basename(f) for f in self._filenames
                    if f.startswith(self.path)
                    ]
        # remove folder name from list
        filelist.remove(self.path)
        return filelist

    def exists(
            self
    ) -> bool:
        """Check if file or folder exists in archive."""
        if not self.path:
            # path points to root in 7z
            return True

        return self.path in self._filenames

    def is_file(
            self,
    ) -> bool:
        """Check if path points to file in archive."""
        _, ext = os.path.splitext(self.path)

        return self.exists() and not ext == ''

    def is_folder(
            self
    ) -> bool:
        """Check if path points to folder in archive."""
        _, ext = os.path.splitext(self.path)

        return self.exists() and ext == ''

    def __repr__(self):
        return ('Py7zPath(archive="{}", path="{}"'
                .format(self.archive.filename, self.path)
                )

    def __str__(self):
        return 'Py7zPath object at {}'.format(self.archive.filename)


# supported repository types
RepoType = Union[str, zipfile.Path, Py7zPath]


class Repository:
    """
    Base class representing a study data repository.

    Paths to files and folders are always given relative to the study root.

    Attributes:
        base : str, zipfile.Path, py7zr.SevenZipFile
            path to EAM data repository provided at instantiation.

        root : str, zipfile.Path, py7zr.SevenZipFile
            data root, might be different from base
            Type depends on repository:
            For folders this is a string, for ZIP archives this is
            zipfile.Path, for 7z archives this is py7zr.SevenZipFile.
        pwd : str (optional)
            password used for protected archives
    """

    FUNC_ID = {
        str: '_path_',
        zipfile.Path: '_zip_',
        py7zr.SevenZipFile: '_7z_',
        Py7zPath: '_7z_',
    }

    def __init__(
            self,
            path: str,
            pwd: str = ''
    ) -> None:
        """
        Constructor.

        Parameters:
             path : str
                path to repository
             pwd : bytes (optional)
                password for protected archives

        Raises:
            TypeError : If path type is not supported
            FileNotFoundError : If specified path points to nowhere
        """

        if not isinstance(path, str):
            raise TypeError('unknown type {} provided, must be {}'
                            .format(type(path), self.FUNC_ID.keys())
                            )

        if not os.path.exists(path):
            raise FileNotFoundError('path {} does not exist'.format(path))

        # base points to location given at creation time
        self.base = self.init_repository(path)

        # root points to actual data root. This might be different to base
        # if data lies within nested archive
        self.root = self.base

        self.pwd = pwd

    def is_file(
            self,
            filepath: RepoType
    ) -> bool:
        """
        Check if path points to existing file in EAM repository.

        Parameters:
             filepath : str, zipfile.Path, Py7zPath

        Returns:
            bool
        """

        func = getattr(self, self.FUNC_ID.get(type(filepath)) + 'is_file')

        return func(filepath)

    def is_folder(
            self,
            path: RepoType
    ) -> bool:
        """
        Check if path points to existing folder in EAM repository.

        Parameters:
             path : str, zipfile.Path, Py7zPath

        Returns:
            bool
        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'is_folder')

        return func(path)

    def is_archive(
            self,
            path: RepoType
    ) -> bool:
        """
        Check if path points to existing archive (ZIP, 7z) in EAM repository.

        Parameters:
            path : str, zipfile.Path, Py7zPath

        Returns:
            bool
        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'is_archive')

        return func(path)

    def list_dir(
            self,
            path: RepoType,
            regex: str = ''
    ) -> List[str]:
        """
        List contents of directory in EAM repository.

        Parameters:
             path : str, zipfile.Path, Py7zPath
             regex : str (optional)
                regular expression for pattern matching

        Returns:
            list of str
        """

        func = getattr(self, self.FUNC_ID.get(type(path)) + 'list_dir')

        return func(path, regex=regex)

    def join(
            self,
            path: str
    ) -> RepoType:
        """
        Build path within EAM repository.

        Parameters:
             path : str
                path to file/folder relative to repository root

        Returns:
            str, zipfile.Path, Py7zPath
        """

        func = getattr(self, self.FUNC_ID.get(type(self.root)) + 'join')

        return func(path)

    def open(
            self,
            filepath: RepoType,
            pwd: str = '',
            mode: str = 'rb'
    ) -> IO:
        """
        Open a file within EAM repository.

        Parameters:
            filepath : str, zipfile.Path, Py7zPath
            pwd : str (optional)
                password to use for protected archives
            mode : str (optional)
                file open mode

        Returns:
            file object
        """

        func = getattr(self, self.FUNC_ID.get(type(filepath)) + 'open')

        return func(filepath, pwd=pwd, mode=mode)

    def build_export_basename(
            self,
            folder_name: str
    ) -> str:
        """
        Build path to output folder for study data.

        If study root points to the folder containing <study>.xml the export
        folder is created in folder above root.
        If study root points to a file (in case of an invalid root) the export
        folder is created in the same folder as root.

        Parameters:
            folder_name : str
                folder to append to base name

        Returns:
            path  : str
        """

        study_base = self.get_base_string()

        if os.path.isfile(study_base):
            # study root points to file, export to same directory
            export_folder = os.path.join(os.path.dirname(study_base),
                                         folder_name)
        else:
            # study root points to a folder, export to folder above
            export_folder = os.path.join(study_base,
                                         '../',
                                         folder_name)

        # this should make path platform-independent
        return os.path.abspath(export_folder)

    @staticmethod
    def init_repository(
            path: Union[str, zipfile.Path, py7zr.SevenZipFile, Py7zPath]
    ) -> RepoType:
        """
        Build valid repository path.

        Parameters:
            path : str, zipfile.Path, py7zr.SevenZipFile

        Returns:
            str, zipfile.Path, Py7zPath
                If repository can not be build, an empty string is returned.
        """

        if isinstance(path, str):
            # make path absolute, platform independent
            path = os.path.abspath(path)
            if not os.path.exists(path):
                log.warning('EAM path {} not found!'.format(path))
                return ''

        if isinstance(path, Py7zPath):
            # already valid
            return path
        elif isinstance(path, zipfile.Path):
            # already valid
            return path
        elif zipfile.is_zipfile(path):
            # path points to ZIP archive, build zipfile.Path
            return zipfile.Path(path)
        elif py7zr.is_7zfile(path):
            # path points to 7z archive, build Py7zpath object
            return Py7zPath(path)
        elif os.path.isdir(path):
            # path points to folder, probably valid
            return path
        else:
            # location or file type is not supported
            log.warning('file {} is not supported as study repository!'
                        .format(os.path.basename(path))
                        )
        return ''

    def update_root(
            self,
            path: RepoType
    ) -> None:
        """
        Set repository root to new location.

        Note: This does not change path initially provided upon instantiation.

        Parameters:
            path: str, zipfile.Path, py7zr.SevenZipFile

        Returns:
            None
        """

        func1 = getattr(self, self.FUNC_ID.get(type(path)) + 'is_folder')
        func2 = getattr(self, self.FUNC_ID.get(type(path)) + 'is_archive')
        if not func1(path) and not func2(path):
            raise FileExistsError()

        self.root = self.init_repository(path)

    def get_base_string(self) -> str:
        """
        Get repository base location as string.

        Returns:
            str
        """
        study_base = self.base
        if isinstance(self.base, zipfile.Path):
            study_base = os.path.abspath(self.base.root.filename)
        elif isinstance(self.base, py7zr.SevenZipFile):
            study_base = os.path.abspath(self.base.filename)
        elif isinstance(self.base, Py7zPath):
            study_base = os.path.abspath(self.base.archive.filename)

        return study_base

    def get_root_string(self) -> str:
        """
        Get repository root location as string.

        Returns:
            str
        """
        study_root = self.root
        if isinstance(self.root, zipfile.Path):
            study_root = os.path.abspath(self.root.root.filename)
        if isinstance(self.root, py7zr.SevenZipFile):
            study_root = os.path.abspath(self.root.filename)

        return study_root

    @staticmethod
    def _path_is_file(
            filepath: str
    ) -> bool:
        """Check if path points to file."""
        return os.path.isfile(filepath)

    @staticmethod
    def _zip_is_file(
            filepath: zipfile.Path
    ) -> bool:
        """Check if path points to file in ZIP archive."""
        return filepath.exists() and filepath.is_file()

    @staticmethod
    def _7z_is_file(
            filepath: Py7zPath
    ) -> bool:
        """Check if path points to file in 7z archive."""
        return filepath.is_file()

    @staticmethod
    def _path_is_folder(
            path: str
    ) -> bool:
        """Check if path points to folder."""
        return os.path.isdir(path)

    @staticmethod
    def _zip_is_folder(
            path: zipfile.Path
    ) -> bool:
        """Check if path points to folder in ZIP archive."""
        if not path.name:
            # path points to root, treat as folder
            return True
        return path.exists() and path.is_dir()

    @staticmethod
    def _7z_is_folder(
            path: Py7zPath
    ) -> bool:
        """Check if path points to folder in 7z archive."""
        return path.is_folder()

    @staticmethod
    def _path_is_archive(
            path: str
    ) -> bool:
        """Check if path points to archive."""
        return zipfile.is_zipfile(path) or py7zr.is_7zfile(path)

    @staticmethod
    def _zip_is_archive(
            path: zipfile.Path
    ) -> bool:
        """Check if path points to archive within ZIP archive."""
        log.warning('nested ZIP archives are not supported!')
        return False

    @staticmethod
    def _7z_is_archive(
            path: Py7zPath
    ) -> bool:
        """Check if path points to archive within 7z archive."""
        log.warning('nested 7z archives are not supported!')
        return False

    @staticmethod
    def _path_list_dir(
            path: str,
            regex: str = ''
    ) -> List[str]:
        """List contents of folder."""
        return [f for f in os.listdir(path) if re.match(regex, f)]

    @staticmethod
    def _zip_list_dir(
            path: zipfile.Path,
            regex: str = ''
    ) -> List[str]:
        """List contents of folder within ZIP archive."""
        return [f.name for f in path.iterdir() if re.match(regex, f.name)]

    @staticmethod
    def _7z_list_dir(
            path: Py7zPath,
            regex: str = ''
    ) -> List[str]:
        """List contents of folder within ZIP archive."""
        return [f for f in path.list() if re.match(regex, f)]

    def _path_join(
            self,
            path: str
    ) -> str:
        """Append path to repository root."""
        return os.path.join(self.root, path)

    def _zip_join(
            self,
            path: str
    ) -> zipfile.Path:
        """Append path to repository root."""
        # check if folder or file is requested
        # folder names in zipfile end with "/"
        _, ext = os.path.splitext(path)
        path = path + '/' if path and not ext else path

        try:
            return self.root.joinpath(path)
        except FileNotFoundError:
            return self.root

    def _7z_join(
            self,
            path: str
    ) -> Py7zPath:
        """Build valid path if root is 7z archive."""
        try:
            return self.root.join(path)
        except FileNotFoundError:
            return self.root

    @staticmethod
    def _path_open(
            filepath: str,
            pwd: str = '',
            mode='rb'
    ) -> IO:
        """Open file."""
        return open(filepath, mode=mode)

    @staticmethod
    def _zip_open(
            filepath: zipfile.Path,
            pwd: str = '',
            mode: str = 'rb'
    ) -> IO:
        """Open file within ZIP archive."""
        version = sys.version_info
        if version.major == 3 and version.minor == 8:
            return filepath.open(mode='r', pwd=pwd.encode('utf-8'))

        # mode "rb" has to be given to read as binary file-like, although
        # not in documentation...?
        return filepath.open(mode='rb', pwd=pwd.encode('utf-8'))

    @staticmethod
    def _7z_open(
            filepath: Py7zPath,
            pwd: str = '',
            mode: str = 'rb'
    ) -> IO:
        return filepath.open()

    def __str__(self):
        return format(self.root)
