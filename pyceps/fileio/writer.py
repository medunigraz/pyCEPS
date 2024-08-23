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
from typing import Optional, Union, List
import numpy as np
import json

from .igb import IGBFile


class FileWriter:
    """
    A class implementing various file export methods.

    For writing files a FileWriter object is instantiated and the .dump
    method is used. File type to be written is determined from requested
    file extension.

    Attributes:
        log : logging.Logger
        _fileName : str
    """

    FILETYPES = {'.elem': '_write_elem',
                 '.lon': '_write_lon',
                 '.pts': '_write_pts',
                 '.pts_t': '_write_pts_t',
                 '.dat': '_write_dat',
                 '.dat_t': '_write_dat_t',
                 '.vtk': '_write_vtk',
                 '.igb': '_write_igb',
                 '.json': '_write_json'
                 }

    _np_to_vtk = {
        'uint8': 'unsigned_char',
        'uint16': 'unsigned_short',
        'uint32': 'unsigned_int',
        'uint64': 'unsigned_long',
        'int8': 'char',
        'int16': 'short',
        'int32': 'int',
        'int64': 'long',
        'float32': 'float',
        'float64':  'double',
        'complex64': 'float',
        'complex128': 'double'
    }

    def __init__(
            self
    ) -> None:
        """Constructor."""

        self.log = logging.getLogger(self.__class__.__name__)
        self._fileName = None

    def dump(
            self,
            filename: str,
            *args, **kwargs
    ) -> str:
        """
        Write data to file.

        File type is derived from extension of filename.

        Parameters:
            filename : str

        Returns:
            filename : str
        """

        _, ext = os.path.splitext(filename)
        if ext not in FileWriter.FILETYPES.keys():
            raise TypeError('filetype "{}" not supported!'.format(ext))

        self._fileName = filename

        # get the write function for file extension
        func = getattr(self, self.FILETYPES.get(ext))
        # write data and return filename
        return func(*args, **kwargs)

    def _write_elem(
            self,
            surfs: np.ndarray,
            etags: Optional[np.ndarray] = None
    ) -> str:
        """
        Write surface indices to .elem file.

        File specification:
            Single header line containing the number of elements, followed
            by one element definition per line. The element definitions are
            composed of an element type specifier string, the nodes for that
            element, then optionally an integer specifying the region to
            which the element belongs.

        Parameters:
            surfs : ndarray
                elements of shape (n, 3) for triangles or shape (n, 4) for
                quads
            etags : ndarray (optional)
                region(s) the elements belong to. Shape (n, 1)
                Note: explicit second dimension must be given!

        Raises:
            TypeError : if elements is other than triangles or quads.
            ValueError : if dimensions of surfs and etags do not match.

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.elem')

        isTris = surfs.shape[1] == 3
        isQuads = surfs.shape[1] == 4

        if not etags:
            etags = np.full((surfs.shape[0], 1), 0)
        # check etags dimensions
        if not etags.shape[0] == surfs.shape[0]:
            raise ValueError('dimension of etags does not match elements')

        # concatenate data
        data = np.concatenate((surfs, etags), axis=1)

        # header equals number of elements
        header = str(surfs.shape[0])

        if isTris:
            np.savetxt(self._fileName, data,
                       fmt='Tr %u %u %u %u',
                       comments='', header=header)
        elif isQuads:
            np.savetxt(self._fileName, data,
                       fmt='Qd %u %u %u %u %u',
                       comments='', header=header)
        else:
            raise TypeError('Export of only triangles and quads is supported!')

        self.log.debug('exported elements to {}'.format(self._fileName))

        return self._fileName

    def _write_lon(
            self,
            fibers,
            sheets: Optional[np.ndarray] = None
    ) -> str:
        """
        Write fibers (and sheets) to .lon file.

        File specification:
            Single header line with the number of fibre vectors defined in
            the file (1 for fibre direction only, 2 for fibre and sheet
            directions), and then one line per element with the values of
            the fibre vector(s).

        Parameters:
            fibers : ndarray
            sheets : ndarray (optional)

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.lon')

        header = '1' if sheets is None else '2'

        if sheets is None:
            np.savetxt(self._fileName, fibers,
                       fmt='%f %f %f',
                       comments='',
                       header=header)
        else:
            np.savetxt(self._fileName, np.hstack((fibers, sheets)),
                       fmt='%f %f %f %f %f %f',
                       comments='',
                       header=header)

        self.log.debug('exported fibers/sheets to {}'.format(self._fileName))

        return self._fileName

    def _write_pts(
            self,
            points: np.ndarray
    ) -> str:
        """
        Write point coordinates to .pts file.

        File specification:
            Single header line with the number of nodes, followed by the
            coordinates (x,y,z) of the nodes, one per line, in mm.

        Parameters:
            points : ndarray

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.pts')

        # header equals number of nodes
        header = str(points.shape[0])
        np.savetxt(self._fileName, points,
                   fmt='%f %f %f',
                   comments='',
                   header=header)

        self.log.debug('exported points to {}'.format(self._fileName))

        return self._fileName

    def _write_pts_t(
            self,
            points: Union[np.ndarray, List[np.ndarray]]
    ) -> str:
        """
        Write point coordinates to .pts_t file.

        File specification:
            Single header line with the number of frames, followed by
            frames. Each frame section starts with number of nodes, followed by
            coordinates (x,y,z) of the nodes, one per line, in mm.

        Parameters:
            points : ndarray, list of ndarray
                nodes for single frame frames or list of nodes per frame

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.pts_t')

        if not isinstance(points, list):
            points = [points]

        with open(self._fileName, 'w') as fid:
            # write number of frames
            fid.write('{}\n'.format(len(points)))

            for data in points:
                # write points per frame
                fid.write('{}\n'.format(data.shape[0]))
                np.savetxt(fid, data, fmt='%f %f %f')

        self.log.debug('exported points to {}'.format(self._fileName))

        return self._fileName

    def _write_dat(
            self,
            data: np.ndarray
    ) -> str:
        """
        Write data to .dat file.

        File specification:
            Data vectors with one value per line

        Parameters:
            data : ndarray
                data to write, must be 1D-array!

        Raises:
            ValueError : if data is not 1D-array

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.dat')

        if data.ndim != 1:
            raise ValueError('multidimensional data is not supported for '
                             'DAT files!')

        with open(self._fileName, 'w+')as f:
            data_str = '\n'.join(data.astype(str))
            f.write(data_str)
            f.write('\n')  # meshalyzer needs CR at end of file
            f.flush()

        self.log.debug('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_dat_t(
            self,
            data: Union[np.ndarray, List[np.ndarray]]
    ) -> str:
        """
        Write data to .dat_t file.

        File specification:
            Single header line with the number of frames, followed by
            frames. Each frame section starts with number of data points,
            followed by data at the nodes, one per line.

        Parameters:
            data : ndarray, list of ndarray
                nodes for single frame frames or list of nodes per frame

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.dat_t')

        if not isinstance(data, list):
            data = [data]

        with open(self._fileName, 'w+')as f:
            # write number of frames
            f.write('{}\n'.format(len(data)))

            for frame in data:
                f.write('{}\n'.format(frame.shape[0]))
                data_str = np.asarray(frame, dtype=str)
                data_str = '\n'.join(data_str)
                f.write(data_str)

            f.write('\n')  # meshalyzer needs CR at end of file
            f.flush()

        self.log.debug('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_vtk(
            self,
            points: np.ndarray,
            polygons: np.ndarray,
            data: Optional[List[dict]] = None
    ) -> str:
        """
        Write data as VTK file(s).

        File specification:
            file format follows VTK DataFile version 2.0
            see: https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html

        Parameters:
            points : ndarray (n, 3)
                x, y, z coordinates of mesh nodes
            polygons : ndarray (n, 3)
                triangulation
            data : list of dict (optional)
                additional point data or cell data
                each data segment is specified by a dict with fields
                name : string
                values : ndarray
                location : str (pointData or cellData)

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.vtk')

        with open(self._fileName, 'w+') as f:
            # write a title, followed by a blank line
            f.write('# vtk DataFile Version 2.0\n')
            f.write('LV display\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')

            # write vertex data
            f.write('POINTS {:d} float\n'.format(points.shape[0]))
            np.savetxt(f, points, fmt='%f')
            f.write('\n')

            # write triangulation data
            f.write('POLYGONS {:d} {:d}\n'
                    .format(polygons.shape[0], polygons.shape[0] * 4))
            np.savetxt(
                f,
                np.c_[np.full(polygons.shape[0], 3, dtype=np.int32), polygons],
                fmt='%d')
            f.write('\n')

            point_data = []
            cell_data = []
            if data:
                point_data = [d for d in data if d['location'] == 'pointData']
                cell_data = [d for d in data if d['location'] == 'cellData']

            # add PointData
            if point_data and len(point_data) == 1:
                name = point_data[0]['name']
                values = point_data[0]['values']
                try:
                    dim = values.shape[1]
                except IndexError:
                    values = np.expand_dims(values, axis=1)
                vtk_type = self._np_to_vtk[values.dtype.name]
                # add data to vtk file
                header = 'POINT_DATA {}\n'.format(values.shape[0])
                header += 'SCALARS {} {} {}\n'.format(name,
                                                      vtk_type,
                                                      values.shape[1])
                header += 'LOOKUP_TABLE default\n'
                np.savetxt(f, values, fmt='%s', comments='', header=header)
                f.write('\n')
            elif point_data and len(point_data) > 1:
                f.write('POINT_DATA {}\n'.format(points.shape[0]))
                f.write('FIELD FieldData {}\n'.format(len(point_data)))
                for field in point_data:
                    # add data to vtk file
                    vtk_type = self._np_to_vtk[field['values'].dtype.name]
                    header = '{} 1 {} {}\n'.format(field['name'],
                                                   field['values'].shape[0],
                                                   vtk_type)
                    np.savetxt(f, field['values'],
                               fmt='%s', comments='', header=header
                               )
                    f.write('\n')

            # add CellData
            if cell_data and len(cell_data) == 1:
                name = cell_data[0]['name']
                values = cell_data[0]['values']
                vtk_type = self._np_to_vtk[values.dtype.name]
                # add data to vtk file
                header = 'CELL_DATA {}\n'.format(values.shape[0])
                header += 'SCALARS {} {} {}\n'.format(name,
                                                      vtk_type,
                                                      values.shape[1])
                header += 'LOOKUP_TABLE default\n'
                np.savetxt(f, values, fmt='%s', comments='', header=header)
                f.write('\n')
            elif cell_data and len(cell_data) > 1:
                f.write('CELL_DATA {}\n'.format(points.shape[0]))
                f.write('FIELD FieldData {}\n'.format(len(cell_data)))
                for field in cell_data:
                    # add data to vtk file
                    vtk_type = self._np_to_vtk[field['values'].dtype.name]
                    header = '{} 1 {} {}\n'.format(field['name'],
                                                   field['values'].shape[0],
                                                   vtk_type)
                    np.savetxt(f, field['values'],
                               fmt='%s', comments='', header=header
                               )
                    f.write('\n')

        self.log.debug('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_igb(
            self,
            header: str,
            data: np.ndarray
    ) -> str:
        """
        Save spatio-temporal data to .igb file.

        File specification:
            1024 bytes long ASCII header followed by binary data
            header is composed of keyword/value pairs separated by white
            space and padded to 1024 bytes

        Parameters:
            header : str
            data : ndarray (n_points, t)

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.igb')

        igb_obj = IGBFile(self._fileName, mode='w')
        igb_obj.write(data.astype(np.single).T, header=header)
        igb_obj.close()

        self.log.debug('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_json(
            self,
            data_dict,
            **kwargs
    ) -> str:
        """
        Save data dictionary as JSON.

        Returns:
            filename : str
        """

        assert self._fileName.endswith('.json')

        with open(self._fileName, 'w') as f:
            json_string = json.dumps(data_dict, **kwargs)
            f.write(json_string)

        self.log.debug('exported data to {}'.format(self._fileName))

        return self._fileName
