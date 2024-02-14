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
import numpy as np
import json

from .igb import IGBFile


class FileWriter:
    """A class implementing various file export methods."""

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

    def __init__(self):
        """Constructor."""

        self.log = logging.getLogger(self.__class__.__name__)
        self._fileName = None

    def dump(self, filename, *args, **kwargs):
        """Write data to file."""

        _, ext = os.path.splitext(filename)
        if ext not in FileWriter.FILETYPES.keys():
            raise TypeError('filetype "{}" not supported!'.format(ext))

        self._fileName = filename

        # get the write function for file extension
        func = getattr(self, self.FILETYPES.get(ext))
        # write data and return filename
        return func(*args, **kwargs)

    def _write_elem(self, surfs, etags=0):
        """Write surface indices to .elem file."""

        assert self._fileName.endswith('.elem')

        isTris = True if surfs.shape[1] == 3 else False
        isQuads = True if surfs.shape[1] == 4 else False

        # header equals number of elements
        header = str(surfs.shape[0])
        if isTris:
            np.savetxt(self._fileName, surfs,
                       fmt='Tr %u %u %u {}'.format(etags),
                       comments='', header=header)
        elif isQuads:
            np.savetxt(self._fileName, surfs,
                       fmt='Qd %u %u %u %u {}'.format(etags),
                       comments='', header=header)
        else:
            raise TypeError('Export of only triangles and quads is supported!')

        self.log.info('exported elements to {}'.format(self._fileName))

        return self._fileName

    def _write_lon(self, fibers, sheets=None):
        """Write fibers (and sheets) to .lon file."""

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

        self.log.info('exported fibers/sheets to {}'.format(self._fileName))

        return self._fileName

    def _write_pts(self, points):
        """Write point coordinates to .pts file."""

        assert self._fileName.endswith('.pts')
        if not isinstance(points, np.ndarray):
            raise TypeError('Received unknown datatype for points')

        # header equals number of nodes
        header = str(points.shape[0])
        np.savetxt(self._fileName, points,
                   fmt='%f %f %f',
                   comments='',
                   header=header)

        self.log.info('exported points to {}'.format(self._fileName))

        return self._fileName

    def _write_pts_t(self, points):
        """Write point coordinates to .pts_t file."""

        assert self._fileName.endswith('.pts_t')
        if not isinstance(points, np.ndarray):
            raise TypeError('Received unknown datatype for points')

        # TODO: add support for multidimensional points

        # header equals number of nodes
        header = '{}\n{}'.format(1, points.shape[0])
        np.savetxt(self._fileName, points,
                   fmt='%f %f %f',
                   comments='',
                   header=header)

        self.log.info('exported points to {}'.format(self._fileName))

        return self._fileName

    def _write_dat(self, data):
        """Write data int .dat file."""

        assert self._fileName.endswith('.dat')

        with open(self._fileName, 'w+')as f:
            data_str = np.asarray(data, dtype=str)
            data_str = '\n'.join(data_str)
            f.write(data_str)
            f.write('\n')  # meshalyzer needs CR at end of file
            f.flush()

        self.log.info('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_dat_t(self, data):
        """Write data int .dat_t file."""

        assert self._fileName.endswith('.dat_t')

        # TODO: add support for multidimensional points

        with open(self._fileName, 'w+')as f:
            data_str = np.asarray(data, dtype=str)
            f.write('{}\n{}\n'.format(1, data.shape[0]))
            data_str = '\n'.join(data_str)
            f.write(data_str)
            f.write('\n')  # meshalyzer needs CR at end of file
            f.flush()

        self.log.info('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_vtk(self, points, polygons, data=None):
        """Write data as VTK file(s)."""

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

        self.log.info('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_igb(self, header, data):
        """Save ecg data to .igb file."""

        assert self._fileName.endswith('.igb')

        igb_obj = IGBFile(self._fileName, mode='w')
        igb_obj.write(data.astype(np.single).T, header=header)
        igb_obj.close()

        self.log.info('exported data to {}'.format(self._fileName))

        return self._fileName

    def _write_json(self, data_dict, **kwargs):
        """Save data dictionary as JSON."""

        assert self._fileName.endswith('.json')

        with open(self._fileName, 'w') as f:
            json_string = json.dumps(data_dict, **kwargs)
            f.write(json_string)

        self.log.info('exported data to {}'.format(self._fileName))

        return self._fileName
