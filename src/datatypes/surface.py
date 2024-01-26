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

import logging
import numpy as np
from collections import namedtuple

from src.fileio.writer import FileWriter


log = logging.getLogger(__name__)


class Surface:
    """
    A class representing a surface recorded by a clinical system.

    Attributes:
        X : ndarray (n_vertices, 3)
            mesh vertices coordinates
        tris : ndarray(n_tris, 3)
            triangulation
        XNormals : ndarray (n_vertices, 3)
            vertices normals
        trisNormals : ndarray (n_vertices, 3)
            surface normals
        labels : list of SurfaceLabel
        signalMaps : list of SurfaceSignalMap
            clinical system-defined or custom signal property maps

    Methods:
        has_points()
            check if there are any vertices/faces in this surface
        get_map_names()
            get list of all included surface signal map names
        get_map(name)
            get a specific surface signal map
        get_label_names()
            get list of all included surface label names
        get_label(name)
            get a specific surface label
        add_signal_maps(maps)
         add SurfaceSignalMap to surface
        add_labels(labels)
            add SurfaceLabel to surface
        dump_mesh_vtk(filename, maps_to_add=None, labels_to_add=None)
            export triangulated surface mesh to VTK. SurfaceSignalMap(s) and
            SurfaceLabel(s) can be added by name
        dump_mesh_carp(filename)
            export triangulated surface mesh to openCARP compatible format (
            .pts and .elem)
        dump_signal_maps(basename, which=None)
            export SurfaceSignalMap(s) to openCARP compatible format (.dat)
        get_closest_vertex(points, limit_to_triangulation=False)
            find the closest surface vertex to a point or list of points
            given in cartesian coordinates
        get_center_of_mass()
            find center of mass for the triangulated surface mesh
        get_free_boundary()
            find indices of vertices add an edge
        remove_vertices(index)
            safely remove vertices from triangulated mesh
        remove_tris(index)
            safely remove faces from triangulated mesh

    """

    def __init__(self, vertices, triangulation,
                 vertices_normals=np.empty((0, 3), dtype=np.single),
                 tris_normals=np.empty((0, 3), dtype=np.single),
                 labels=None,
                 signal_maps=None):
        """
        Constructor.

        Parameters:
            vertices : ndarray (n_verts, 3)
                cartesian coordinates of vertices building the mesh
            triangulation : ndarray (n_tris, 3)
                the triangulation
            vertices_normals : ndarray (n_verts, 3) (optional)
            tris_normals :  ndarray (n_tris, 3) (optional)
            labels : SurfaceLabel or list of SurfaceLabel
                labels added to surface
            signal_maps : SurfaceSignalMap or list of SurfaceSignalMap
                surface signal maps added to surface

        """

        self.X = vertices
        self.tris = triangulation

        self.XNormals = vertices_normals
        self.trisNormals = tris_normals
        self.labels = []
        self.signalMaps = []

        self.add_signal_maps(signal_maps)
        self.add_labels(labels)

    def has_points(self):
        """Check if there are any vertices/faces in this surface."""

        return len(self.X) > 0

    def get_map_names(self):
        """Return list of included surface signal map names."""
        if not self.signalMaps:
            return []

        return [x.name for x in self.signalMaps]

    def get_map(self, name):
        """
        Get a surface signal map by name.

        Parameters:
             name : str

        Raises:
            KeyError : if name not found in surface signal maps

        Returns:
            SurfaceSignalMap

        """

        smap = [m for m in self.signalMaps if m.name == name]
        if len(smap) != 1:
            raise KeyError('no or multiple map(s) with name {} found in '
                           'surface'
                           .format(name))
        return smap[0]

    def get_label_names(self):
        """Return list of included surface label names."""
        if not self.labels:
            return []

        return [x.name for x in self.labels]

    def get_label(self, name):
        """
        Get a surface label by name.

        Parameters:
             name : str

        Raises:
            KeyError : if name not found in surface signal maps

        Returns:
            SurfaceLabel

        """
        label = [lbl for lbl in self.labels if lbl.name == name]
        if len(label) != 1:
            raise KeyError('no or multiple label(s) with name {} found in '
                           'surface'
                           .format(name))
        return label[0]

    def add_signal_maps(self, maps):
        """Add surface maps.

        Parameters:
            maps : SurfaceSignalMap or list of SurfaceSignalMap

        Raises:
              TypeError : if  other type than SurfaceSignalMap
              KeyError : if signal map with same name already exits

        Returns:
              None
        """

        log.debug('adding signal maps')

        if not maps:
            return

        if not issubclass(type(maps), list):
            maps = [maps]
        if not all(isinstance(m, SurfaceSignalMap) for m in maps):
            raise TypeError('surface signal maps must be of type {}'
                            .format(type(SurfaceSignalMap)))

        for m in maps:
            if m.name in self.get_map_names():
                raise KeyError('cannot add surface map with same name: {}'
                               .format(m.name))
            self.signalMaps.append(m)
            log.debug('added signal map {}'.format(m.name))

        return

    def add_labels(self, labels):
        """Add surface labels.

        Parameters:
            labels : SurfaceLabel or list of SurfaceLabel

        Raises:
              TypeError : if  other type than SurfaceLabel
              KeyError : if label with same name already exits

        Returns:
              None
        """

        log.debug('adding surface labels')

        if not labels:
            return

        if not issubclass(type(labels), list):
            labels = [labels]
        if not all(isinstance(m, SurfaceLabel) for m in labels):
            raise TypeError('surface labels must be of type {}'
                            .format(type(SurfaceSignalMap)))

        for m in labels:
            if m.name in self.get_label_names():
                raise KeyError('cannot add label with same name: {}'
                               .format(m.name))
            self.labels.append(m)
            log.debug('added surface label {}'.format(m.name))

        return

    def dump_mesh_vtk(self, filename, maps_to_add=None, labels_to_add=None):
        """
        Save mesh to VTK file.
        Surface signal map(s) and surface label(s) can be added by name.

        Parameters:
            filename : str
                path to output file
            maps_to_add : list of str (optional)
                names of the surface signal maps to add.
            labels_to_add : list of str
                names of the surface signal maps to add.

        Raises:
            KeyError : if a name is not found in signal maps or labels

        Returns:
            filename : str
                name of the file data was saved to

        """

        if not filename.endswith('.vtk'):
            filename += '.vtk'

        log.debug('exporting mesh to VTK: {}'.format(filename))

        data = []
        if maps_to_add:
            for name in maps_to_add:
                if name not in self.get_map_names():
                    raise KeyError('Surface map "{}" not found!'.format(name))
                m = [x for x in self.signalMaps if x.name == name][0]
                data.append(m.to_vtk_dict())
                log.debug('added surface map {}'.format(name))
        if labels_to_add:
            for name in labels_to_add:
                if name not in self.get_label_names():
                    raise KeyError('Surface label "{}" not found!'
                                   .format(name))
                m = [x for x in self.labels if x.name == name][0]
                data.append(m.to_vtk_dict())
                log.debug('added surface label {}'.format(name))

        writer = FileWriter()
        # TODO: add surface map data and group-ID if available
        return writer.dump(filename,
                           self.X,
                           self.tris,
                           data=data if data else None)

    def dump_mesh_carp(self, filename):
        """
        Save mesh to openCARP files .pts and .elem.

        Parameters:
            filename : str
                path to output file

        Returns:
            filename : str
                name of the file data was saved to

        """

        log.debug('exporting mesh to openCARP: {}'.format(filename))

        writer = FileWriter()

        writer.dump(filename + '.pts', self.X)
        writer.dump(filename + '.elem', self.tris)

        return filename

    def dump_signal_map(self, basename, which=None):
        """
        Export interpolated surface parameter maps in DAT format.

        Naming convention:
            <basename>.map.<parameter>.dat

        Parameters:
            basename : str
            which : list of str
                names of the maps to export, export all if None

        Returns:
            str : information if export was successful or not

        """

        status = 'Exporting signal maps for surface\n'

        if len(self.signalMaps) == 0:
            status += 'no map data found, nothing to export!\n'
            return status

        if not which:
            which = [m.name for m in self.signalMaps]

        for n in which:
            try:
                m = [x for x in self.signalMaps if x.name == n][0]
                f = m.export(basename=basename)
                status += 'successfully exported {} to {}\n'.format(m.name, f)
            except IndexError:
                status += 'signal map {} not found in data!\n'.format(n)

        return status

    def get_closest_vertex(self, points, limit_to_triangulation=False):
        """
        Find the closest vertex to given point(s).

        Parameters:
            points : list of tuple
                cartesian coordinates of points
            limit_to_triangulation : boolean (optional)
                Not implemented yet

        Returns:
            ndarray : coordinates of closest vertex
            ndarray : Euclidean distance

        """

        # TODO: handle limitation to triangulation
        if limit_to_triangulation:
            pass

        vertices = np.full(len(points), np.iinfo(np.int32).min, dtype=np.int32)
        distances = np.full(len(points), np.nan, dtype=np.float32)

        for i, point in enumerate(points):
            test = self.X - point
            test = np.sum(test**2, axis=1)
            idx_min = np.argmin(test)

            vertices[i] = idx_min
            distances[i] = np.sqrt(test[idx_min])

        return self.X[vertices, :], distances, self.XNormals[vertices, :]

    def get_center_of_mass(self):
        """
        Get the center of mass of the mesh.

        Returns:
            ndarray : Cartesian coordinates of center of mass

        """

        if not self.has_points():
            return [0.0, 0.0, 0.0]

        return np.sum(self.X, axis=0) / self.X.shape[0]

    def get_free_boundary(self):
        """
        Find free boundary vertices, i.e. vertices at the edge.

        Returns:
            ndarray : index of vertices at edge

        """

        # first find the edges which make up the triangles
        l0 = np.sort(self.tris[:, [0, 1]], axis=1)
        l1 = np.sort(self.tris[:, [0, 2]], axis=1)
        l2 = np.sort(self.tris[:, [1, 2]], axis=1)
        edges = np.concatenate((l0, l1, l2))

        # edges at the boundary are only listed once
        _, idx, counts = np.unique(edges,
                                   axis=0,
                                   return_index=True,
                                   return_counts=True)
        boundary_idx = idx[np.where(counts == 1)]

        # get the vertices which made up the unique boundary edges
        v_free = edges[boundary_idx].flatten()
        # vertices are listed twice, remove duplicates
        _, idx = np.unique(v_free, return_index=True)

        return np.delete(v_free, idx)

    def remove_vertices(self, index):
        """
        Remove vertices from mesh. Triangulation, vertex normals, signal
        maps and labels are automatically updated.

        Parameters:
            index: ndarray of int

        Returns:
            None

        """

        # renumber triangulation first
        self.tris = self._renumber_tris(index)
        # now we can remove
        self.X = np.delete(self.X, index, axis=0)
        if self.XNormals.shape[0] > 0:
            self.XNormals = np.delete(self.XNormals, index, axis=0)
        for signal_map in self.signalMaps:
            if signal_map.location == 'pointData':
                signal_map.values = np.delete(signal_map.values,
                                              index,
                                              axis=0)
        for label in self.labels:
            if label.location == 'pointData':
                label.values = np.delete(label.values,
                                         index,
                                         axis=0)

    def remove_tris(self, index):
        """
        Remove elements from triangulation. Triangulation, vertex normals,
        signal maps and labels are automatically updated.

        Parameters:
            index: ndarray of int

        Returns:
            None

        """

        self.tris = np.delete(self.tris, index, axis=0)
        if self.trisNormals.shape[0] > 0:
            self.trisNormals = np.delete(self.trisNormals, index, axis=0)
        for signal_map in self.signalMaps:
            if signal_map.location == 'cellData':
                signal_map.values = np.delete(signal_map.values,
                                              index,
                                              axis=0)
        for label in self.labels:
            if label.location == 'cellData':
                label.values = np.delete(label.values,
                                         index,
                                         axis=0)

    def _renumber_tris(self, vertices_to_remove):
        """
        Renumber triangulation before removing vertices from a mesh.

        Parameters:
            vertices_to_remove : ndarray
                Vertex ID(s) to be removed from mesh

        Raises:
            ValueError : If index of removed vertex exceeds tris numbering

        Returns:
            tris_corrected : ndarray (n_tris, 3)
                triangulation with adjusted numbering
        """

        tris_shape = self.tris.shape  # save shape for later
        # work out unique vertices IDs
        tris = self.tris.flatten()
        tris_sort_idx = tris.argsort()  # remember sorting index for undo
        tris_unique, unique_idx = np.unique(tris[tris_sort_idx],
                                            return_inverse=True)
        # check if remove is possible
        if vertices_to_remove.max() > tris_unique.shape[0]:
            raise ValueError('cannot remove vertex: index exceeds number of '
                             'vertices!')
        # build offset vector
        tris_offset = np.full(tris_unique.shape[0], 0, dtype=int)
        for vert in vertices_to_remove:
            tris_offset[vert:] = tris_offset[vert:] + 1

        # shift vertex indices by number of deleted vertices
        tris_unique = tris_unique - tris_offset

        # restore original shape of triangulation
        tris_flat_restore = tris_unique[unique_idx]  # undo unique
        tris_corrected = np.empty_like(tris)
        tris_corrected[tris_sort_idx] = tris_flat_restore  # undo sort
        tris_corrected = np.reshape(tris_corrected, tris_shape)  # reshape

        return tris_corrected


class SurfaceSignalMap:
    """A class representing a signal property map.

    Attributes:
        name : string
        values : ndarray, dim (n, 1)
            explicit second dimension must be given for VTK export
            compatibility
        location : string
            location of data on the mesh
            Options: 'pointData', 'cellData'
        description : string
            additional information on data

    Methods:
        to_vtk_dict()
            convert class to VTK compatible representation
        export(basename)
            export surface map data to DAT format

    """

    def __init__(self, name, values, location,
                 description=None):
        """
        Constructor.

        Parameters:
            name : str
                name/description of this map
            values : ndarray (n_verts/n_tris, 1)
            location : str
                location of data, either 'pointData' or 'cellData'

        """

        self.name = name
        # adjust values dims if necessary
        try:
            values.shape[1]
        except IndexError:
            np.expand_dims(values, axis=1)
        self.values = values
        self.location = location
        self.description = description

    def to_vtk_dict(self):
        """Convert class to VTK compatible representation."""

        return {'name': self.name,
                'values': self.values,
                'location': self.location
                }

    def export(self, basename):
        """
        Export surface map data to DAT file.

        Naming Convention:
            <basename>.map.<surface_map_name>.dat

        Parameters:
            basename : str
                path to export file

        Returns:
            filename : str
        """

        if basename.endswith('.dat'):
            basename = basename[:-4]

        filename = basename + '.map.' + self.name + '.dat'

        writer = FileWriter()
        writer.dump(filename, self.values[:, 0])

        return filename


class SurfaceLabel:
    """A class representing a signal property map.

    Attributes:
        name : string
        values : ndarray, dim (n, 1)
            explicit second dimension must be given for VTK export
            compatibility
        location : string
            location of data on the mesh
            Options: 'pointData', 'cellData'
        description : string
            additional information on data

    Methods:
        to_vtk_dict()
            convert class to VTK compatible representation
        export(basename)
            export surface map data to DAT format

    """

    def __init__(self, name, values, location, description=''):
        """
        Constructor.

        Parameters:
            name : str
                name/description of this surface label
            values : ndarray (n_verts/n_tris, 1)
            location : str
                location of data, either 'pointData' or 'cellData'

        """

        self.name = name
        self.values = values
        self.location = location
        self.description = description

    def to_vtk_dict(self):
        """Convert class to VTK compatible representation."""

        return {'name': self.name,
                'values': self.values,
                'location': self.location
                }

    def export(self, basename):
        """
        Export surface map data to DAT file.

        Naming Convention:
            <basename>.<surface_label_name>.dat

        Parameters:
            basename : str
                path to export file

        Returns:
            filename : str
        """

        if basename.endswith('.dat'):
            basename = basename[:-4]

        filename = basename + '.' + self.name + '.dat'

        writer = FileWriter()
        writer.dump(filename, self.values[:, 0])

        return filename


Mesh = namedtuple('Mesh', ['registrationMatrix', 'fileNames'])
Mesh.__doc__ = """
Additional meshes.

Attributes:
    registrationMatrix : ndarray (4, 4)
        transformation matrix (scale, rotate, shift)
    filenames: list of str
        files containing triangulation data
"""
