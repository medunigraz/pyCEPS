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

from typing import Optional, Union
import numpy as np
from scipy.spatial import cKDTree


def inverse_distance_weighting(
        sampling_points: np.ndarray,
        sampling_vals: np.ndarray,
        interp_points: np.ndarray,
        kdtree: Optional[cKDTree] = None,
        p: int = 2,
        k: int = 1000,
        return_weights: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Interpolate data by inverse distance weighting algorithm.
    See: https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Parameters:
        sampling_points : np.ndarray
            Coords of the sampled points
        sampling_vals : np.ndarray
            Values at the sampled points
        interp_points : np.ndarray
            Coords of points to be calculated
        kdtree : cKDTree (optional)
            KD-Tree to be used (useful in repeated evaluations), default: None
        p : int (optional)
            order of the distance function (w = 1/d^p), default: 2
        k : int (optional)
            Number of neighbors to consider (> 1), default: 1000
        return_weights : bool (optional)
            If true, also returns the interpolation weights, default: False

    Returns:
        np.ndarray
            Inverse distance weights points, sampled at interp_points

    """
    assert (k > 1)

    k = np.minimum(k, sampling_points.shape[0])

    if kdtree is None:
        kdtree = cKDTree(sampling_points)

    dist, inds = kdtree.query(interp_points, k=k)
    with np.errstate(divide='ignore', invalid='ignore'):
        zero_dist = np.isclose(dist, 0.)
        assert (np.all(np.sum(zero_dist, axis=-1) <= 1),
                'You have multiple sampling points on top of each other'
                )
        weights = 1./(dist**p)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        weights = np.where(np.any(zero_dist, axis=-1, keepdims=True),
                           zero_dist.astype(sampling_vals.dtype),
                           weights
                           )

    interp_vals = np.sum(weights * sampling_vals[inds], axis=-1)

    if not interp_vals.shape:
        # single point only, make array anyway for downstream compatibility
        interp_vals = np.expand_dims(interp_vals, 0)

    if return_weights:
        return interp_vals, weights

    return interp_vals


def remove_redundant_points(
        points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove points of the point cloud that lie on top of each other.

    Parameters:
        points : ndarray (N, 1)

    Returns:
        tuple[np.ndarray, np.ndarray]
            Tuple containing both the non-redundant points and their indices

    """
    kdtree = cKDTree(points)
    dists = kdtree.query(points, k=2)[0]
    mask = np.isclose(dists[..., 1], 0)
    inds = np.arange(points.shape[0])
    while np.any(mask):
        points = np.delete(points, np.where(mask)[0][0], axis=0)
        inds = np.delete(inds, np.where(mask)[0][0], axis=0)
        kdtree = cKDTree(points)
        dists = kdtree.query(points, k=2)[0]
        mask = np.isclose(dists[..., 1], 0)

    return points, inds

# def interp_ueg(carto_map : CartoMap, k=10, p=2) -> np.ndarray:
#     """Interpolates the UEGs in a Carto Map on the given mesh using :ref:`inverse_distance_weighting`.
#
#     Parameters
#     ----------
#     carto_map : CartoMap
#         The given CARTO map for which to interpolate
#     k : int, optional
#         Parameter of the interpolation see :ref:`inverse_distance_weighting`, by default 10
#     p : int, optional
#         Parameter of the interpolation see :ref:`inverse_distance_weighting`, by default 2
#
#     Returns
#     -------
#     np.ndarray
#         A [N,T] np.ndarray with the UEG interpolated for all mesh points N over all time steps T
#     """
#     mesh_points, mesh_tris = carto_map.surface.triRep
#     valid_points = [p for p in carto_map.points if p.lat_in_woi()]
#     points_surf = np.stack([p.egmSurfX for p in valid_points]) #Projected points
#     ueg = np.stack([p.egmUni for p in valid_points])[..., 0]
#     ref_annotations = np.array([p.refAnnotation for p in valid_points])
#     wois = np.stack([p.woi for p in valid_points])
#     valid_windows = wois + ref_annotations[:, np.newaxis]
#     time_range = np.arange(valid_windows[:, 0].min(), valid_windows[:, 1].max())
#
#     ueg_unique_points, unique_inds = remove_redundant_points(points_surf)
#     ueg_all_interp = np.zeros(shape=[mesh_points.shape[0], time_range.size], dtype=np.float32)
#     for ti, t in enumerate(time_range):
#         valid_point_mask = ((valid_windows[:, 0] <= t) & (valid_windows[:, 1] >= t))[unique_inds] #Current time inside valid time
#         unique_points_valid = ueg_unique_points[valid_point_mask]
#         ueg_interp = inverse_distance_weighting(unique_points_valid, ueg[unique_inds, t][valid_point_mask], mesh_points, kdtree=None, k=k, p=p).astype(np.float32)
#
#         ueg_all_interp[..., ti] = ueg_interp
#
#     return ueg_all_interp
