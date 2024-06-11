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


import numpy as np
from dash import Dash, html, no_update, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from .dashlayout import get_layout
from .dashutils import (to_drop_option, colormaps,
                        find_closest_point,
                        empty_figure, get_bsecg_figure,
                        get_point_egm_figure, get_point_ecg_figure,
                        get_colorbar,
                        )


# VTK_BGND_COLOR = 'rgb(51, 76, 102)'
VTK_BGND_COLOR = 'rgb(255, 255, 255)'
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
          '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
DEFAULT_SURFACE_COLORMAP = 'Warm to Cool (Extended)'
DEFAULT_POINTS_COLORMAP = 'Blue to Red Rainbow'
DEFAULT_LESION_COLORMAP = 'Plasma (matplotlib)'


def get_dash_app(study, bgnd=None):
    """Build dash app."""

    # get standard background color for VTK view if not specified
    if not bgnd:
        bgnd = VTK_BGND_COLOR

    app = Dash(__name__,
               external_stylesheets=[dbc.themes.BOOTSTRAP],
               # suppress_callback_exceptions=True
               )
    app.layout = get_layout(bgnd,
                            surf_colormap_name=DEFAULT_SURFACE_COLORMAP,
                            point_colormap_name=DEFAULT_POINTS_COLORMAP,
                            lesion_colormap_name=DEFAULT_LESION_COLORMAP)

    """Initialize Page."""
    @app.callback(
        Output('study-name', 'children'),
        Output('map-selector', 'disabled'),
        Output('map-selector', 'options'),
        Output('map-selector', 'value'),
        Input('study-name', 'children')
    )
    def init_page(title):
        title = 'Study: {} ({})'.format(study.name, study.system.upper())

        map_selection = [{'label': name,
                          'value': name,
                          'disabled': name not in study.maps}
                         for name in study.mapNames
                         if name in study.maps
                         and study.maps[name].surface.has_points()
                         ]

        return (title,          # title string
                False,          # map selector enabled
                map_selection,  # map selector items
                None,           # map selector value to trigger events
                )

    """Handle Controls."""
    @app.callback(
        Output('surface-poly', 'points'),
        Output('surface-poly', 'polys'),
        Output('mapping-points-poly', 'points', allow_duplicate=True),
        Output('mapping-points-selected', 'values', allow_duplicate=True),
        Output('mapping-points-id', 'values', allow_duplicate=True),
        Output('lesions-poly', 'points'),
        Output('vtk-view', 'cameraPosition'),
        Output('rep-selector', 'disabled'),
        Input('map-selector', 'value'),
        prevent_initial_call=True,
    )
    def update_vtk_view(map_name):
        if not study or not map_name:
            return ([],  # surface mesh points
                    [],  # surface mesh polys
                    [],  # recording points location
                    [],  # selected mapping points
                    [],  # mapping points ID
                    [],  # lesion points location
                    no_update,  # camera position (center of gravity)
                    True,  # surface representation type disabled
                    )

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        # build mesh points and polys
        mesh_points = p_map.surface.X.ravel().tolist()
        verts = p_map.surface.tris
        face_points = np.full(shape=(len(verts)),
                              fill_value=3,
                              dtype=int)  # all faces have 3 vertices
        faces = np.concatenate([face_points[:, np.newaxis], verts], axis=1)
        faces = faces.ravel().tolist()

        # recording points, default "surface" and "valid"
        points = [p for p in p_map.points if p.is_valid()]
        point_location = np.asarray([p.prjX for p in points]).ravel().tolist()
        point_names = [int(p.name.split('P')[1]) for p in points]
        point_selected = np.full(len(points), 0, dtype=int)

        # lesions
        lesion_location = np.asarray(
            [site.X for site in p_map.lesions.sites]
        ).ravel().tolist()

        # calc center of mass to set camera position right
        cg = p_map.surface.get_center_of_mass()

        return (mesh_points,     # surface mesh points
                faces,           # surface mesh polys
                point_location,  # recording points (projected and valid)
                point_selected,  # selected mapping points
                point_names,     # mapping points ID
                lesion_location,  # lesion points location
                cg,              # camera position (center of gravity)
                False,           # surface representation type enabled
                )

    @app.callback(
        Output('point-egm-figure', 'figure'),
        Output('point-ecg', 'value', allow_duplicate=True),
        Input('mapping-points-selected', 'values'),
        State('map-selector', 'value'),
        State('mapping-points-id', 'values'),
        prevent_initial_call=True,
    )
    def update_point_egm_figure(selection, map_name, point_id):
        # update selection array
        point_index = np.argwhere(selection)

        if not len(point_index) > 0:
            return (
                empty_figure(bgnd='rgb(255, 255, 255)'),  # empty EGm figure
                False  # set ECG to "Map"
            )

        point_index = point_index.flatten()[0]

        # update EGM figure
        p_name = 'P{}'.format(point_id[point_index])
        p_map = [m for m in study.maps.values() if m.name == map_name][0]
        point = [p for p in p_map.points if p.name == p_name][0]

        return (
            get_point_egm_figure(point),
            no_update
        )

    @app.callback(
        Output('surf-ecg-figure', 'figure'),
        Input('map-selector', 'value'),
        Input('mapping-points-selected', 'values'),
        Input('point-ecg', 'value'),
        State('mapping-points-id', 'values'),
        prevent_initial_call=True,
    )
    def update_point_ecg_figure(map_name, selection, point_ecg,
                                point_id):

        if not map_name:
            return empty_figure(bgnd='rgb(255, 255, 255)')

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        # find which input triggered callback
        button_clicked = ctx.triggered_id
        if button_clicked == 'map-selector':
            # new map selected or BSECG requested
            return get_bsecg_figure(p_map.bsecg, COLORS)
        elif button_clicked == 'point-ecg' and not point_ecg:
            # ECG view changed to Map ECG
            return get_bsecg_figure(p_map.bsecg, COLORS)
        elif button_clicked == 'mapping-points-selected' and not any(selection):
            # this happens on new map selected, don't know why...?
            return get_bsecg_figure(p_map.bsecg, COLORS)
        elif not point_ecg:
            # new point selected but map ECG is set, no update needed
            return no_update
        else:
            # new point selected and point ECG requested
            point_index = np.argwhere(selection)
            if not len(point_index) > 0:
                return no_update

            point_index = point_index.flatten()[0]

            # update EGM figure
            p_name = 'P{}'.format(point_id[point_index])
            p_map = [m for m in study.maps.values() if m.name == map_name][0]
            point = [p for p in p_map.points if p.name == p_name][0]

            return get_point_ecg_figure(point, COLORS)

    @app.callback(
        Output('map-info-button', 'disabled'),
        Output('surface-color-by', 'disabled'),
        Output('surface-color-by', 'options'),
        Output('surface-opacity', 'disabled'),
        Output('mapping-points-visible', 'value'),
        Output('mapping-points-visible', 'disabled'),
        Output('mapping-points-location', 'value'),
        Output('mapping-points-location', 'disabled'),
        Output('mapping-points-type', 'value'),
        Output('mapping-points-type', 'disabled'),
        Output('lesions-visible', 'value'),
        Output('lesions-visible', 'disabled'),
        Output('lesions-color-by', 'options'),
        Output('lesions-color-by', 'disabled'),
        Output('surface-color-by', 'value'),
        Output('lesions-color-by', 'value'),
        Output('point-ecg', 'disabled'),
        Output('point-ecg', 'value'),
        Input('map-selector', 'value'),
        prevent_initial_call=True,
    )
    def on_map_selected(map_name):
        if not study or not map_name:
            return (True,   # map info button disabled
                    True,   # surface parameter map selection disabled
                    {},     # surface parameter map selection
                    True,   # surface opacity disabled
                    False,  # mapping points visible selector value
                    True,   # mapping points visible selector disabled
                    False,  # mapping points location selector value
                    True,   # mapping points location selector disabled
                    False,  # mapping points type selector value
                    True,   # mapping points type selector disabled
                    False,  # lesions visible selector value
                    True,   # lesions visible selector disabled
                    {},     # lesion RFI options
                    True,   # lesion RFI selector disabled
                    None,   # trigger downstream surface map events
                    None,   # trigger downstream lesion events
                    True,   # map/point ecg selector disabled
                    False,  # map/point ecg selector value (Map)
                    )

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        # get available surface maps
        surf_maps = [to_drop_option(name)
                     for name in p_map.surface.get_map_names()
                     ]
        has_maps = len(surf_maps) > 0
        has_points = len(p_map.points) > 0
        has_lesions = len(p_map.lesions.sites) > 0
        has_ecg = any([len(p.ecg) > 0 for p in p_map.points])

        # get available RF indexes
        rfi_names = [to_drop_option(name)
                     for name in p_map.lesions.get_rfi_names()]
        has_rfi = len(rfi_names) > 0

        return (False,  # map info button enabled
                not has_maps,       # surface map selection disabled
                surf_maps,          # surface parameter map selection
                False,              # surface opacity enabled
                False,              # mapping points visible selector value
                not has_points,     # mapping points visible selector enabled
                False,              # mapping points location selector value
                not has_points,     # mapping points location selector enabled
                False,              # mapping points type selector value
                not has_points,     # mapping points type selector disabled
                False,              # lesions visible selector value
                not has_lesions,    # lesions visible selector enabled
                rfi_names,          # lesion RFI options
                not has_rfi,        # lesion RFI selector disabled
                None,               # trigger downstream surface map events
                None,               # trigger downstream lesion events
                not has_ecg,        # map/point ecg selector disabled
                False,              # map/point ecg selector value (Map)
                )

    @app.callback(
        Output('surface-point-data', 'values'),
        Output('surface-cell-data', 'values'),
        Output('mesh-rep', 'mapper'),
        Output('surface-colormap', 'disabled'),
        Output('surface-color-range', 'disabled'),
        Output('surface-color-range', 'min', allow_duplicate=True),
        Output('surface-color-range', 'max', allow_duplicate=True),
        Output('surface-color-range', 'value', allow_duplicate=True),
        Output('surface-range-min', 'disabled'),
        Output('surface-range-min', 'value'),
        Output('surface-range-min', 'min'),
        Output('surface-range-min', 'max'),
        Output('surface-range-max', 'disabled'),
        Output('surface-range-max', 'value'),
        Output('surface-range-max', 'min'),
        Output('surface-range-max', 'max'),
        Output('surface-map-range', 'data'),
        Output('surface-colormap', 'value'),
        Input('surface-color-by', 'value'),
        State('map-selector', 'value'),
        State('mesh-rep', 'mapper'),
        State('surface-colormap', 'value'),
        prevent_initial_call=True,
    )
    def on_surface_map_changed(surf_map_name, map_name, mapper, cmap_name):
        if not surf_map_name or not map_name:
            mapper['colorByArrayName'] = None
            mapper['scalarMode'] = 0  # DEFAULT

            return ([],         # point data
                    [],         # cell data
                    mapper,     # mesh representation mapper
                    True,       # surface colormap selector disabled
                    True,       # color range slider disabled
                    0,          # color range slider min
                    100,        # color range slider max
                    [0, 100],   # color range slider value
                    True,       # color range input min disabled
                    None,       # color range input min value
                    0,          # color range input min lowest value
                    100,        # color range input min highest value
                    True,       # color range input max disabled
                    None,       # color range input max value
                    0,          # color range input max lowest value
                    100,        # color range input max highest value
                    [0, 100],   # DCC Store surface map range
                    cmap_name,  # trigger surface colormap update
                    )

        # get map
        p_map = [m for m in study.maps.values() if m.name == map_name][0]
        # get surface map values
        surf_data = p_map.surface.get_map(surf_map_name)
        values = surf_data.values.ravel().tolist()
        # get data range as integers for UI objects
        rng = [np.floor(min(values)).astype(int),
               np.ceil(max(values)).astype(int)
               ]

        # set VTK mapper to correct mode
        if surf_data.location == 'pointData':
            mapper['colorByArrayName'] = 'pointData'
            mapper['scalarMode'] = 3  # USE_POINT_FIELD_DATA
        elif surf_data.location == 'cellData':
            mapper['colorByArrayName'] = 'cellData'
            mapper['scalarMode'] = 4  # USE_CELL_FIELD_DATA
        else:
            raise KeyError('unknown location of surface data: {}'
                           .format(surf_data.location))

        return (values if surf_data.location == 'pointData' else [],
                values if surf_data.location == 'cellData' else [],
                mapper,   # mesh representation mapper
                False,    # surface colormap selector enabled
                False,    # color range slider disabled
                rng[0],   # color range slider min
                rng[1],   # color range slider max
                rng,      # color range slider value
                False,    # color range input min disabled
                rng[0],   # color range input min
                rng[0],   # color range input min lowest value
                rng[1],   # color range input min highest value
                False,    # color range input max disabled
                rng[1],   # color range input max
                rng[0],   # color range input max lowest value
                rng[1],   # color range input max highest value
                rng,      # DCC Store map range
                cmap_name,    # trigger surface colormap update
                )

    @app.callback(
        Output('mesh-rep', 'property', allow_duplicate=True),
        Input('rep-selector', 'value'),
        State('mesh-rep', 'property'),
        prevent_initial_call=True,
    )
    def on_surface_representation_changed(mesh_type, mesh_rep_properties):
        if not mesh_rep_properties:
            return no_update

        mesh_rep_properties['representation'] = mesh_type

        return mesh_rep_properties

    @app.callback(
        Output('mesh-rep', 'colorMapPreset'),
        Output('mesh-rep', 'colorDataRange'),
        Output('surface-colorbar', 'figure'),
        Input('surface-colormap', 'value'),
        Input('surface-color-range', 'value'),
        State('surface-color-by', 'value'),
        prevent_initial_call=True,
    )
    def on_surface_colormap_changed(colormap_name, rng, surface_map_name):
        if not colormap_name or not surface_map_name:
            return no_update, no_update, empty_figure(bgnd=bgnd)

        # update colorbar
        c_map = [cm for cm in colormaps if cm.name == colormap_name][0]

        fig = get_colorbar(c_map.colorscale, surface_map_name, rng, bgnd=bgnd)

        return c_map.name, rng, fig

    @app.callback(
        Output('surface-color-range', 'min', allow_duplicate=True),
        Output('surface-color-range', 'max', allow_duplicate=True),
        Output('surface-color-range', 'value', allow_duplicate=True),
        Input('surface-range-min', 'value'),
        Input('surface-range-max', 'value'),
        State('surface-map-range', 'data'),
        prevent_initial_call=True,
    )
    def on_data_clipping_changed(rng_min, rng_max, rng_default):
        if not rng_min and rng_min != 0:
            rng_min = rng_default[0]
        if not rng_max and rng_max != 0:
            rng_max = rng_default[1]

        return (rng_min,   # color range slider min
                rng_max,   # color range slider max
                [rng_min, rng_max]  # color range slider value
                )

    @app.callback(
        Output('mapping-points-rep', 'actor'),
        Output('point-select', 'disabled'),
        Input('mapping-points-visible', 'value'),
        State('mapping-points-rep', 'actor'),
        prevent_initial_call=True,
    )
    def on_show_mapping_points(visible_checked, points_actor):
        points_actor['visibility'] = visible_checked
        return points_actor, not visible_checked

    @app.callback(
        Output('mapping-points-poly', 'points', allow_duplicate=True),
        Output('point-select', 'options'),
        Input('mapping-points-location', 'value'),
        State('mapping-points-type', 'value'),
        State('map-selector', 'value'),
        prevent_initial_call=True,
    )
    def on_mapping_points_location_changed(location, invalid, map_name):
        if not map_name:
            return [], {}  # mapping points location

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        # recording points
        if invalid:
            points = [p for p in p_map.points if not p.is_valid()]
        else:
            points = [p for p in p_map.points if p.is_valid()]

        if location:
            point_location = np.asarray([p.recX for p in points])
        else:
            point_location = np.asarray([p.prjX for p in points])

        point_location = point_location.ravel().tolist()
        point_names = [to_drop_option(p.name)
                       for p in points]

        return point_location, point_names

    @app.callback(
        Output('mapping-points-poly', 'points', allow_duplicate=True),
        Output('mapping-points-selected', 'values', allow_duplicate=True),
        Output('mapping-points-id', 'values', allow_duplicate=True),
        Output('point-select', 'options', allow_duplicate=True),
        Output('point-select', 'value', allow_duplicate=True),
        Input('mapping-points-type', 'value'),
        State('mapping-points-location', 'value'),
        State('map-selector', 'value'),
        prevent_initial_call=True,
    )
    def on_mapping_points_type_changed(invalid, location, map_name):
        if not map_name:
            return ([],    # mapping points location
                    [],    # mapping points selected
                    [],    # mapping points ID
                    {},    # mapping points selection dropdown
                    None,  # mapping points selection dropdown no selection
                    )

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        # recording points
        if invalid:
            points = [p for p in p_map.points if not p.is_valid()]
        else:
            points = [p for p in p_map.points if p.is_valid()]

        if location:
            point_location = np.asarray([p.recX for p in points])
        else:
            point_location = np.asarray([p.prjX for p in points])

        point_location = point_location.ravel().tolist()
        point_names = [int(p.name.split('P')[1]) for p in points]
        point_names_dd = [to_drop_option(p.name) for p in points]
        point_selected = np.full(len(points), 0, dtype=int)

        return (point_location,  # mapping points location
                point_selected,  # mapping points selected
                point_names,     # mapping points ID
                point_names_dd,  # mapping points selection dropdown
                None,  # mapping points selection dropdown no selection
                )

    @app.callback(
        Output('mapping-points-selected', 'values'),
        Output('point-select', 'value'),
        Input('vtk-view', 'clickInfo'),
        State('mapping-points-poly', 'points'),
        State('mapping-points-visible', 'value'),
        State('mapping-points-id', 'values'),
        prevent_initial_call=True,
    )
    def on_mapping_point_clicked(click_data,
                                 points, points_visible, point_ids):
        if not click_data or not points_visible:
            return no_update

        click_coords = click_data['worldPosition']
        # get point coordinates as (n, 3) array for search
        points = np.reshape(points, (int(len(points) / 3), 3))
        # find the closest point to click position
        distance, index = find_closest_point(click_coords, points)

        # check if a point was found
        if distance == float('inf'):
            return no_update

        # update selection array
        selection = np.full(points.shape[0], 0, dtype=int)
        selection[index] = 1

        return selection, 'P{}'.format(point_ids[index])

    @app.callback(
        Output('mapping-points-selected', 'values', allow_duplicate=True),
        Input('point-select', 'value'),
        State('mapping-points-id', 'values'),
        State('mapping-points-visible', 'value'),
        prevent_initial_call=True,
    )
    def on_mapping_point_selected(point_name,
                                  point_ids, points_visible):
        if not point_name or not points_visible:
            return no_update

        # update selection array
        p_id = int(point_name[1:])  # remove "P" from point name
        point_index = np.argwhere(np.array(point_ids) == p_id)

        if not point_index.size == 1:
            return no_update

        point_index = point_index.flatten()[0]

        # update selection array
        selection = np.full(len(point_ids), 0, dtype=int)
        selection[point_index] = 1

        return selection

    @app.callback(
        Output('mesh-rep', 'property', allow_duplicate=True),
        Input('surface-opacity', 'value'),
        State('mesh-rep', 'property'),
        prevent_initial_call=True,
    )
    def on_surface_opacity_changed(opacity, mesh_rep_properties):
        if not mesh_rep_properties:
            return no_update

        mesh_rep_properties['opacity'] = opacity

        return mesh_rep_properties

    @app.callback(
        Output('lesions-rep', 'actor'),
        Output('lesions-rep', 'mapper', allow_duplicate=True),
        Input('lesions-visible', 'value'),
        State('lesions-rep', 'actor'),
        State('lesions-rep', 'mapper'),
        State('lesions-color-by', 'value'),
        prevent_initial_call=True,
    )
    def on_show_lesions(visible_checked, lesions_actor, lesion_mapper,
                        rfi_name):
        # update representation actor
        lesions_actor['visibility'] = visible_checked

        # update representation mapper
        if not rfi_name:
            lesion_mapper['colorByArrayName'] = None
            lesion_mapper['scalarMode'] = 0  # DEFAULT
        else:
            lesion_mapper['colorByArrayName'] = 'rfi'
            lesion_mapper['scalarMode'] = 3  # USE_POINT_FIELD_DATA

        return lesions_actor, lesion_mapper

    @app.callback(
        Output('lesions-rfi', 'values'),
        Output('lesions-rep', 'colorDataRange'),
        Output('lesion-glyphs', 'state'),
        Output('lesions-rep', 'mapper'),
        Output('lesions-color-range', 'disabled'),
        Output('lesions-color-range', 'min', allow_duplicate=True),
        Output('lesions-color-range', 'max', allow_duplicate=True),
        Output('lesions-color-range', 'value', allow_duplicate=True),
        Output('lesion-range-min', 'disabled'),
        Output('lesion-range-min', 'value'),
        Output('lesion-range-min', 'min'),
        Output('lesion-range-min', 'max'),
        Output('lesion-range-max', 'disabled'),
        Output('lesion-range-max', 'value'),
        Output('lesion-range-max', 'min'),
        Output('lesion-range-max', 'max'),
        Output('lesion-range', 'data'),
        Input('lesions-color-by', 'value'),
        State('lesion-glyphs', 'state'),
        State('lesions-rep', 'mapper'),
        State('map-selector', 'value'),
        prevent_initial_call=True,
    )
    def on_lesion_parameter_changed(rfi_name, lesion_glyphs_state,
                                    lesion_mapper, map_name):
        if not rfi_name or not map_name:
            lesion_mapper['colorByArrayName'] = None
            lesion_mapper['scalarMode'] = 0  # DEFAULT

            return ([],                   # lesions RFI value
                    [0, 1],               # lesions RFI color range
                    lesion_glyphs_state,  # glyph algorithm state
                    lesion_mapper,        # lesion representation mapper
                    True,                 # lesion color range slider disabled
                    0,                    # lesion color range slider min
                    100,                  # lesion color range slider max
                    None,                 # lesion color range slider value
                    True,                 # lesion clipping min disabled
                    None,                 # lesion clipping min value
                    0,                    # lesion clipping min lowest
                    100,                  # lesion clipping min highest
                    True,                 # lesion clipping max disabled
                    None,                 # lesion clipping max value
                    0,                    # lesion clipping max lowest
                    100,                  # lesion clipping max highest
                    [0, 1]                # dcc Store lesion range
                    )

        p_map = [m for m in study.maps.values() if m.name == map_name][0]

        rfi = [rfi.value if rfi.name == rfi_name else np.nan
               for site in p_map.lesions.sites
               for rfi in site.RFIndex
               ]

        # get data range as integers for UI objects
        rng = [np.floor(np.nanmin(rfi)).astype(int),
               np.ceil(np.nanmax(rfi)).astype(int)
               ]

        diameter = np.median([site.diameter for site in p_map.lesions.sites])

        # update glyph representation
        lesion_glyphs_state['radius'] = diameter / 2
        lesion_mapper['colorByArrayName'] = 'rfi'
        lesion_mapper['scalarMode'] = 3  # USE_POINT_FIELD_DATA

        return (rfi,                    # lesions RFI values
                rng,                    # lesions RFI color range
                lesion_glyphs_state,    # glyph algorithm state
                lesion_mapper,          # lesion representation mapper
                False,                  # lesion color range slider disabled
                rng[0],                 # lesion color range slider min
                rng[1],                 # lesion color range slider max
                None,                   # lesion color range slider value
                False,                  # lesion clipping min disabled
                None,                   # lesion clipping min value
                rng[0],                 # lesion clipping min lowest
                rng[1],                 # lesion clipping min highest
                False,                  # lesion clipping max disabled
                None,                   # lesion clipping max value
                rng[0],                 # lesion clipping max lowest
                rng[1],                 # lesion clipping max highest
                rng                     # dcc Store lesion range
                )

    @app.callback(
        Output('lesions-color-range', 'min', allow_duplicate=True),
        Output('lesions-color-range', 'max', allow_duplicate=True),
        Output('lesions-color-range', 'value', allow_duplicate=True),
        Input('lesion-range-min', 'value'),
        Input('lesion-range-max', 'value'),
        State('lesion-range', 'data'),
        prevent_initial_call=True,
    )
    def on_lesion_clipping_changed(rng_min, rng_max, rng_default):
        if not rng_min and rng_min != 0:
            rng_min = rng_default[0]
        if not rng_max and rng_max != 0:
            rng_max = rng_default[1]

        return (rng_min,  # color range slider min
                rng_max,  # color range slider max
                [rng_min, rng_max]  # color range slider value
                )

    @app.callback(
        Output('lesions-rep', 'colorDataRange', allow_duplicate=True),
        Output('lesion-colorbar', 'figure'),
        Input('lesions-color-range', 'value'),
        State('lesions-color-by', 'value'),
        prevent_initial_call=True,
    )
    def on_lesion_colormap_changed(rng, rfi_name):
        if not rng or not rfi_name:
            return no_update, empty_figure(bgnd=bgnd)

        # update colorbar
        c_map = [cm for cm in colormaps
                 if cm.name == DEFAULT_LESION_COLORMAP][0]

        fig = get_colorbar(c_map.colorscale, rfi_name, rng, bgnd=bgnd)

        return rng, fig

    @app.callback(
        Output('info-canvas', 'is_open'),
        Output('map-info-title', 'children'),
        Output('map-info-body', 'children'),
        Output('point-info-title', 'children'),
        Output('point-info-body', 'children'),
        Input('map-info-button', 'n_clicks'),
        State('map-selector', 'value'),
        State('mapping-points-selected', 'values'),
        State('mapping-points-id', 'values'),
        prevent_initial_call=True,
    )
    def on_map_info(n_clicks, map_name, selection, point_id):
        if not n_clicks:
            return no_update

        # default text
        map_title = ['Map Info']
        map_info = ['No map selected yet...']
        point_title = ['Point Info']
        point_info = ['No point selected yet...']

        if map_name:
            ep_map = study.maps[map_name]
            map_title = [map_name]
            map_info = [
                'Valid points: {}'.format(
                    len([p for p in ep_map.points if p.is_valid()])
                ),
                html.Br(),
                'Invalid points: {}'.format(
                    len([p for p in ep_map.points if not p.is_valid])
                ),
            ]

            # get info for currently selected point
            point_index = np.argwhere(selection)
            if len(point_index) > 0:
                point_index = np.argwhere(selection)
                point_index = point_index.flatten()[0]

                # update point info
                p_name = 'P{}'.format(point_id[point_index])
                point = [p for p in ep_map.points if p.name == p_name][0]

                point_title = [point.name]
                point_info = [
                    'Distance to surface: {:.3f} mm'.format(point.prjDistance),
                    html.Br(),
                    # TODO: force should be np.nan, but unpickling turns it
                    #  to None. Works for impedance but not here, why?
                    'Force: {:.2f} [g?]'.format(point.force if point.force
                                                else np.nan),
                    html.Br(),
                    'BIP voltage: {} mV'.format(point.bipVoltage),
                    html.Br(),
                    'UNI voltage: {} mV'.format(point.uniVoltage),
                    html.Br(),
                    'Impedance: {} [?]'.format(point.impedance),
                ]

        return (True,         # info off canvas open
                map_title,    # map info title
                map_info,     # map info text
                point_title,  # point info title
                point_info    # point info text
                )

    # @app.callback(
    #     Output('tooltip', 'children'),
    #     Input('vtk-view', 'hoverInfo'),
    #     prevent_initial_call=True,
    # )
    # def display_hover_info(hover_data):
    #     if not hover_data:
    #         return []
    #
    #     return [json.dumps(hover_data, indent=2)]

    return app
