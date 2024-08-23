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

from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_vtk

from .dashutils import (to_drop_option, colormaps, rgb_norm,
                        )


def get_vtk_view(bgnd, surf_colormap_name, point_colormap_name,
                 lesion_colormap_name):
    """
    Anatomical Shell Visualization
    """
    return dash_vtk.View(
        id="vtk-view",
        background=rgb_norm(bgnd),
        pickingModes=[
            "click",
            "hover"
        ],
        children=[
            dash_vtk.GeometryRepresentation(
                id="mesh-rep",
                children=[
                    dash_vtk.PolyData(
                        id='surface-poly',
                        points=[],
                        polys=[],
                        children=[
                            dash_vtk.PointData(
                                children=[
                                    dash_vtk.DataArray(
                                        id='surface-point-data',
                                        registration='addArray',
                                        name='pointData',
                                        values=[]
                                    ),
                                ],
                            ),
                            dash_vtk.CellData(
                                children=[
                                    dash_vtk.DataArray(
                                        id='surface-cell-data',
                                        registration='addArray',
                                        name='cellData',
                                        values=[]
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                actor={
                    'visibility': True,
                    'origin': (0, 0, 0),
                    'position': (0, 0, 0),
                },
                property={
                    'opacity': 1,
                    'representation': 2,
                },
                mapper={
                    'scalarMode': 3,  # USE_POINT_FIELD_DATA
                    'arrayAccessMode': 1,  # BY_NAME
                    'colorByArrayName': None,
                },
                colorMapPreset=surf_colormap_name,
                colorDataRange=[2, 10],
            ),

            dash_vtk.GlyphRepresentation(
                id='mapping-points-rep',
                children=[
                    dash_vtk.PolyData(
                        id='mapping-points-poly',
                        port=0,
                        points=[],
                        children=[
                            dash_vtk.PointData(
                                children=[
                                    dash_vtk.DataArray(
                                        id='mapping-points-selected',
                                        registration='addArray',
                                        name='point-selected',
                                        values=[]
                                    ),
                                    dash_vtk.DataArray(
                                        id='mapping-points-id',
                                        registration='addArray',
                                        name='point-id',
                                        values=[]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dash_vtk.Algorithm(
                        port=1,
                        vtkClass='vtkSphereSource',
                        state={
                            'radius': 1,
                            'phiResolution': 10,
                            'thetaResolution': 20,
                        },
                    ),
                ],
                actor={
                    'visibility': False,
                },
                property={
                    'opacity': 1,
                },
                mapper={
                    'scalarMode': 3,  # USE_POINT_FIELD_DATA
                    'arrayAccessMode': 1,  # BY_NAME
                    'colorByArrayName': 'point-selected',
                },
                colorDataRange=[0, 1],
                colorMapPreset=point_colormap_name,  # selected is red (1)
            ),
            dash_vtk.GlyphRepresentation(
                id='lesions-rep',
                children=[
                    dash_vtk.PolyData(
                        id="lesions-poly",
                        port=0,
                        points=[],
                        children=[
                            dash_vtk.PointData(
                                children=[
                                    dash_vtk.DataArray(
                                        id='lesions-rfi',
                                        registration='addArray',
                                        name='rfi',
                                        values=[]
                                    ),
                                ],
                            ),
                        ]
                    ),
                    dash_vtk.Algorithm(
                        id='lesion-glyphs',
                        port=1,
                        vtkClass='vtkSphereSource',
                        state={
                            'radius': 3,
                            'phiResolution': 10,
                            'thetaResolution': 20,
                        },
                    ),
                ],
                actor={
                    'visibility': False,
                },
                property={
                    'opacity': 1,
                },
                mapper={
                    'scalarMode': 3,  # USE_POINT_FIELD_DATA
                    'arrayAccessMode': 1,  # BY_NAME
                    'colorByArrayName': 'rfi',
                },
                colorDataRange=[0, 1],
                colorMapPreset=lesion_colormap_name,
            ),
        ],
    )


def get_controls(surf_colormap_name):
    """
    Controls
    """
    return [
        # html.H4('Controls',
        #         style={'textAlign': 'center'}
        #         ),

        dbc.Card(
            outline=True,
            children=[
                dbc.CardHeader(
                    'Mesh',
                    style={
                        'fontSize': 12,
                        'fontWeight': 'bold',
                    },
                ),
                dbc.CardBody(
                    style={'fontSize': 12},
                    children=[
                        dbc.Row(
                            className='gx-1',
                            align='end',
                            children=[
                                dbc.Col(
                                    width=6,
                                    children=[
                                        dbc.Label('Map:'),
                                        dcc.Dropdown(
                                            id='map-selector',
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    width=4,
                                    children=[
                                        dbc.Label('Type:'),
                                        dcc.Dropdown(
                                            id='rep-selector',
                                            options=[
                                                {'label': 'points', 'value': 0},
                                                {'label': 'wireframe', 'value': 1},
                                                {'label': 'surface', 'value': 2},
                                            ],
                                            value=2,
                                            disabled=True,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    width=2,
                                    children=[
                                        dbc.Button(
                                            'Info',
                                            id='map-info-button',
                                            outline=True,
                                            color='dark',
                                            # size='sm',
                                            disabled=True,
                                            style={
                                                'fontSize': '12px'
                                            }
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        html.Br(),

        # surface coloring
        dbc.Card(
            outline=True,
            children=[
                dbc.CardHeader(
                    'Surface Coloring',
                    style={
                        'fontSize': 12,
                        'fontWeight': 'bold',
                    }
                ),
                dbc.CardBody(
                    style={'fontSize': 12},
                    children=[
                        dbc.Form(
                            style={'fontSize': 12},
                            children=[
                                dbc.Label(
                                    'Parameter',
                                    width=4,
                                    size='sm',
                                    style={
                                        'fontSize': 12
                                    }
                                ),
                                dcc.Dropdown(
                                    id='surface-color-by',
                                    disabled=True,
                                ),
                                dbc.Label(
                                    'Colormap',
                                    width=4,
                                    size='sm',
                                    style={
                                        'fontSize': 12
                                    }
                                ),
                                dcc.Dropdown(
                                    id='surface-colormap',
                                    options=[
                                        to_drop_option(cm.name)
                                        for cm in colormaps
                                    ],
                                    value=surf_colormap_name,
                                    disabled=True,
                                ),
                                dbc.Label(
                                    'Range',
                                    width=4,
                                    size='sm',
                                    style={
                                        'fontSize': 12
                                    }
                                ),
                                dcc.RangeSlider(
                                    id='surface-color-range',
                                    min=0.0, max=100.0,
                                    value=[0, 100],
                                    step=0.5,
                                    marks=None,
                                    allowCross=False,
                                    tooltip={
                                        'placement': 'bottom',
                                        'always_visible': True,
                                    },
                                    disabled=True,
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Label(
                                            'Clipping',
                                            width='auto',
                                            size='sm',
                                            style={
                                                'fontSize': 12
                                            }
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id='surface-range-min',
                                                type='number',
                                                placeholder='min',
                                                min=0,
                                                max=10,
                                                step=1,
                                                size='sm',
                                                disabled=True,
                                            ),
                                            width=4,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id='surface-range-max',
                                                type='number',
                                                placeholder='max',
                                                min=0,
                                                max=10,
                                                step=1,
                                                size='sm',
                                                disabled=True,
                                            ),
                                            width=4,
                                        ),
                                    ],
                                    className='g-2 mt-1'
                                ),
                                dbc.Label(
                                    'Opacity',
                                    width=4,
                                    size='sm',
                                    style={
                                        'fontSize': 12
                                    }
                                ),
                                dcc.Slider(
                                    id='surface-opacity',
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    marks=dict(
                                        [(0.1 * i, {'label': '{:.1f}'
                                          .format(0.1 * i)})
                                         for i in range(0, 11)
                                         ]
                                    ),
                                    value=1.0,
                                    disabled=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        html.Br(),

        # mapping points
        dbc.Card(
            children=[
                dbc.CardHeader(
                    'Mapping Points',
                    style={
                        'fontSize': 12,
                        'fontWeight': 'bold',
                    }
                ),
                dbc.CardBody(
                    style={'fontSize': 12},
                    children=[
                        dbc.Row(
                            justify='center',
                            children=[
                                dbc.Col(dbc.Label('Visible'), width=3),
                                dbc.Col(
                                    dbc.Switch(
                                        id='mapping-points-visible',
                                        # label='Visible?',
                                        value=False,
                                        disabled=True
                                    ),
                                    width=2
                                ),
                                dbc.Col(dbc.Label(''), width=3),
                            ],
                        ),
                        dbc.Row(
                            justify='center',
                            children=[
                                dbc.Col(dbc.Label('Surface'), width=3),
                                dbc.Col(
                                    dbc.Switch(
                                        id='mapping-points-location',
                                        # label='Recording Pos?',
                                        value=False,
                                        disabled=True
                                    ),
                                    width=2
                                ),
                                dbc.Col(dbc.Label('Recording'), width=3),
                            ],
                        ),
                        dbc.Row(
                            justify='center',
                            children=[
                                dbc.Col(dbc.Label('Valid'), width=3),
                                dbc.Col(
                                    dbc.Switch(
                                        id='mapping-points-type',
                                        # label='Invalid?',
                                        value=False,
                                        disabled=True
                                    ),
                                    width=2
                                ),
                                dbc.Col(dbc.Label('Invalid'), width=3),
                            ],
                        ),
                        dbc.Row(
                            justify='left',
                            children=[
                                dbc.Col(
                                    dbc.Label(
                                        'Point Select:',
                                        size='sm',
                                        style={
                                            'fontSize': 12
                                        }
                                    ),
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='point-select',
                                        disabled=True,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),

        html.Br(),

        # lesions
        dbc.Card(
            children=[
                dbc.CardHeader(
                    'Lesions',
                    style={
                        'fontSize': 12,
                        'fontWeight': 'bold',
                    }
                ),
                dbc.CardBody(
                    style={'fontSize': 12},
                    children=[
                        dbc.Row(
                            align='end',
                            children=[
                                dbc.Col(
                                    width=4,
                                    children=[
                                        dbc.Switch(
                                            id='lesions-visible',
                                            label='Visible',
                                            value=False,
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    style={'fontSize': 12},
                                    width=8,
                                    children=[
                                        dcc.Dropdown(
                                            id='lesions-color-by',
                                            placeholder='RF Index',
                                            disabled=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        dbc.Form(
                            style={'fontSize': 12},
                            children=[
                                dbc.Label(
                                    'Range',
                                    width=4,
                                    size='sm',
                                    style={
                                        'fontSize': 12
                                    }
                                ),
                                dcc.RangeSlider(
                                    id='lesions-color-range',
                                    min=0.0, max=100.0,
                                    value=[0, 100],
                                    step=0.5,
                                    marks=None,
                                    allowCross=False,
                                    tooltip={
                                        'placement': 'bottom',
                                        'always_visible': True,
                                    },
                                    disabled=True,
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Label(
                                            'Clipping',
                                            width='auto',
                                            size='sm',
                                            style={
                                                'fontSize': 12
                                            }
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id='lesion-range-min',
                                                type='number',
                                                placeholder='min',
                                                min=0,
                                                max=10,
                                                step=1,
                                                size='sm',
                                                disabled=True,
                                            ),
                                            width=4,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id='lesion-range-max',
                                                type='number',
                                                placeholder='max',
                                                min=0,
                                                max=10,
                                                step=1,
                                                size='sm',
                                                disabled=True,
                                            ),
                                            width=4,
                                        ),
                                    ],
                                    className='g-2 mt-1'
                                ),
                            ],
                        ),
                    ]
                )
            ]
        ),

        html.Br(),

        # ECGs
        dbc.Card(
            children=[
                dbc.CardHeader(
                    'ECG',
                    style={
                        'fontSize': 12,
                        'fontWeight': 'bold',
                    }
                ),
                dbc.CardBody(
                    style={'fontSize': 12},
                    children=[
                        dbc.Row(
                            justify='center',
                            children=[
                                dbc.Col(dbc.Label('Map'), width=3),
                                dbc.Col(
                                    dbc.Switch(
                                        id='point-ecg',
                                        # label='Visible?',
                                        value=False,
                                        disabled=True
                                    ),
                                    width=2
                                ),
                                dbc.Col(dbc.Label('Point'), width=3),
                            ],
                        ),
                    ],
                ),
            ]
        ),
    ]


def get_off_canvas():
    """
    Off Canvas Info
    """
    return dbc.Offcanvas(
        id='info-canvas',
        title='Info',
        is_open=False,
        scrollable=True,
        children=[
            dbc.Card(
                dbc.CardBody(
                    children=[
                        html.H5('Map Info',
                                id='map-info-title',
                                className='card-title'
                                ),
                        html.P('No map selected yet...',
                               id='map-info-body',
                               className='card-text'
                               ),
                    ],
                ),
                style={'marginBottom': '10px'},
            ),
            dbc.Card(
                dbc.CardBody(
                    children=[
                        html.H5('Point Info',
                                id='point-info-title',
                                className='card-title'
                                ),
                        html.P('No point selected yet...',
                               id='point-info-body',
                               className='card-text'
                               ),
                    ],
                ),
                style={'marginBottom': '10px'},
            ),
        ],
    )
