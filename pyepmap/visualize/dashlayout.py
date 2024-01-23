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

from .dashelements import get_vtk_view, get_controls, get_off_canvas
from .dashutils import empty_figure


def get_layout(vtk_bgnd, surf_colormap_name, point_colormap_name,
               lesion_colormap_name):
    return dbc.Container(
        fluid=True,
        style={
            'marginTop': '15px',
            'height': 'calc(100vh - 30px)'
        },
        children=[
            dcc.Store(
                id='surface-map-range',
                data=[0, 1]
            ),
            dcc.Store(
                id='lesion-range',
                data=[0, 1]
            ),

            get_off_canvas(),

            dbc.Row(
                dbc.Col(
                    html.H3(
                        'Title',
                        id='study-name',
                        style={'textAlign': 'center'}
                    ),
                    width={'size': True}
                ),
            ),

            dbc.Row(html.Hr()),

            dbc.Row(
                style={
                    'height': '90vh',
                    'marginLeft': '5px'
                },
                children=[
                    dbc.Col(
                        width=5,
                        style={
                            'backgroundColor': vtk_bgnd,
                        },
                        children=[
                            html.Div(
                                dbc.Spinner(
                                    html.Div(
                                        id='vtk-view-container',
                                        children=get_vtk_view(
                                            vtk_bgnd,
                                            surf_colormap_name,
                                            point_colormap_name,
                                            lesion_colormap_name,
                                        ),
                                        style={
                                            'height': '90vh',
                                            'width': '100%'
                                        },
                                    ),
                                    color='light',
                                    delay_show=100,
                                    spinner_style={
                                        'marginLeft': '55%',
                                    }
                                ),
                                style={
                                    'height': '100%',
                                    'width': '100%',
                                    'backgroundColor': vtk_bgnd,
                                },
                            ),
                        ],
                    ),

                    dbc.Col(
                        width=1,
                        style={
                            'height': '100%',
                            'backgroundColor': vtk_bgnd,
                        },
                        children=[
                            html.Div(
                                dcc.Graph(
                                    id='surface-colorbar',
                                    figure=empty_figure(bgnd=vtk_bgnd),
                                    config={
                                        'displayModeBar': False,
                                    },
                                    style={
                                        'height': '100%',
                                        'width': '100%',
                                        'backgroundColor': vtk_bgnd,
                                    },
                                ),
                                style={
                                    'height': '50%',
                                    'width': '100%',
                                    'backgroundColor': vtk_bgnd,
                                },
                            ),
                            html.Div(
                                dcc.Graph(
                                    id='lesion-colorbar',
                                    figure=empty_figure(bgnd=vtk_bgnd),
                                    config={
                                        'displayModeBar': False,
                                    },
                                    style={
                                        'height': '100%',
                                        'width': '100%',
                                        'backgroundColor': vtk_bgnd,
                                    },
                                ),
                                style={
                                    'height': '50%',
                                    'width': '100%',
                                },
                            ),
                        ],
                    ),

                    dbc.Col(
                        width=4,
                        style={
                            'height': '100%',
                        },
                        children=[
                            html.Div(
                                dbc.Spinner(
                                    dcc.Graph(
                                        id='point-egm-figure',
                                        figure=empty_figure(),
                                        config={
                                            'displayModeBar': True,
                                        },
                                        style={
                                            'height': '100%',
                                            'width': '100%',
                                        },
                                    ),
                                    color='primary',
                                    delay_show=100,
                                    spinner_style={
                                        'marginLeft': '55%',
                                    }
                                ),
                                style={
                                    'height': '50%',
                                    'width': '100%',
                                },
                            ),
                            html.Div(
                                dbc.Spinner(
                                    dcc.Graph(
                                        id='surf-ecg-figure',
                                        figure=empty_figure(),
                                        config={
                                            'displayModeBar': True,
                                        },
                                        style={
                                            'height': '100%',
                                            'width': '100%',
                                        },
                                    ),
                                    color='primary',
                                    delay_show=100,
                                    spinner_style={
                                        'marginLeft': '55%',
                                    }
                                ),
                                style={
                                    'height': '50%',
                                    'width': '100%',
                                },
                            ),
                        ],
                    ),

                    dbc.Col(
                        width=2,
                        children=get_controls(surf_colormap_name),
                        style={
                            'height': '100%',
                        },
                        className='',
                    ),
                ],
            ),
            # for debugging hover and click info
            html.Pre(
                id="tooltip",
                style={
                    "position": "absolute",
                    "bottom": "30px",
                    "left": "25px",
                    "zIndex": 1,
                    "color": "white",
                    'fontSize': 10,
                },
            ),
        ],
    )
