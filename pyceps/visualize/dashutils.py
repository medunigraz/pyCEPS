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
from collections import namedtuple
import json
import numpy as np
from scipy.spatial import KDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots


"""
Load VTK colormaps and translate to plotly
"""
colormaps = []
plotlyColormap = namedtuple('plotlyColormap', ['name', 'colorscale'])

with open(os.path.join(os.path.dirname(__file__), 'colormaps.json')) as f:
    data = json.load(f)

for cmap in data:
    if 'RGBPoints' not in cmap:
        continue
    vtk_colors = np.asarray(cmap['RGBPoints'])
    mpl_colors = vtk_colors.reshape((4, -1), order='F').T
    plotly_colors = []
    for color in mpl_colors:
        plotly_colors.append([color[0],
                              'rgb'+str((color[1], color[2], color[3]))]
                             )
    colormaps.append(plotlyColormap(cmap['Name'], plotly_colors))


def to_drop_option(name):
    """
    Single name to dcc dropdown menu option.
    Value of the dropdown option is the same as its name.

    Parameters
    ----------
    name : string
        dropdown menu option

    Returns
    -------
    option : dict

    """

    return {"label": name, "value": name}


def rgb_norm(rgb_string):
    """
    Convert RGB string to list of normalized RGB values.

    Parameters:
        rgb_string : str
            RGB string of form "rgb(R, G, B)" with values ranging from 0 to 255

    Raises:
        ValueError : if wrong string format

    Returns:
        list of float

    """

    if not rgb_string.startswith('rgb(') or not rgb_string.endswith(')'):
        raise ValueError('RGB values must be given as string of form '
                         '"rgb(R, G, B)" with values ranging from 0 to 255')

    return [float(x)/255 for x in rgb_string[4:-1].split(',')]


def rgb_complement(rgb_string):
    """
    Get complementary color.

    Parameters:
        rgb_string : str
            RGB string of form "rgb(R, G, B)" with values ranging from 0 to 255

    Raises:
        ValueError : if wrong string format

    Returns:
        str

    """

    values = [255 - int(x) for x in rgb_string[4:-1].split(',')]

    return 'rgb({}, {}, {})'.format(values[0], values[1], values[2])


def find_closest_point(location, points, k=1, distance_upper_bound=5.0):
    """
    Find the closest point to a given location.

    Parameters:
        location : ndarray (1, 3)
            cartesian coordinates
        points : ndarray (n, 3)
            point cloud to search
        k : int
            number of nearest neighbors to return
        distance_upper_bound : float
            return only neighbors within this distance

    Returns:
        dd : float
            distance to the closest point, 'inf' if no point found within
            distance_upper_bound.
        ii : int
            index of the closest point
    """

    return KDTree(points).query(location,
                                k=k,
                                distance_upper_bound=distance_upper_bound)


def empty_figure(bgnd='rgb(255, 255, 255)'):
    """
    Create empty Plotly.Go figure canvas.Canvas

    Parameters
    ----------
    bgnd : rgb
        background color of the figure.

    Returns
    -------
    fig : plotly.go.Figure

    """

    fig = go.Figure(
        layout=go.Layout(
            paper_bgcolor=bgnd,
            plot_bgcolor=bgnd,
            xaxis=go.layout.XAxis(
                visible=False,
            ),
            yaxis=go.layout.YAxis(
                visible=False,
            ),
        )
    )

    return fig


def get_point_egm_figure(point, bgnd='rgb(255, 255, 255)'):
    """
    Plot EGM traces for a Precision point.

    Parameters:
        point : PrecisionPoint
            plot egm traces for this point.
        bgnd : RGB color
            background color of the figure.

    Raises:
        TypeError : if point is not a PrecisionPoint

    Returns:
        plotly.go.Figure

    """

    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        shared_yaxes=False,
                        )

    # add bipolar trace
    if point.egmBip:
        n_samples = len(point.egmBip.data)
        fs = point.egmBip.fs
        t = np.linspace(0, n_samples / fs, num=n_samples, endpoint=True)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=point.egmBip.data,
                mode='lines',
                name=point.egmBip.name
            ),
            row=1, col=1
        )
        # mark annotation times
        # if point.refAnnotation:
        #     fig.add_vline(
        #         x=point.refAnnotation / fs,
        #         annotation_text='REF',
        #         annotation_position='bottom',
        #         line_width=1.5,
        #         line_dash='dot',
        #         line_color='black',
        #         row=1, col=1,
        #     )
        if point.latAnnotation:
            fig.add_vline(
                x=point.latAnnotation / fs,
                annotation_text='LAT',
                annotation_position='top left',
                line_width=1.5,
                line_dash='dot',
                line_color='green',
                row=1, col=1,
            )

    # add unipolar trace
    if point.egmUni:
        for trace in point.egmUni:
            n_samples = len(trace.data)
            fs = trace.fs
            t = np.linspace(0, n_samples / fs, num=n_samples, endpoint=True)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=trace.data,
                    mode='lines',
                    name=trace.name
                ),
                row=2, col=1
            )
            # mark annotation times
            # if point.refAnnotation:
            #     fig.add_vline(
            #         x=point.refAnnotation / fs,
            #         annotation_text='REF',
            #         annotation_position='bottom',
            #         line_width=1.5,
            #         line_dash='dot',
            #         line_color='black',
            #         row=2, col=1,
            #     )
            if point.latAnnotation:
                fig.add_vline(
                    x=point.latAnnotation / fs,
                    annotation_text='LAT',
                    annotation_position='top left',
                    line_width=1.5,
                    line_dash='dot',
                    line_color='green',
                    row=2, col=1,
                )

    # add reference trace
    if point.egmRef:
        n_samples = len(point.egmRef.data)
        fs = point.egmRef.fs
        t = np.linspace(0, n_samples / fs, num=n_samples, endpoint=True)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=point.egmRef.data,
                mode='lines',
                name=point.egmRef.name
            ),
            row=3, col=1
        )
        # mark annotation times
        if point.refAnnotation:
            fig.add_vline(
                x=point.refAnnotation / fs,
                annotation_text='REF',
                annotation_position='bottom right',
                line_width=1.5,
                line_dash='dot',
                line_color='black',
                row=3, col=1,
            )

    # update x-axis properties
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)

    # update yaxis properties
    fig.update_yaxes(title_text='bip (mV)', row=1, col=1)
    fig.update_yaxes(title_text='uni (mV)', row=2, col=1)
    fig.update_yaxes(title_text='ref (mV)', row=3, col=1)

    # update layout
    fig.update_layout(title='Point {}'.format(point.name),
                      title_x=0.5,
                      yaxis1={'tickformat': '.1f'},
                      yaxis2={'tickformat': '.1f'},
                      yaxis3={'tickformat': '.1f'},
                      legend=dict(
                          orientation='h',
                          yanchor='bottom',
                          y=1.02,
                          xanchor='right',
                          x=1,
                      ),
                      plot_bgcolor=bgnd,
                      ),

    return fig


def get_point_ecg_figure(point, colors, bgnd='rgb(255, 255, 255)'):
    """
    Plot surface ECG traces for a Precision Map.

    Parameters:
         point : EPPoint
         colors : list of HEX colors
         bgnd : RGB color (optional)
            background color of the figure.

    Raises:
         TypeError : If traces are not Trace objects

    Returns:
        plotly.go.Figure

    """

    if not len(point.ecg) > 0:
        return empty_figure(bgnd='rgb(255, 255, 255)')

    ecg = point.ecg

    fig = make_subplots(rows=3, cols=4,
                        shared_xaxes='all',
                        shared_yaxes=False,
                        subplot_titles=[t.name for t in ecg],
                        )

    # get max y-range in data
    rng = [min([t.data.min() for t in ecg]),
           max([t.data.max() for t in ecg])
           ]
    # limit range in case of amplifier saturation, etc.
    rng[0] = max(rng[0], -10.0)
    rng[1] = min(rng[1], +10.0)

    for i, trace in enumerate(ecg):
        pos = np.unravel_index(i, (3, 4))
        t = np.linspace(0, trace.data.shape[0] / trace.fs,
                        num=trace.data.shape[0],
                        endpoint=True)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=trace.data,
                mode='lines',
                line=dict(color=colors[0]),
                name=trace.name,
                showlegend=False,
            ),
            row=pos[0] + 1, col=pos[1] + 1
        )

    # mark annotation time
    if not np.isnan(point.refAnnotation):
        fig.add_vline(
            x=point.refAnnotation / ecg[0].fs,
            line_width=1.5,
            line_dash='dot',
            line_color='green'
        )

    # update x-axis properties
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='ECG (mV)', row=3, col=1)

    # update layout
    # update layout
    fig.update_layout(title='Point ECGs',
                      title_x=0.5,
                      yaxis1={'tickformat': '.1f',
                              'range': rng},
                      yaxis2={'tickformat': '.1f',
                              'range': rng},
                      yaxis3={'tickformat': '.1f',
                              'range': rng},
                      yaxis4={'tickformat': '.1f',
                              'range': rng},
                      yaxis5={'tickformat': '.1f',
                              'range': rng},
                      yaxis6={'tickformat': '.1f',
                              'range': rng},
                      yaxis7={'tickformat': '.1f',
                              'range': rng},
                      yaxis8={'tickformat': '.1f',
                              'range': rng},
                      yaxis9={'tickformat': '.1f',
                              'range': rng},
                      yaxis10={'tickformat': '.1f',
                               'range': rng},
                      yaxis11={'tickformat': '.1f',
                               'range': rng},
                      yaxis12={'tickformat': '.1f',
                               'range': rng},
                      # font_size=10,
                      # legend=dict(
                      #     orientation='h',
                      #     yanchor='bottom',
                      #     y=1.02,
                      #     xanchor='right',
                      #     x=1,
                      # ),
                      # showlegend=False,
                      plot_bgcolor=bgnd,
                      )

    return fig


def get_bsecg_figure(bsecg, colors, bgnd='rgb(255, 255, 255)'):
    """
    Plot surface ECG traces for a Precision Map.

    Parameters:
         bsecg : list of BodySurfaceECG objects
         colors : list of HEX colors
         bgnd : RGB color (optional)
            background color of the figure.

    Raises:
         TypeError : If traces are not Trace objects

    Returns:
        plotly.go.Figure

    """

    if not len(bsecg) > 0:
        return empty_figure(bgnd='rgb(255, 255, 255)')

    fig = make_subplots(rows=3, cols=4,
                        shared_xaxes='all',
                        shared_yaxes=False,
                        subplot_titles=[o.name for o in bsecg[0].traces],
                        )

    # get max y-range in data
    rng = [min([t.data.min() for m in bsecg for t in m.traces]),
           max([t.data.max() for m in bsecg for t in m.traces])
           ]
    # limit range in case of amplifier saturation, etc.
    rng[0] = max(rng[0], -10.0)
    rng[1] = min(rng[1], +10.0)

    for c, surf_ecg in enumerate(bsecg):
        method = surf_ecg.method
        traces = surf_ecg.traces

        for i, trace in enumerate(traces):
            pos = np.unravel_index(i, (3, 4))
            t = np.linspace(0, trace.data.shape[0] / trace.fs,
                            num=trace.data.shape[0],
                            endpoint=True)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=trace.data,
                    mode='lines',
                    line=dict(color=colors[c]),
                    name=method,
                    legendgroup=method,
                    showlegend=True if i == 0 else False,
                ),
                row=pos[0] + 1, col=pos[1] + 1
            )
            # update yaxis properties
            # fig.update_yaxes(title_text='{} (mV)'.format(name),
            #
    # mark annotation time
    if not np.isnan(bsecg[0].refAnnotation):
        fig.add_vline(
            x=bsecg[0].refAnnotation / bsecg[0].traces[0].fs,
            # annotation_text='LAT',
            # annotation_position='bottom right',
            line_width=1.5,
            line_dash='dot',
            line_color='green'
        )

    # update x-axis properties
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)

    # update layout
    # update layout
    fig.update_layout(title='Map ECGs',
                      title_x=0.5,
                      yaxis1={'tickformat': '.1f',
                              'range': rng},
                      yaxis2={'tickformat': '.1f',
                              'range': rng},
                      yaxis3={'tickformat': '.1f',
                              'range': rng},
                      yaxis4={'tickformat': '.1f',
                              'range': rng},
                      yaxis5={'tickformat': '.1f',
                              'range': rng},
                      yaxis6={'tickformat': '.1f',
                              'range': rng},
                      yaxis7={'tickformat': '.1f',
                              'range': rng},
                      yaxis8={'tickformat': '.1f',
                              'range': rng},
                      yaxis9={'tickformat': '.1f',
                              'range': rng},
                      yaxis10={'tickformat': '.1f',
                               'range': rng},
                      yaxis11={'tickformat': '.1f',
                               'range': rng},
                      yaxis12={'tickformat': '.1f',
                               'range': rng},
                      # font_size=10,
                      # legend=dict(
                      #     orientation='h',
                      #     yanchor='bottom',
                      #     y=1.02,
                      #     xanchor='right',
                      #     x=1,
                      # ),
                      # showlegend=False,
                      plot_bgcolor=bgnd,
                      )

    return fig


def get_colorbar(colormap, title, rng, bgnd='rgb(51, 76, 102)'):
    """
    Create colorbar figure from matplotlib colorbar name.

    Parameters
    ----------
    colormap : matplotlib.ListedColormap
        matplotlib colormap.
    title : string
        colorbar title.
    rng : list/ndarray
        colorbar range.
    bgnd : rgb (optional)
        figure background color.

    Returns
    -------
    fig : plotly.go.Figure

    """

    # get complementary color for text
    text_color = rgb_complement(bgnd)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       colorscale=colormap,
                       showscale=True,
                       cmin=rng[0],
                       cmax=rng[1],
                       colorbar=dict(
                           title_text=title,
                           title_font_color=text_color,
                           title_side='top',
                           thicknessmode="pixels", thickness=50,
                           # lenmode="pixels", len=500,
                           yanchor="middle", y=0.5, ypad=10,
                           xanchor="left", x=0., xpad=10,
                           ticks="outside",
                           tickcolor=text_color,
                           tickfont={'color': text_color},
                           #  dtick=5
                       )
                   ),
                   hoverinfo='none'
                   )
    )
    fig.update_layout(width=100,
                      margin={'b': 0, 'l': 0, 'r': 0, 't': 0},
                      autosize=False,
                      plot_bgcolor=bgnd)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig
