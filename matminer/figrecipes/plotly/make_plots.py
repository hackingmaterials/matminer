from __future__ import division, unicode_literals, print_function
import numpy as np
import os.path
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as FF
import warnings

from copy import deepcopy
from scipy import stats
from pandas.api.types import is_numeric_dtype

__authors__ = 'Saurabh Bajaj <sbajaj@lbl.gov>, Alex Dunn <ardunn@lbl.gov>, ' \
              'Alireza Faghaninia  <alireza@lbl.gov>'

# todo: font_scale instead of all options, etc., bigger fonts
# todo: No multiple colors on violin
# todo: Change plot mode in each method if you want, and accept a plotlydict obj
# todo: fix nonetypes in __init__
# todo: clean this mess

class PlotlyFig:
    def __init__(self, df=None, plot_mode='offline', plot_title=None,
                 x_title=None, y_title=None, colbar_title='auto',
                 hovermode='closest', filename='auto',
                 show_offline_plot=True, username=None, colorscale='Viridis',
                 api_key=None, textsize=25, ticksize=25,
                 fontfamily='Courier', height=None, width=None, scale=None,
                 margins=100, pad=0, marker_scale=1.0, x_scale='linear',
                 y_scale='linear', hoverinfo='x+y+text'):
        """
        Class for making Plotly plots

        Args:
            df (DataFrame): A pandas dataframe object which can be used to
                generate several plots.
            plot_mode: (str)
                (i) 'offline': creates and saves plots on the local disk
                (ii) 'notebook': to embed plots in a IPython/Jupyter notebook,
                (iii) 'online': save the plot in your online plotly account,
                (iv) 'static': save a static image of the plot locally
                (v) 'return': Any plotting method returns its Plotly Figure
                object. Useful for fine tuning the plot.
                NOTE: Both 'online' and 'static' modes require either the fields
                'username' and 'api_key' or Plotly credentials file.
            plot_title: (str) title of plot
            x_title: (str) title of x-axis
            y_title: (str) title of y-axis
            colbar_title (str or None): the colorbar (z) title. If set to
                "auto" the name of the third column (if pd.Series) is chosen.
            hovermode: (str) determines the mode of hover interactions. Can be
                'x'/'y'/'closest'/False
            filename: (str) name/filepath of plot file
            show_offline_plot: (bool) automatically open the plot (the plot is
                saved either way); only applies to 'offline' mode.
            username: (str) plotly account username
            colorscale: (str) Sets the colorscale (colormap). It can be an array
                containing arrays mapping a normalized value to an rgb, rgba,
                hex, hsl, hsv, or named color string. At minimum, a mapping for
                the lowest (0) and highest (1) values are required.
                Example: '[[0, 'rgb(0,0,255)', [1, 'rgb(255,0,0)']]'.
                Alternatively, it may be a palette name from the following list:
                Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Jet,
                Picnic, Rainbow, Portland, Hot, Blackbody, Earth, Electric, Viridis
            api_key: (str) plotly account API key
            textsize: (int) size of text of plot title and axis titles
            ticksize: (int) size of ticks
            fontfamily: (str) HTML font family - the typeface that will be applied by the web browser. The web browser
                will only be able to apply a font if it is available on the system which it operates. Provide multiple
                font families, separated by commas, to indicate the preference in which to apply fonts if they aren't
                available on the system. The plotly service (at https://plot.ly or on-premise) generates images on a
                server, where only a select number of fonts are installed and supported. These include "Arial", "Balto",
                 "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT",
                 "Open Sans", "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman".
            height: (float) output height (in pixels)
            width: (float) output width (in pixels)
            scale: (float) Increase the resolution of the image by `scale` amount, eg: 3. Only valid for PNG and
                JPEG images.
            margins (float or [float]): Specify the margin (in px) with a list [top, bottom, right, left], or a
                number which will set all margins.
            pad: (float) Sets the amount of padding (in px) between the plotting area and the axis lines
            marker_scale (float): scale the size of all markers w.r.t. defaults
            x_scale: (str) Sets the x axis scaling type. Select from 'linear', 'log', 'date', 'category'.
            y_scale: (str) Sets the y axis scaling type. Select from 'linear', 'log', 'date', 'category'.
            hoverinfo: (str) Any combination of "x", "y", "z", "text", "name"
                joined with a "+" OR "all" or "none" or "skip".
                Examples: "x", "y", "x+y", "x+y+z", "all"
                Determines which trace information appear on hover. If `none` or `skip` are set, no information is
                displayed upon hovering. But, if `none` is set, click and hover events are still fired.
        Returns: None

        """
        self.df = df
        self.title = plot_title
        self.x_title = x_title
        self.x_scale = x_scale
        self.y_title = y_title
        self.y_scale = y_scale
        self.colbar_title = colbar_title
        self.hovermode = hovermode
        self.filename = filename
        self.plot_mode = plot_mode
        self.show_offline_plot = show_offline_plot
        self.username = username
        self.colorscale = colorscale
        self.api_key = api_key
        self.textsize = textsize
        self.ticksize = ticksize
        self.fontfamily = fontfamily
        self.height = height
        self.width = width
        self.scale = scale
        self.autosize = True if not height and not width else False

        if not isinstance(margins, (list, tuple, np.ndarray)):
            margins = [margins] * 4

        self.margins = dict(t=margins[0],
                            b=margins[1] + self.ticksize + self.textsize,
                            r=margins[2],
                            l=margins[3] + self.ticksize + self.textsize,
                            pad=pad)

        # AF: the following is what I added
        self.marker_scale = marker_scale
        self.plot_counter = 1
        self.hoverinfo = hoverinfo

        # Make default layout
        self.layout = dict(
            title=self.title,
            titlefont=dict(size=self.textsize, family=self.fontfamily),
            xaxis=dict(title=self.x_title, type=self.x_scale,
                       titlefont=dict(size=self.textsize,
                                      family=self.fontfamily),
                       tickfont=dict(size=self.ticksize,
                                     family=self.fontfamily)),
            yaxis=dict(title=self.y_title, type=self.y_scale,
                       titlefont=dict(size=self.textsize,
                                      family=self.fontfamily),
                       tickfont=dict(size=self.ticksize,
                                     family=self.fontfamily)),
            hovermode=self.hovermode,
            autosize=self.autosize,
            margin=self.margins,
            legend=dict(font=dict(family=self.fontfamily, size=self.textsize))
        )

        if width is not None:
            self.layout['width'] = width
        if height is not None:
            self.layout['height'] = height

        if self.plot_mode == 'online' or self.plot_mode == 'static':
            if not os.path.isfile('~/.plotly/.credentials'):
                if not self.username:
                    raise ValueError(
                        'field "username" must be filled in online plotting '
                        'mode')
                if not self.api_key:
                    raise ValueError(
                        'field "api_key" must be filled in online plotting '
                        'mode')
                plotly.tools.set_credentials_file(username=self.username,
                                                  api_key=self.api_key)

        if self.plot_mode == 'static':
            if not self.filename or not self.filename.lower().endswith(
                    ('.png', '.svg', '.jpeg', '.pdf')):
                raise ValueError(
                    'field "filename" must be filled in static plotting mode '
                    'and must have an extension ending in ('
                    '".png", ".svg", ".jpeg", ".pdf")')

    def create_plot(self, fig):
        """
        Creates a plotly plot based on its dictionary representation.
        The modes of plotting are:
            (i) offline: Makes an offline html.
            (ii) notebook: Embeds in Jupyter notebook
            (iii) online: Send to Plotly, requires credentials
            (iv) static: Creates a static image of the plot
            (v) return: Returns the dictionary representation of the plot.

        Args:
            fig: (dictionary) contains data and layout information

        Returns:
            A Plotly Figure object (if self.plot_mode = 'return')

        """

        if self.filename == 'auto':
            filename = 'auto_{}'.format(self.plot_counter)
        else:
            filename = self.filename

        if self.plot_mode == 'offline':
            if not filename.endswith('.html'):
                filename += '.html'
            plotly.offline.plot(fig, filename=filename,
                                auto_open=self.show_offline_plot)
        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            if filename:
                plotly.plotly.plot(fig, filename=filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=filename,
                                        height=self.height, width=self.width,
                                        scale=self.scale)

        elif self.plot_mode == 'return':
            return fig

        self.plot_counter += 1


    def data_from_col(self, col, df=None):
        """
        try to get data based on column name in dataframe and return
            informative error if failed.
        Args:
            col (str): column name to look for
        Returns (pd.Series or col itself):
        """
        # todo: df arg is not doing anything?
        if isinstance(col, str):
            try:
                return self.df[col]
            except:
                raise ValueError('"{}" not in the data!'.format(col))
        else:
            return col


    def xy(self, xy_pairs, colbar=None, colbar_range=None, labels=None,
           names=None, sizes=None, modes='markers', markers=None, lines=None,
           colorscale=None, showlegends=None, normalize_size=True):
        """
        Make an XY scatter plot, either using arrays of values, or a dataframe.
        Args:
            xy_pairs (tuple or [tuple]): x & y columns of scatter plots
                with possibly different lengths are extracted from this arg
                example 1: ([1, 2], [3, 4])
                example 2: [(df['x1'], df['y1']), (df['x2'], df['y2'])]
                example 3: [('x1', 'y1'), ('x2', 'y2')]
            colbar (list or np.ndarray or pd.Series): set the colorscale for
                the colorbar (list of numbers); overwrites marker['color']
            colbar_range ([min, max]): the range of numbers included in colorbar.
                if any number is outside of this range, it will be forced to
                either one. Note that if colbar_range is set, the colorbar ticks
                will be updated to reflext -min or max+ at the two ends.
            labels (list or [list]): to individually set annotation for scatter
                point either the same for all traces or can be set for each
            names (str or [str]): list of trace names used for legend. By
                default column name (or trace if NA) used if pd.Series passed
            sizes (str, float, [float], [list]). Options:
                str: column name in data with list of numbers used for marker size
                float: a single size used for all traces in xy_pairs
                [float]: list of fixed sizes used for traces (length==len(xy_pairs))
                [list]: list of list of sizes for each trace in xy_pairs
            modes (str or [str]): trace style; can be 'markers'/'lines'/'lines+markers'
            markers (dict or [dict]): gives the ability to fine tune marker
                of each scatter plot individually if list of dicts passed. Note
                that the key "size" is forbidden in markers. Use sizes arg instead.
            lines (dict or [dict]: similar to markers though only if mode=='lines'
            colorscale (str):  see the colorscale doc in __init__
            showlegends (bool or [bool]): indicating whether to show legend
                for each trace (or simply turn it on/off for all if not list)
            normalize_size (bool): if True, normalize the size list.
        Returns: A Plotly Scatter plot Figure object.
        """
        if not isinstance(xy_pairs, list):
            xy_pairs = [xy_pairs]
        if not isinstance(showlegends, list):
            showlegends = [showlegends]
        if len(showlegends) == 1:
            showlegends *= len(xy_pairs)
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names] * len(xy_pairs)
        else:
            assert len(names) == len(xy_pairs)
        if sizes is None:
            sizes = [10 * self.marker_scale] * len(xy_pairs)
        elif isinstance(sizes, str):
            sizes = [self.data_from_col(sizes)] * len(xy_pairs)
        else:
            if len(sizes) != len(xy_pairs):
                raise ValueError('"sizes" must be the same length as "xy_pairs"')
            for i, _ in enumerate(sizes):
                sizes[i] = self.data_from_col(sizes[i])

        # if zscore_size:
        #     for i, _ in enumerate(sizes):
        #         if isinstance(sizes[i], (list, np.ndarray, pd.Series)):
        #             sizes[i] = (stats.zscore(pd.Series(sizes[i])) + 5) * 3

        if normalize_size:
            for i, size in enumerate(sizes):
                if isinstance(sizes[i], (list, np.ndarray, pd.Series)):
                    size = pd.Series(size).fillna(size.min())
                    sizes[i] = ((size-size.min())/(size.max()-size.min())+0.05)* 30 * self.marker_scale
                    print(sizes[i])

        if isinstance(modes, str):
            modes = [modes] * len(xy_pairs)
        else:
            assert len(modes) == len(xy_pairs)
        if colbar is None:
            showscale = False
        else:
            showscale = True
            colorbar = self.data_from_col(colbar)
            assert isinstance(colorbar, (list, np.ndarray, pd.Series))
            if colbar_range:
                colorbar = pd.Series(colorbar)
                colorbar[colorbar < colbar_range[0]] = colbar_range[0]
                colorbar[colorbar > colbar_range[1]] = colbar_range[1]
        data = []
        for pair in xy_pairs:
            data.append((self.data_from_col(pair[0]),
                         self.data_from_col(pair[1])))
            if isinstance(pair[1], str):
                names.append(pair[1])
            else:
                try:
                    names.append(pair[1].name)
                except:
                    names.append(None)

        if not isinstance(labels, list):
            labels = self.data_from_col(labels)
            labels = [labels] * len(data)
        else:
            labels = [self.data_from_col(l) for l in labels]
        markers = markers or [{'symbol': 'circle', 'line': {'width': 1,
                    'color': 'black'}} for _ in data]
        if isinstance(markers, dict):
            markers = [markers.copy() for _ in data]

        if self.colbar_title=='auto':
            colbar_title = pd.Series(colorbar).name
        else:
            colbar_title = self.colbar_title

        for im, _ in enumerate(markers):
            markers[im]['showscale'] = showscale
            if markers[im].get('size', None) is None:
                markers[im]['size'] = sizes[im]
            else:
                raise ValueError('"size" must not be set in markers, use sizes argument instead')
            if colorbar is not None:
                markers[im]['color'] = colorbar
                fontd = {'family': self.fontfamily, 'size': 0.75*self.ticksize}
                markers[im]['colorbar'] = {'title': colbar_title, 'titleside': 'right',
                                           'tickfont': fontd, 'titlefont': fontd}
                if colbar_range is not None:
                    tickvals = np.linspace(colbar_range[0], colbar_range[1], 6)
                    ticktext = [str(round(tick, 1)) for tick in tickvals]
                    ticktext[0] = '-'+ ticktext[0]
                    ticktext[-1] = ticktext[-1] + '+'
                    markers[im]['colorbar']['tickvals'] = tickvals
                    markers[im]['colorbar']['ticktext'] = ticktext
            if markers[im].get('colorscale') is None:
                markers[im]['colorscale'] = colorscale or self.colorscale

        lines = lines or [{'dash': 'solid', 'width': 2}] * len(data)

        for var in [labels, markers, lines]:
            assert len(list(var)) == len(data)

        traces = []
        for i, xy_pair in enumerate(data):
            traces.append(go.Scatter(x=xy_pair[0], y=xy_pair[1], mode=modes[i],
                                     marker=markers[i], line=lines[i],
                                     text=labels[i], hoverinfo=self.hoverinfo,
                                     name=names[i], showlegend=showlegends[i],
                                     ))
        layout = self.layout.copy()
        if layout['xaxis'].get('title') is None and len(data) == 1:
            layout['xaxis']['title'] = pd.Series(data[0][0]).name
        if layout['yaxis'].get('title') is None and len(data) == 1:
            layout['yaxis']['title'] = pd.Series(data[0][1]).name

        fig = {'data': traces, 'layout': layout}
        if showscale:
            fig['layout']['legend']['x'] = 0.9
        return self.create_plot(fig)



    def xy_plot(self, x_col, y_col, text=None, color='rgba(70, 130, 180, 1)',
                size=6, colorscale='Viridis', legend=None,
                showlegend=False, mode='markers', marker='circle',
                marker_fill='fill', hoverinfo='x+y+text',
                add_xy_plot=None, marker_outline_width=0,
                marker_outline_color='black', linedash='solid',
                linewidth=2, lineshape='linear', error_type=None,
                error_direction=None, error_array=None,
                error_value=None, error_symmetric=True, error_arrayminus=None,
                error_valueminus=None):
        """
        Make an XY scatter plot, either using arrays of values, or a dataframe.

        Args:
            x_col: (array) x-axis values, which can be a list/array/dataframe column
            y_col: (array) y-axis values, which can be a list/array/dataframe column
            text: (str/array) text to use when hovering over points; a single string, or an array of strings, or a
                dataframe column containing text strings
            color: (str/array) in the format of a (i) color name (eg: "red"), or (ii) a RGB tuple,
                (eg: "rgba(255, 0, 0, 0.8)"), where the last number represents the marker opacity/transparency, which
                must be between 0.0 and 1.0., (iii) hexagonal code (eg: "FFBAD2"), or (iv) name of a dataframe
                numeric column to set the marker color scale to
            size: (int/array) marker size in the format of (i) a constant integer size, or (ii) name of a dataframe
                numeric column to set the marker size scale to. In the latter case, scaled Z-scores are used.
            colorscale: (str) Sets the colorscale. The colorscale must be an array containing arrays mapping a
                normalized value to an rgb, rgba, hex, hsl, hsv, or named color string. At minimum, a mapping for the
                lowest (0) and highest (1) values are required. For example, `[[0, 'rgb(0,0,255)',
                [1, 'rgb(255,0,0)']]`. Alternatively, `colorscale` may be a palette name string of the following list:
                Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
                Blackbody, Earth, Electric, Viridis
            legend: (str) plot legend
            mode: (str) marker style; can be 'markers'/'lines'/'lines+markers'
            marker: (str) Shape of marker symbol. For all options, please see
                https://plot.ly/python/reference/#scatter-marker-symbol
            marker_fill: (str) Shape fill of marker symbol. Options are "fill"/"open"/"dot"/"open-dot"
            hoverinfo: (str) Any combination of "x", "y", "z", "text", "name" joined with a "+" OR "all" or "none" or
                "skip".
                Examples: "x", "y", "x+y", "x+y+z", "all"
                default: "x+y+text"
                Determines which trace information appear on hover. If `none` or `skip` are set, no information is
                displayed upon hovering. But, if `none` is set, click and hover events are still fired.
            showlegend: (bool) show legend or not
            add_xy_plot: (list) of dictionaries, each of which contain additional data to add to the xy plot. Keys are
                names of arguments to the original xy_plot method - required keys are 'x_col', 'y_col', 'text', 'mode',
                'name', 'color', 'size'. Values are corresponding argument values in the same format as for the
                original xy_plot. Use None for values not to be set, else a KeyError will be raised. Optional keys are
                'marker' and 'marker_fill' (same format as root keys)
            marker_outline_width: (int) thickness of marker outline
            marker_outline_color: (str/array) color of marker outline - accepts similar formats as other color variables
            linedash: (str) sets the dash style of a line. Options are 'solid'/'dash'
            linewidth: (int) sets the line width (in px)
            lineshape: (str) determines the line shape. With "spline" the lines are drawn using spline interpolation
            error_type: (str) Determines the rule used to generate the error bars. Options are,
                (i) "data": bar lengths are set in variable `error_array`/'error_arrayminus',
                (ii) "percent": bar lengths correspond to a percentage of underlying data. Set this percentage in the
                   variable 'error_value'/'error_valueminus',
                (iii) "constant": bar lengths are of a constant value. Set this constant in the variable
                'error_value'/'error_valueminus'
            error_direction: (str) direction of error bar, "x"/"y"
            error_array: (list/array/series) Sets the data corresponding the length of each error bar.
                Values are plotted relative to the underlying data
            error_value: (float) Sets the value of either the percentage (if `error_type` is set to "percent") or
                the constant (if `error_type` is set to "constant") corresponding to the lengths of the error bars.
            error_symmetric: (bool) Determines whether or not the error bars have the same length in both direction
                (top/bottom for vertical bars, left/right for horizontal bars
            error_arrayminus: (list/array/series) Sets the data corresponding the length of each error bar in the bottom
                (left) direction for vertical (horizontal) bars Values are plotted relative to the underlying data.
            error_valueminus: (float) Sets the value of either the percentage (if `error_type` is set to "percent") or
                the constant (if `error_type` is set to "constant") corresponding to the lengths of the error bars in
                the bottom (left) direction for vertical (horizontal) bars

        Returns: A Plotly Scatter plot Figure object.

        """
        if isinstance(color, str):
            showscale = False
        else:
            showscale = True

        # Use z-scores for sizes
        # If size is a list, convert to array for z-score calculation
        if isinstance(size, list):
            size = np.array(size)
        if isinstance(size, pd.Series):
            size = (stats.zscore(size) + 5) * 3

        if marker_fill != 'fill':
            if marker_fill == 'open':
                marker_fill += '-open'
            elif marker_fill == 'dot':
                marker_fill += '-dot'
            elif marker_fill == 'open-dot':
                marker_fill += '-open-dot'
            else:
                raise ValueError('Invalid marker fill')

        trace0 = go.Scatter(
            x=x_col,
            y=y_col,
            text=text,
            mode=mode,
            name=legend,
            hoverinfo=hoverinfo,
            marker=dict(
                size=size,
                color=color,
                colorscale=colorscale,
                showscale=showscale,
                line=dict(width=marker_outline_width,
                          color=marker_outline_color,
                          colorscale=colorscale),
                symbol=marker,
                colorbar=dict(tickfont=dict(size=int(0.75 * self.ticksize),
                                            family=self.fontfamily))
            ),
            line=dict(dash=linedash, width=linewidth, shape=lineshape)
        )

        # Add error bars
        if error_type:
            if error_direction is None:
                raise ValueError(
                    "The field 'error_direction' must be populated if 'err_type' is specified")
            if error_type == 'data':
                if error_symmetric:
                    trace0['error_' + error_direction] = dict(type=error_type,
                                                              array=error_array)
                else:
                    if not error_arrayminus:
                        raise ValueError(
                            "Please specify error bar lengths in the negative direction")
                    trace0['error_' + error_direction] = dict(type=error_type,
                                                              array=error_array,
                                                              arrayminus=error_arrayminus)
            elif error_type == 'constant' or error_type == 'percent':
                if error_symmetric:
                    trace0['error_' + error_direction] = dict(type=error_type,
                                                              value=error_value)
                else:
                    if not error_valueminus:
                        raise ValueError(
                            "Please specify error bar lengths in the negative direction")
                    trace0['error_' + error_direction] = dict(type=error_type,
                                                              value=error_value,
                                                              valueminus=error_valueminus)
            else:
                raise ValueError(
                    "Invalid error bar type. Please choose from 'data'/'constant'/'percent'.")

        data = [trace0]

        # Additional XY plots
        if add_xy_plot:
            for plot_data in add_xy_plot:

                # Check for symbol parameters, if not present, assign defaults
                if 'marker' not in plot_data:
                    plot_data['marker'] = 'circle'
                    if 'marker_fill' in plot_data:
                        plot_data['marker'] += plot_data['marker_fill']

                data.append(
                    go.Scatter(
                        x=plot_data['x_col'],
                        y=plot_data['y_col'],
                        text=plot_data['text'],
                        mode=plot_data['mode'],
                        name=plot_data['legend'],
                        hoverinfo=hoverinfo,
                        marker=dict(
                            color=plot_data['color'],
                            size=plot_data['size'],
                            colorscale=colorscale,
                            # colorscale is fixed to that of the main plot
                            showscale=showscale,
                            # showscale is fixed to that of the main plot
                            line=dict(width=marker_outline_width,
                                      color=marker_outline_color,
                                      colorscale=colorscale),
                            symbol=plot_data['marker'],
                            colorbar=dict(
                                tickfont=dict(size=int(0.75 * self.ticksize),
                                              family=self.fontfamily))
                        )
                    )
                )

        # Add legend
        self.layout['showlegend'] = showlegend

        fig = dict(data=data, layout=self.layout)
        return self.create_plot(fig)


    def heatmap(self, data=None, cols=None, x_bins=6, y_bins = 4, precision=1,
                annotation='count', annotation_color='black', colorscale=None):
        """
        Args:
            data: (array) an array of arrays. For example, in case of a pandas dataframe 'df', data=df.values.tolist()
            cols ([str]): A list of strings specifying the columns of the
                dataframe (either data or self.df) to use. Currenly, only 3
                columns is supported. Note that the order in cols matter, the
                firts is considered x, second y and the third as z (color)
            x_bins (int or None): if the unique values for x_prop is more than
                x_bins, x_prop is binned to the number of x_bins for better
                presentation
            y_bins (int or None): similar to x_bins
            precision (int): number of floating points used for binning/display
            annotation (str or None): mode of annotation. Options are: None or
                "count": the number of data available in each cell displayed
                "value": the actual value of the cell in addition to colorbar
            annotation_color (str): the color of annotation (text inside cells)
            colorscale: see the __init__ doc for colorscale
        Returns: A Plotly heatmap plot Figure object.
        """

        if data is None:
            if self.df is None:
                raise ValueError(
                    "heatmap requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif isinstance(data, (np.ndarray, list)):
            data = pd.DataFrame(data, columns=cols)

        cols = data.columns.values
        x_prop = cols[0]
        y_prop = cols[1]
        col_prop = cols[2]

        data = data.sort_values(y_prop, ascending=True)
        if y_bins is None or len(data[y_prop].unique()) > y_bins:
            data['y_bin'] = pd.cut(data[y_prop], bins=y_bins, precision=precision).astype(str)
            y_labels = data['y_bin'].unique()
        else:
            y_labels = data[y_prop].unique()

        data = data.sort_values(x_prop, ascending=True)
        if x_bins is None or len(data[x_prop].unique()) > x_bins:
            data['x_bin'] = pd.cut(data[x_prop], bins=x_bins, precision=precision).astype(str)
            x_labels = data['x_bin'].unique()
        else:
            x_labels = data[x_prop].unique()

        data_ = []
        annotations = []
        annotation_template = {'font': {'color' : annotation_color,
                'size': 0.7*self.textsize, 'family': self.fontfamily}, 'showarrow': False}
        for y in y_labels:
            temp = data[data['y_bin'].values == y]
            grouped = temp.groupby('x_bin').mean().reset_index()
            g_count = temp.groupby('x_bin').count().reset_index()
            x_data = []
            for x in x_labels:
                if x in grouped['x_bin'].values:
                    val = grouped[grouped['x_bin'].values == x][col_prop].values[0]
                    count = g_count[g_count['x_bin'].values == x][col_prop].values[0]
                    val = str(round(val, precision))
                else:
                    count = 0
                    val = 'N/A'
                x_data.append(val)
                a_d = annotation_template.copy()
                a_d['x'] = x
                a_d['y'] = y
                if annotation == 'value':
                    a_d['text'] = val
                elif annotation == 'count':
                    a_d['text'] = count
                else:
                    a_d['text'] = annotation
                annotations.append(a_d)
            data_.append(x_data)

        if self.colbar_title=='auto':
            colbar_title = col_prop
        else:
            colbar_title = self.colbar_title
        trace = go.Heatmap(z=data_, x=x_labels, y=y_labels,
            colorscale = colorscale or self.colorscale, colorbar={
                'title': colbar_title, 'titleside': 'right',
            'tickfont': {'size': 0.75 * self.ticksize,'family': self.fontfamily},
            'titlefont': {'size': self.textsize, 'family': self.fontfamily}
        })
        layout = self.layout.copy()

        # heatmap specific formatting:
        layout['xaxis'].pop('type')
        layout['yaxis'].pop('type')
        layout['margin']['l'] += self.ticksize * (2+precision/10.0) + 35
        if layout['xaxis']['title'] is None:
            warnings.warn('xaxis title was automatically set to x_prop value')
            layout['xaxis']['title'] = x_prop
        if layout['yaxis']['title'] is None:
            warnings.warn('yaxis title was automatically set to y_prop value')
            layout['yaxis']['title'] = y_prop
        layout['annotations'] = annotations
        fig = {'data': [trace], 'layout': layout}
        return self.create_plot(fig)


    def heatmap_plot(self, data, x_labels=None, y_labels=None,
                     colorscale='Viridis', colorscale_range=None,
                     annotations_text=None, annotations_text_size=20,
                     annotations_color='white'):
        """
        Make a heatmap plot, either using 2D arrays of values, or a dataframe.

        Args:
            data: (array) an array of arrays. For example, in case of a pandas dataframe 'df', data=df.values.tolist()
            x_labels: (array) an array of strings to label the heatmap columns
            y_labels: (array) an array of strings to label the heatmap rows
            colorscale: (str/array) Sets the colorscale. The colorscale must be an array containing arrays mapping a
                normalized value to an rgb, rgba, hex, hsl, hsv, or named color string. At minimum, a mapping for the
                lowest (0) and highest (1) values are required. For example, `[[0, 'rgb(0,0,255)',
                [1, 'rgb(255,0,0)']]`. Alternatively, `colorscale` may be a palette name string of the following list:
                Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
                Blackbody, Earth, Electric, Viridis
            colorscale_range: (array) Sets the minimum (first array item) and maximum value (second array item)
                of the colorscale
            annotations_text: (array) an array of arrays, with each value being a string annotation to the corresponding
                value in 'data'
            annotations_text_size: (int) size of annotation text
            annotations_color: (str/array) color of annotation text - accepts similar formats as other color variables

        Returns: A Plotly heatmap plot Figure object.

        """
        if not colorscale_range:
            colorscale_min = None
            colorscale_max = None
        elif len(colorscale_range) == 2:
            colorscale_min = colorscale_range[0]
            colorscale_max = colorscale_range[1]
        else:
            raise ValueError(
                "The field 'colorscale_range' must be a list with two values.")

        if annotations_text:
            annotations = []

            for n, row in enumerate(data):
                for m, val in enumerate(row):
                    var = annotations_text[n][m]
                    annotations.append(
                        dict(
                            text=str(var),
                            x=x_labels[m], y=y_labels[n],
                            xref='x1', yref='y1',
                            font=dict(color=annotations_color,
                                      size=annotations_text_size,
                                      family=self.fontfamily),
                            showarrow=False)
                    )
        else:
            annotations = []

        trace0 = go.Heatmap(
            z=data,
            colorscale=colorscale,
            x=x_labels,
            y=y_labels,
            zmin=colorscale_min, zmax=colorscale_max,
            colorbar=dict(tickfont=dict(size=int(0.75 * self.ticksize),
                                        family=self.fontfamily))
        )

        data = [trace0]

        # Add annotations
        self.layout['annotations'] = annotations

        fig = dict(data=data, layout=self.layout)

        return self.create_plot(fig)



    def scatter_matrix(self, data=None, cols=None, colbar=None, marker=None,
                       text=None, **kwargs):
        """
        Create a Plotly scatter matrix plot from dataframes using Plotly.
        Args:
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            cols ([str]): A list of strings specifying the columns of the
                dataframe to use.
            colbar: (str) name of the column used for colorbar
            marker (dict): if size is set, it will override the automatic size
            text (see PlotlyFig.xy_plot documentation):
            **kwargs: keyword arguments of scatterplot. Forbidden args are
                'size', 'color' and 'colorscale' in 'marker'. See example below
        Returns: a Plotly scatter matrix plot

        # Example for more control over markers:
        from matminer.figrecipes.plotly.make_plots import PlotlyFig
        from matminer.datasets.dataframe_loader import load_elastic_tensor
        df = load_elastic_tensor()
        pf = PlotlyFig()
        pf.scatter_matrix(df[['volume', 'G_VRH', 'K_VRH', 'poisson_ratio']],
                colbar_col='poisson_ratio', text=df['material_id'],
                marker={'symbol': 'diamond', 'size': 8, 'line': {'width': 1,
                'color': 'black'}}, colormap='Viridis',
                title='Elastic Properties Scatter Matrix')
        """
        height = self.height or 800
        width = self.width or 1000

        # making sure the combination of input args make sense
        if data is None:
            if self.df is None:
                raise ValueError(
                    "scatter_matrix requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=cols)

        if isinstance(text, str):
            if text in data:
                text = data[text]
            elif text in self.df:
                text = self.df[text]
            else:
                raise ValueError('string "text" arg must be present in data')
        if colbar and colbar not in data:
            if colbar in self.df:
                data[colbar] = self.df[colbar]
            else:
                raise ValueError('"{}" not found in the data'.format(colbar))

        # actual ploting:
        marker = marker or {'symbol': 'circle', 'line': {'width': 1, 'color': 'black'}}
        nplots = len(data.columns) - int(colbar is not None)
        marker_size = marker.get('size') or 5.0 * self.marker_scale
        text_scale = 0.9 / nplots**0.2
        tick_scale = 0.7 / nplots**0.3
        fig = FF.create_scatterplotmatrix(data, index=colbar, diag='histogram',
                        size=marker_size, height=height,width=width, **kwargs)

        # also update fig layout as scatter plot ignores PlotlyFig layout for some reason
        fig['layout'].update(titlefont={'family': self.fontfamily,
                    'size': self.textsize * text_scale})
        ratio ={'x': min(1., width/float(height)), 'y': min(1., height/float(width))}
        # update each plot; we don't update the histograms markers as it causes issues:
        for iplot in range(nplots ** 2):
            fig['data'][iplot].update(hoverinfo=self.hoverinfo)
            for ax in ['x', 'y']:
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'][
                    'family'] = self.fontfamily
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'][
                    'size'] = self.textsize * text_scale * ratio[ax]
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'family'] = self.fontfamily
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'size'] = self.textsize * tick_scale
            if iplot % (nplots + 1) != 0:
                fig['data'][iplot].update(marker=marker, text=text)
        return self.create_plot(fig)


    def histogram(self, data=None, cols=None, orientation="vertical",
                  histnorm="count", n_bins=None, bins=None, colors=None, bargap=0):
        """
        Creates a Plotly histogram. If multiple series of data are available,
        will create an overlaid histogram.

        For n_bins, start, end, size, colors, and bargaps, all defaults are
        Plotly defaults.

        Args:
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            cols ([str]): A list of strings specifying the columns of the
                dataframe to use. Each column will be represented with its own
                histogram in the overlay.
            orientation (str): Determines whether histogram is oriented
                horizontally or vertically. Use "vertical" or "horizontal".
            histnorm: The technique for creating the plot. Can be "probability
                density", "probability", "density", or "" (count).
            n_bins (int or [int]): The number of binds to include on each plot.
                if only one number specified, all histograms will have the same
                number of bins
            bins (dict or [dict]): specifications of the bins including start,
                end and size. If n_bins is set, size cannot be set in bins.
                Also size is ignored if start or end not specified.
                Examples: 1) bins=None, n_bins = 25
                2) bins={'start': 0, 'end': 50, 'size': 2.0}, n_bins=None
            colors (str or list): The list of colors for each histogram (if
                overlaid). If only one series of data is present or all series
                should have the same value, a single str determines the color
                of the bins.
            bargaps (float or list): The gaps between bars for all histograms
                shown.

        Returns:
            Plotly histogram figure.

        """

        # todo: bargap, start, end, size not working?

        if data is None:
            if cols is None and self.df is None:
                raise ValueError(
                    "Histogram requires either dataframe labels and a dataframe"
                    " or a list of numerical values.")
            elif self.df is not None and cols is None:
                cols = self.df.columns.values
            data = self.df[cols]

        if isinstance(cols, str):
            cols = [cols]

        if cols is None:
            if isinstance(data, pd.Series):
                cols = [data.name]
                data = pd.DataFrame({cols[0]: data.tolist()})
            elif isinstance(data, pd.DataFrame):
                cols = data.columns.values
            else:
                data = pd.DataFrame({'trace1': data})
                cols = ['trace1']

        # Transform all entries to listlike, if given as str or numbers
        dtypes = (list, np.ndarray, tuple)
        attrdict = {'colors': colors, 'n_bins': n_bins, 'bins': bins}
        for k, v in attrdict.items():
            if v is None:
                attrdict[k] = [None for _ in cols]
            elif not isinstance(v, dtypes):
                attrdict[k] = [v for _ in cols]
        colors = attrdict['colors']
        n_bins = attrdict['n_bins']
        bins = attrdict['bins']


        hgrams = []
        for i, col in enumerate(cols):
            if bins[i] is not None:
                if bins[i].get('size'):
                    if n_bins[i] is not None:
                        raise ValueError('either set "n_bins" or "bins" to avoid confusion.')
                    if not bins[i].get('start') or not bins[i].get('end'):
                        warnings.warn('size key in bins ignored when start or end not specified')
                elif n_bins[i] is not None:
                    warnings.warn('"size" not specified in "bins", "n_bins" is ignored. Either fully set "bins" or only "n_bins"')
                if bins[i].get('start') is None != bins[i].get('end') is None:
                    warnings.warn('both "start" and "end" must be present; otherwise, it is ignored.')

            d = data[col]
            if isinstance(d, np.ndarray):
                if len(d.shape) == 2:
                    d = d.reshape((len(d),))

            if orientation == 'vertical':
                h = go.Histogram(x=d, histnorm=histnorm,
                                 xbins=bins[i],
                                 nbinsx=n_bins[i],
                                 marker=dict(color=colors[i]), name=col)
            elif orientation == 'horizontal':
                h = go.Histogram(y=d, histnorm=histnorm,
                                 ybins=bins[i],
                                 nbinsy=n_bins[i],
                                 marker=dict(color=colors[i]), name=col)
            else:
                raise ValueError(
                    "The orientation must be 'horizontal' or 'vertical'.")
            hgrams.append(h)

        self.layout['hovermode'] = 'x' if orientation == 'vertical' else 'y'
        self.layout['bargap'] = bargap

        if orientation == 'vertical':
            if not self.y_title:
                self.layout['yaxis']['title'] = histnorm
        elif orientation == 'horizontal':
            if not self.x_title:
                self.layout['xaxis']['title'] = histnorm

        if len(hgrams) > 1:
            self.layout['barmode'] = 'overlay'
            for h in hgrams:
                h['opacity'] = 1.0 / float(len(hgrams)) + 0.1
        # fig = dict(data=hgrams, layout=self.layout)
        fig = {'data': hgrams, 'layout': self.layout}
        return self.create_plot(fig)

    def bar(self, data=None, cols=None, x=None, y=None, labels=None,
            barmode='group', colors=None, bargap=None):
        """
        Create a bar chart using Plotly.

        Can be used with x and y arguments or with a dataframe (passed as 'data'
        or taken from constructor).

        Args:
            data (DataFrame): The column names will become the 'x' axis. The
                rows will become sets of bars (e.g., 3 rows = 3 sets of bars
                for each x point).
            cols ([str]): A list of strings specifying columns of a DataFrame
                passed into the constructor to be used as data. Should not be
                used with 'data'.
            x (list or [list]): A list containing 'x' axis values. Can be a list
                of lists if there is more than one set of bars.
            y (list or [list]): A list containing 'y' values. Can be a list of
                lists if there is more than one set of bars (more than one set
                of data for each 'x' axis value).
            labels (str or [str]): Defines the label for each set of bars. If
                str, defines the column of the DataFrame to use for labelling.
                The column's entry for a row will be the label for that row. If
                it is a list of strings, should be used with x and y, and
                defines the label for each set of bars.
            barmode: Defines how sets of bars are displayed. Can be set to
                "group" or "stack".
            colors ([str]): The list of colors to use for each set of bars.
                The length of this list should be equal to the number of rows
                (sets of bars) present in your data.
            bargap (int/float): Separation between bars.

        Returns:
            A Plotly bar chart object.
        """

        if data is None:
            if self.df is None:
                if cols is None:
                    if x is None and y is None:
                        raise ValueError(
                            "Bar chart requires either dataframe labels and a "
                            "dataframe or lists of values for x and y.")
            else:
                # Default to having all columns represented.
                if cols is None:
                    cols = self.df.columns.values
                data = self.df[cols]
                if isinstance(labels, str):
                    data[labels] = self.df[labels]

        # If data is passed in as a dataframe, not as x,y
        if data is not None and x is None and y is None:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("'data' input type invalid. Valid types are"
                                "DataFrames.")

            if isinstance(labels, str):
                strlabel = deepcopy(labels)
                labels = data[labels]
                data = data.drop(labels=[strlabel], axis=1)
            else:
                labels = data.index.values

        if x is not None and y is not None:
            assert (len(x) == len(y))
            if not isinstance(y[0], (list, tuple, np.ndarray)):
                y = [y]
            if not isinstance(x[0], (list, tuple, np.ndarray)):
                x = [x]
            if labels is None:
                labels = [None] * len(y)
        else:
            y = data.as_matrix().tolist()
            x = [data.columns.values] * len(y)

        if colors is not None:
            if not isinstance(colors, (tuple, list, np.ndarray, pd.Series)):
                if isinstance(colors, str):
                    colors = [colors] * len(y)
                else:
                    raise ValueError("Invalid data type for 'colors.' Please "
                                     "use a list-like object or pandas Series.")
        else:
            colors = [None] * len(y)

        barplots = []
        for i in range(len(x)):
            barplot = go.Bar(x=x[i], y=y[i], name=labels[i],
                             marker=dict(color=colors[i]))
            barplots.append(barplot)

        # Prevent linear default from altering categorical bar plot
        self.layout['xaxis']['type'] = None
        self.layout['barmode'] = barmode
        self.layout['bargap'] = bargap
        fig = dict(data=barplots, layout=self.layout)
        return self.create_plot(fig)

    def violin(self, data=None, cols=None, group_col=None, groups=None,
               title=None, colors=None, use_colorscale=False):
        """
        Create a violin plot using Plotly.

        Args:
            data: (DataFrame or list) A dataframe containing at least one
                numerical column. Also accepts lists of numerical values. If
                None, uses the dataframe passed into the constructor.
            cols: ([str]) The labels for the columns of the dataframe to be
                included in the plot. Not used if data is passed in as list.
            group_col: (str) Name of the column containing the group for each
                row, if it exists. Used only if there is one entry in cols.
            groups: ([str]): All group names to be included in the violin plot.
                Used only if there is one entry in cols.
            title: (str) Title of the violin plot
            colors: (str/tuple/list/dict) either a plotly scale name (Greys,
                YlGnBu, Greens, etc.), an rgb or hex color, a color tuple, a
                list/dict of colors. An rgb color is of the form 'rgb(x, y, z)'
                where x, y and z belong to the interval [0, 255] and a color
                tuple is a tuple of the form (a, b, c) where a, b and c belong
                to [0, 1]. If colors is a list, it must contain valid color
                types as its members. If colors is a dictionary, its keys must
                represent group names, and corresponding values must be valid
                color types (str).
            use_colorscale: (bool) Only applicable if grouping by another
                variable. Will implement a colorscale based on the first 2
                colors of param colors. This means colors must be a list with
                at least 2 colors in it (Plotly colorscales are accepted since
                they map to a list of two rgb colors)

        Returns: A Plotly violin plot Figure object.

        """
        if data is None:
            if self.df is None:
                raise ValueError(
                    "Violin plot requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            data = self.df

        if isinstance(data, pd.Series):
            cols = [data.name]
            data = pd.DataFrame({data.name: data.tolist()})

        if isinstance(data, pd.DataFrame):
            if groups is None:
                if group_col is None:
                    grouped = pd.DataFrame({'data': [], 'group': []})

                    if cols is None:
                        cols = data.columns.values

                    for col in cols:
                        d = data[col].tolist()
                        temp_df = pd.DataFrame(
                            {'data': d, 'group': [col] * len(d)})
                        grouped = grouped.append(temp_df)
                    data = grouped
                    group_col = 'group'
                    groups = cols
                    cols = ['data']
                else:
                    groups = data[group_col].unique()
            else:
                if group_col is None:
                    raise ValueError(
                        "Please specify group_col, the label of the column "
                        "containing the groups for each row.")

            use_colorscale = True
            group_stats = {}

            for g in groups:
                group_data = data.loc[data[group_col] == g]
                group_stats[g] = np.median(group_data[cols])

            # Filter out groups from dataframe that have only 1 row.
            group_value_counts = data[group_col].value_counts().to_dict()

            for j in group_value_counts:
                if group_value_counts[j] == 1:
                    data = data[data[group_col] != j]
                    warnings.warn(
                        'Omitting rows with group = {} which have only one row '
                        'in the dataframe.'.format(
                            j))
        else:
            data = pd.DataFrame({'data': np.asarray(data)})
            cols = ['data']
            group_col = None
            group_stats = None

        fig = FF.create_violin(data=data, data_header=cols[0],
                               group_header=group_col, title=title,
                               height=self.height, width=self.width,
                               colors=colors, use_colorscale=use_colorscale,
                               group_stats=group_stats)

        fig.update(dict(
            layout=dict(
                title=self.title,
                titlefont=dict(size=self.textsize, family=self.fontfamily),
                yaxis=dict(title=self.y_title, type=self.y_scale,
                           titlefont=dict(size=self.textsize,
                                          family=self.fontfamily),
                           tickfont=dict(size=self.ticksize,
                                         family=self.fontfamily)),
            )
        ))

        # Change sizes in all x-axis
        for item in fig['layout']:
            if item.startswith('xaxis'):
                fig['layout'][item].update(
                    dict(
                        titlefont=dict(size=self.textsize,
                                       family=self.fontfamily),
                        tickfont=dict(size=self.ticksize,
                                      family=self.fontfamily)
                    )
                )

        if self.height is None:
            fig['layout']['height'] = 1000
        if self.width is None:
            fig['layout']['width'] = 1400

        return self.create_plot(fig)

    def parallel_coordinates(self, data=None, cols=None, line=None, precision=2,
                             colbar=None):
        """
        Create a Plotly Parcoords plot from dataframes.
        Args:
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            cols ([str]): A list of strings specifying the columns of the
                dataframe to use.
            line (dict): plotly line dict with keys such as "color" or "width"
            precision (int): the number of floating points for columns with
                float data type (2 is recommended for a nice visualization)
        Returns: a Plotly scatter matrix plot
        """
        # making sure the combination of input args make sense
        if data is None:
            if self.df is None:
                raise ValueError(
                    "scatter_matrix requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=cols)

        if cols is None:
            cols = data.columns.values

        if colbar is None:
            colbar = 'blue'
        else:
            colbar = self.data_from_col(colbar)
        if self.colbar_title=='auto':
            colbar_title = pd.Series(colbar).name
        else:
            colbar_title = self.colbar_title

        cols = list(cols)
        if pd.Series(colbar).name in cols:
            cols.remove(pd.Series(colbar).name)

        dimensions = []
        for col in cols:
            if is_numeric_dtype(data[col]) and 'int' not in str(data[col].dtype):
                values = data[col].apply(lambda x: round(x, precision))
            else:
                values = data[col]
            dimensions.append({'label': col, 'values': values})

        fontd = {'family': self.fontfamily, 'size': 0.75 * self.ticksize}
        line = line or {'color': colbar,
                        'colorbar': {'title': colbar_title, 'titleside': 'right',
                                   'tickfont': fontd, 'titlefont': fontd}}
        par_coords = go.Parcoords(line=line, dimensions=dimensions)

        fig = {'data': [par_coords]}
        return self.create_plot(fig)
