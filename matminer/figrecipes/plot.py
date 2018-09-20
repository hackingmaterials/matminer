from __future__ import division, unicode_literals, print_function
import numpy as np
import os.path
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as FF
import warnings

from copy import deepcopy
from pandas.api.types import is_numeric_dtype

__authors__ = 'Saurabh Bajaj <sbajaj@lbl.gov>, Alex Dunn <ardunn@lbl.gov>, ' \
              'Alireza Faghaninia  <alireza.faghaninia@gmail.com>'


# todo: common function for if then checking data types + automatically ignore non-numerical data
# todo: remove boilerplate code (would be mostly done by ^^^)
# todo: clean up argument names and docs


class PlotlyFig:
    def __init__(self, df=None, mode='offline', title=None, x_title=None,
                 y_title=None, colorbar_title='auto', x_scale='linear',
                 y_scale='linear', ticksize=25, fontscale=1, fontsize=25,
                 fontfamily='Courier', bgcolor="white", fontcolor=None,
                 colorscale='Viridis', height=None, width=None,
                 resolution_scale=None, margins=100, pad=0, username=None,
                 api_key=None, filename='temp-plot', show_offline_plot=True,
                 hovermode='closest', hoverinfo='x+y+text', hovercolor=None):
        """
        Class for making Plotly plots

        Args:

            Data:
                df (DataFrame): A pandas dataframe object which can be used to
                    generate several plots.
                mode: (str)
                    (i) 'offline': creates and saves plots on the local disk
                    (ii) 'notebook': to embed plots in IPython/Jupyter notebook,
                    (iii) 'online': save the plot in your online plotly account,
                    (iv) 'static': save a static image of the plot locally
                    NOTE: Both 'online' and 'static' modes require either
                    'username' and 'api_key' or Plotly credentials file.

            Axes:
                title: (str) title of plot
                x_title: (str) title of x-axis
                y_title: (str) title of y-axis
                colorbar_title (str or None): the colorbar (z) title. If set to
                    "auto" the name of the third column (if pd.Series) is chosen.
                x_scale: (str) Sets the x axis scaling type. Select from
                    'linear', 'log', 'date', 'category'.
                y_scale: (str) Sets the y axis scaling type. Select from
                    'linear', 'log', 'date', 'category'.
                ticksize: (int) size of ticks in px

            Fonts:
                fontscale (int/float): The relative scale of the font to the
                    rest of the plot
                fontsize: (int) size of text of plot title and axis titles
                fontfamily: (str) The HTML font family to use in browser - for
                    example, "Arial", or "Times New Roman". If multiple passed,
                    the list is an order of preference in case fonts are not
                    found on the system.

            Colors:
                bgcolor: (str) Sets the background color. For example, "grey".
                fontcolor: (str) Sets all font colors. For example, "black".
                colorscale: (str/list) Sets the colorscale (colormap). See
                    https://plot.ly/python/colorscales/ for details on what
                    data types are acceptable for color maps. String names
                    of colormaps can also be used, e.g., 'Jet' or 'Viridis'. A
                    useful list of Plotly builtins is: Greys, YlGnBu, Greens,
                    YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow,
                    Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis.

            Formatting:
                height: (float) output height (in pixels)
                width: (float) output width (in pixels)
                resolution_scale: (float) Increase the resolution of the image
                    by `scale` amount, eg: 3. Only valid for PNG and JPEG.
                margins (float or [float]): Specify the margin (in px) with a
                    list [top, bottom, right, left], or a number which will set
                    all margins.
                pad: (float) Sets the amount of padding (in px) between the
                    plotting area and the axis lines

            Plotly:
                username: (str) plotly account username
                api_key: (str) plotly account API key

            Offline:
                filename: (str) name/filepath of plot file
                show_offline_plot: (bool) automatically opens the plot offline

            Intreractivity:
                hovermode: (str) determines the mode of hover interactions. Can
                    be 'x'/'y'/'closest'/False
                hoverinfo: (str) Determines displayed information on mouseover.
                    Any combination of "x", "y", "z", "text", "name" with a "+"
                    OR "all" or "none" or "skip".
                    Examples: "x", "y", "x+y", "x+y+z", "all"
                hovercolor: (str) The color to set for the hover background.
                    If None, uses the trace color.

        Returns: None

        Attributes:
        These are either fields that Plotly's 'layout' cannot work with directly
        or are managerial values PlotlyFig uses separate from PlotlyDict.

            df (DataFrame): The dataframe which can be used to generate multiple
                plots.
            mode (str): The plot mode, specified above in the argument.
            show_offline_plot (bool): If True, opens up plot offline.
            username (str): The Plotly username
            api_key (str): The Plotly api key
            resolution_scale (int/float): Scale up the resolution of static
                images proportionally using this parameter.
            layout (dict): The dictionary passed to Plotly which specifies
                the PlotlyDict 'layout' value.
            font_style (dict): The general font style, in Plotly syntax.
            plot_counter (int): The number appended onto generated offline plots
            colorbar_title (str): The title of the colorbar
            colorscale (str): See argument documentation above.
            hoverinfo (str): See argument documentation above.
            ticksize (int): See argument documentation above.


        """

        # Plotly offline latex does not work (a Plotly issue).
        if mode == 'offline':
            for s in [title, x_title, y_title]:
                if s is not None and s.count('$') > 1:
                    warnings.warn("Plotly currently does not support LaTeX in"
                                  "offline plotting mode. To render LaTeX, please"
                                  "use Plotly online by setting changing the mode"
                                  "of PlotlyFig, or by clicking 'Export to Plotly'"
                                  "in the opened broswer window.")
            if fontfamily not in ['Courier', 'Times New Roman', 'Arial']:
                warnings.warn("The font family selected may not render "
                              "correctly in offline mode. To render more fonts,"
                              " use Plotly online by setting changing the mode"
                              "of PlotlyFig, or by clicking 'Export to Plotly' "
                              "in the opened broswer window.")


        # Fix fonts
        fontsize = float(fontsize) * float(fontscale)

        title = "" if title is None else title
        y_title = "" if y_title is None else y_title
        x_title = "" if x_title is None else x_title

        # Fix margins
        if not isinstance(margins, (list, tuple, np.ndarray)):
            margins = [margins] * 4
        margins = {'t': margins[0],
                   'b': margins[1] + ticksize + fontsize,
                   'r': margins[2],
                   'l': margins[3] + ticksize + fontsize,
                   'pad': pad}

        # kwargs which can be directly used in Plotly layout
        kwargs = {'y_scale': y_scale,
                  'x_scale': x_scale,
                  'font_size': fontsize,
                  'fontfamily': fontfamily,
                  'fontcolor': fontcolor,
                  'hovermode': hovermode,
                  'hovercolor': hovercolor,
                  'margin': margins,
                  'pad': pad,
                  'width': width,
                  'height': height,
                  'fontscale': fontscale,
                  'x_title': x_title,
                  'y_title': y_title,
                  'title': title,
                  'bgcolor': bgcolor}

        # Remove None entries to prevent Plotly silent errors
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

        # Set all attr dictionary entries which can be None
        pfkwargs = {'df': df,
                    'mode': mode,
                    'filename': filename,
                    'show_offline_plot': show_offline_plot,
                    'username': username,
                    'api_key': api_key,
                    'resolution_scale': resolution_scale,
                    'colorscale': colorscale,
                    'colorbar_title': colorbar_title,
                    'hoverinfo': hoverinfo,
                    'ticksize': ticksize,
                    }

        # Fix attributes of PlotlyFig object
        for k, v in pfkwargs.items():
            setattr(self, k, v)

        self.layout = {}
        font_style = {'size': fontsize, 'family': fontfamily}

        if fontcolor:
            font_style['color'] = fontcolor

        self.layout['titlefont'] = font_style
        self.layout['legend'] = {'font': font_style}
        self.layout['xaxis'] = {'title': x_title, 'type': x_scale,
                                'titlefont': font_style, 'tickfont': font_style}
        self.layout['yaxis'] = {'title': y_title, 'type': y_scale,
                                'titlefont': font_style, 'tickfont': font_style}
        self.layout['plot_bgcolor'] = bgcolor
        self.layout['paper_bgcolor'] = bgcolor
        self.layout['hoverlabel'] = {'font': font_style}
        self.layout['title'] = title

        if 'hovercolor' in kwargs:
            self.layout['hoverlabel']['bgcolor'] = hovercolor

        optional_fields = ['hovermode', 'margin', 'autosize', 'width', 'height',
                           'hoverinfo', 'ticksize']
        for k in optional_fields:
            if k in kwargs.keys():
                self.layout[k] = kwargs[k]

        self.plot_counter = 0
        self.font_style = font_style

    def set_arguments(self, **kwargs):
        """
        Method to modify some of the layout and PlotlyFig arguments after
        instantiation.

        Allowed arguments: title, x_title, y_title, colorbar_title, filename,
        mode, api_key, username, show_offline_plot

        Args:
            **kwargs: allowed variables to change are listed below:
        Returns: None
        """
        for kw in kwargs:
            if kw in ['x_title', 'y_title']:
                self.layout['{}axis'.format(kw[0])]['title'] = kwargs[kw]
            elif kw == 'title':
                self.layout[kw] = kwargs[kw]
            elif kw in ['filename', 'mode', 'api_key', 'username',
                        'show_offline_plot', 'colorbar_title']:
                setattr(self, kw, kwargs[kw])
                if kw in ['filename', 'mode']:
                    self.plot_counter = 0
            else:
                raise ValueError('changing "{}" is not supported!'.format(kw))

    def create_plot(self, fig, return_plot=False):
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
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

        Returns:
            A Plotly Figure object (if return_plot = True)

        """
        if return_plot:
            return fig
        if self.plot_counter > 0:
            filename = '{}_{}'.format(self.filename, self.plot_counter)
        else:
            filename = self.filename

        if self.mode in ['online', 'static']:
            plotly.tools.set_credentials_file(username=self.username,
                                              api_key=self.api_key)

            if not os.path.isfile('~/.plotly/.credentials'):
                if self.username is None:
                    raise ValueError(
                        'Field "username" must be filled in online and static '
                        'plotting modes.')
                if self.api_key is None:
                    raise ValueError(
                        'Field "api_key" must be filled in online and static'
                        'plotting modes.')

        if self.mode == 'offline':
            if not filename.endswith('.html'):
                filename += '.html'
            plotly.offline.plot(fig, filename=filename,
                                auto_open=self.show_offline_plot)
        elif self.mode == 'notebook':
            plotly.offline.init_notebook_mode()
            plotly.offline.iplot(fig)

        elif self.mode == 'online':
            if filename:
                plotly.plotly.plot(fig, filename=filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.mode == 'static':
            if 'height' in self.layout.keys():
                height = self.layout['height']
            else:
                height = 1080

            if 'width' in self.layout.keys():
                width = self.layout['width']
            else:
                width = 1920

            allowed_extensions = ('.png', '.svg', '.jpeg', '.pdf')
            if not self.filename or not self.filename.lower().endswith(
                    allowed_extensions):
                raise ValueError(
                    'field "filename" must be filled in static plotting '
                    'mode and must have an extension ending in '
                    '{}'.format(allowed_extensions))

            plotly.plotly.image.save_as(fig, filename=filename,
                                        height=height,
                                        width=width,
                                        scale=self.resolution_scale)

        self.plot_counter += 1

    def _data_from_str(self, col, data=None, is_color=False):
        """
        Try to get data based on column name in dataframe and return
        informative error if failed.

        Args:
            col (str): column name to look for
            data (pandas.DataFrame): if dataframe try to get col column from it
            is_color (bool): whether col could be used as a color in plotly
        Returns (pd.Series or col itself):
        """
        if isinstance(col, str):
            try:
                return data[col]
            except (KeyError, TypeError):
                if self.df is not None and col in self.df:
                    return self.df[col]
                elif is_color:
                    return col
                else:
                    raise ValueError('"{}" not in the data!'.format(col))
        else:
            return col

    def xy(self, xy_pairs, colors=None, color_range=None, labels=None,
           limits=None, names=None, sizes=None, modes='markers', markers=None,
           marker_scale=1.0, lines=None, colorscale=None, showlegends=None,
           error_bars=None, normalize_size=True, return_plot=False):
        """
        Make an XY scatter plot, either using arrays of values, or a dataframe.

        Args:
            xy_pairs (tuple or [tuple]): each tuple in the list of tuples
                is a trace on xy scatter plot. Each tuple contains a pair of
                x & y lists with the same length.
                example: ([1, 2], [2, 4]) # one trace, one tuple
                example: [([1,2,3], [2,4,6]), ([1,3], [2.5,5.5])] # 2 traces
                example: [(df['x1'], df['y1']), (df['x2'], df['y2'])]
                example: [('x1', 'y1'), ('x2', 'y2'), ('x1', 'y2')] # 3 traces
            colors (list or np.ndarray or pd.Series): set the colors for traces
                It can also be used to set the colors of the markers shown in
                the colorbar (list of numbers); overwrites marker['color'] and
                will override colorscales if trace colors are specified as
                strings.
                example: "red" # all traces and lines will be red
                example: 'GDP' or df['GDP'] # colorscale based on GDP (if
                    available in self.df or df respectively)
                example: ["green", "GDP"] # trace 1 is green and the markers of
                    trace 2 are colored based on GDP
            color_range ([min, max]): the range of numbers included in colorbar.
                if any number is outside of this range, it will be forced to
                either one. Note that if colorcol_range is set, the colorbar
                ticks will be updated to reflect -min or max+ at the two ends.
            labels (str or [str] or [list]): to set annotation for scatter
                points the same for all traces. Note that, several column
                names can be simultaneously used as labels but it is important
                to understand that when labels is set, it is assumed that all
                traces have the same length as the same labels are assigned to
                all traces (if there are more than one trace of course).
                    Examples:
                        labels = 'formula'
                        ['material_id', 'formula'] these 2 columns must be available
                        [['red', 'green', 'blue'], ['warm', 'mild', 'cold']] the
                        latter example assumes all xy traces have 3 points then
                        point one has ('red', 'warm') label, 2 has ('green', 'mild')
                        and finally point 3 ('blue', 'cold')
            limits (dict): The x and y limits defining the ranges the plot will
                show. Should be in the form {'x': (lower, higher), 'y': (lower,
                higher)}. Omit either key to prevent limits from being imposed
                on that axis.
            names (str or [str]): list of trace names used for legend. By
                default column name (or trace if NA) used if pd.Series passed
            sizes (str, float, [float], [list]). Options:
                str: column name in data with list of numbers used for marker size
                float: a single size used for all traces in xy_pairs
                [float]: list of fixed sizes used for traces (length==len(xy_pairs))
                [list]: list of list of sizes for each trace in xy_pairs
            modes (str or [str]): trace style; can be 'markers', 'lines' or
                'lines+markers'.
            markers (dict or [dict]): gives the ability to fine tune marker
                of each scatter plot individually if list of dicts passed. Note
                that the key "size" is forbidden in markers. Use sizes arg instead.
            lines (dict or [dict]: similar to markers though only if mode=='lines'
            showlegends (bool or [bool]): indicating whether to show legend
                for each trace (or simply turn it on/off for all if not list)
            error_bars ([str or list]): numbers used for error bars in the y
                direction. String input is interpreted as dataframe column name
            normalize_size (bool): if True, normalize the size list.
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

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
        elif len(names) != len(xy_pairs):
            raise ValueError('"names" must have the same length as "xy_pairs"')
        if sizes is None:
            sizes = [10 * marker_scale] * len(xy_pairs)
        elif isinstance(sizes, str):
            sizes = [self._data_from_str(sizes)] * len(xy_pairs)
        elif isinstance(sizes, (int, float)):
            sizes = [sizes]*len(xy_pairs)
        else:
            if len(sizes) != len(xy_pairs):
                raise ValueError(
                    '"sizes" must be the same length as "xy_pairs"')
            for i, _ in enumerate(sizes):
                sizes[i] = self._data_from_str(sizes[i])

        if error_bars is not None:
            if isinstance(error_bars, str):
                error_bars = [error_bars] * len(xy_pairs)
            error_bars = [self._data_from_str(ebar) for ebar in error_bars]
            if len(error_bars) != len(xy_pairs):
                raise ValueError(
                    '"error_bars" must be the same length as "xy_pairs"')

        if normalize_size:
            for i, size in enumerate(sizes):
                if isinstance(sizes[i], (list, np.ndarray, pd.Series)):
                    size = pd.Series(size).fillna(size.min())
                    sizes[i] = ((size - size.min()) / (
                            size.max() - size.min()) + 0.05) * 30 * marker_scale

        if isinstance(modes, str):
            modes = [modes] * len(xy_pairs)
        if len(modes) != len(xy_pairs):
            raise ValueError('"modes" and "xy_pairs" have different lengths!')

        data = []
        for pair in xy_pairs:
            if len(pair) != 2:
                raise ValueError('each xy within xy_pairs must have only 2 axes'
                                 ' : x and y (hence the "pair"); e.g. (x, y)')
            data.append((self._data_from_str(pair[0]),
                         self._data_from_str(pair[1])))
            if len(list(pair[0])) != len(list(pair[1])):
                warnings.warn('inequal number of points in x and y: part of the'
                              ' data not plotted!')
            if isinstance(pair[1], str):
                names.append(pair[1])
            else:
                try:
                    names.append(pair[1].name)
                except AttributeError:
                    names.append(None)

        showscale = [False] * len(data)
        colors_list = []
        colorbar = None
        if colors is not None:
            if isinstance(colors, str):
                colors = self._data_from_str(colors, is_color=True)
            if isinstance(colors, str):
                colors = [colors] * len(data)
            colorbar = colors
            if len(list(colors)) == len(data) and any([isinstance(c, str) for c in colors]):
                for itrace, color in enumerate(colors):
                    trace_color = self._data_from_str(color, is_color=True)
                    if not isinstance(trace_color, (str, int, float)) and \
                            len(trace_color) == len(data[itrace][0]) and \
                            not any([isinstance(c, str) for c in trace_color]):
                        colorbar = trace_color
                        showscale[itrace] = True
                        colors_list.append(trace_color)
                    else:
                        colors_list.append(trace_color)
            elif not any([isinstance(c, str) for c in colorbar]):
                showscale[0] = True # in this case all traces use one colorbar

            if not isinstance(colorbar, (list, np.ndarray, pd.Series)):
                raise ValueError('"colors" must be a dataframe column name, '
                                 'list, np.ndarray or pd.Series')
            if color_range:
                colorbar = pd.Series(colorbar)
                colorbar[colorbar < color_range[0]] = color_range[0]
                colorbar[colorbar > color_range[1]] = color_range[1]

        labels = self.setup_labels(labels=labels,
                                   data=None,
                                   expected_length=len(data[0][0]))

        markers = markers or [{'symbol': 'circle',
                               'line': {'width': 1,'color': 'black'}
                               } for _ in data]
        if isinstance(markers, dict):
            markers = [deepcopy(markers) for _ in data]

        if self.colorbar_title == 'auto':
            colorbar_title = pd.Series(colorbar).name
        else:
            colorbar_title = self.colorbar_title

        for im, _ in enumerate(markers):
            markers[im]['showscale'] = showscale[im]
            if markers[im].get('size', None) is None:
                markers[im]['size'] = sizes[im]
            else:
                raise ValueError('"size" must not be set in markers, '
                                 'use sizes argument instead')
            if colorbar is not None:
                if len(colors_list) > 0 and isinstance(colors_list[im], str):
                    markers[im]['color'] = colors_list[im]
                else:
                    markers[im]['color'] = colorbar
                fontd = {'family': self.font_style['family'],
                         'size': 0.75 * self.font_style['size']}
                if 'color' in self.font_style.keys():
                    fontd['color'] = self.font_style['color']
                markers[im]['colorbar'] = {'title': colorbar_title,
                                           'titleside': 'right',
                                           'tickfont': fontd,
                                           'titlefont': fontd}
                if color_range is not None:
                    tickvals = np.linspace(color_range[0], color_range[1], 6)
                    ticktext = [str(round(tick, 1)) for tick in tickvals]
                    ticktext[0] = '-' + ticktext[0]
                    ticktext[-1] = ticktext[-1] + '+'
                    markers[im]['colorbar']['tickvals'] = tickvals
                    markers[im]['colorbar']['ticktext'] = ticktext
            if markers[im].get('colorscale') is None:
                markers[im]['colorscale'] = colorscale or self.colorscale

        if not lines:
            lines = []
            for i, _ in enumerate(data):
                linedict = {'dash': 'solid', 'width': 2}
                lines.append(linedict)

        if isinstance(lines, dict):
            lines = [deepcopy(lines) for _ in data]

        # for var in [labels, markers, lines]:
        for var in [markers, lines]:
            if len(list(var)) != len(data):
                raise ValueError('"labels", "markers" or "lines" length does'
                                 ' not match with that of xy_pairs')
        layout = deepcopy(self.layout)

        traces = []

        for i, xy_pair in enumerate(data):
            traces.append(go.Scatter(x=xy_pair[0], y=xy_pair[1], mode=modes[i],
                                     marker=markers[i], line=lines[i],
                                     text=labels,
                                     hoverinfo=self.hoverinfo,
                                     hoverlabel=layout['hoverlabel'],
                                     name=names[i], showlegend=showlegends[i],
                                     ))
        if layout['xaxis'].get('title') is None and len(data) == 1:
            layout['xaxis']['title'] = pd.Series(data[0][0]).name
        if layout['yaxis'].get('title') is None and len(data) == 1:
            layout['yaxis']['title'] = pd.Series(data[0][1]).name

        if limits and 'x' in limits:
            layout['xaxis']['range'] = limits['x']
        if limits and 'y' in limits:
            layout['yaxis']['range'] = limits['y']

        if error_bars is not None:
            for i, _ in enumerate(traces):
                traces[i].error_y = {'type': 'data', 'array': error_bars[i],
                                     'visible': True}

        fig = {'data': traces, 'layout': layout}
        if any([show for show in showscale]):
            fig['layout']['legend']['x'] = 0.1
            fig['layout']['legend']['y'] = 1.1
            fig['layout']['legend']['orientation'] = 'h'
        return self.create_plot(fig, return_plot)


    def scatter_matrix(self, data=None, cols=None, colors=None, marker=None,
                       labels=None, marker_scale=1.0, return_plot=False,
                       default_color='#98AFC7', **kwargs):
        """
        Create a Plotly scatter matrix plot from dataframes using Plotly.
        Args:
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            cols ([str]): A list of strings specifying the columns of the
                dataframe to use.
            colors: (str) name of the column used for colorbar
            marker (dict): if size is set, it will override the automatic size
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).
            labels (see PlotlyFig.xy_plot documentation):
            default_color (str): default marker color. Ignored if colors is
                set. Histograms color is always set by this default_color.
            **kwargs: keyword arguments of scatterplot. Forbidden args are
                'size', 'color' and 'colorscale' in 'marker'. See example below
        Returns: a Plotly scatter matrix plot

        # Example for more control over markers:
        from matminer.figrecipes.plotly.make_plots import PlotlyFig
        from matminer.datasets.dataframe_loader import load_elastic_tensor
        df = load_elastic_tensor()
        pf = PlotlyFig()
        pf.scatter_matrix(df[['volume', 'G_VRH', 'K_VRH', 'poisson_ratio']],
                colorcol='poisson_ratio', text=df['material_id'],
                marker={'symbol': 'diamond', 'size': 8, 'line': {'width': 1,
                'color': 'black'}}, colormap='Viridis',
                title='Elastic Properties Scatter Matrix')
        """
        if 'colorscale' in kwargs.keys():
            kwargs['colormap'] = kwargs['colorscale']
            kwargs.pop('colorscale')

        height = 1000 if not 'height' in self.layout else self.layout['height']
        width = 1300 if not 'width' in self.layout else self.layout['width']

        # making sure the combination of input args make sense
        if data is None:
            if self.df is None:
                raise ValueError(
                    "scatter_matrix requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            elif cols is None:
                data = self.df
            else:
                data = self.df[cols]
        elif isinstance(data, (np.ndarray, list)):
            data = pd.DataFrame(data, columns=cols)

        data = data.select_dtypes(include=['float', 'int', 'bool'])
        labels = self._data_from_str(labels, data)
        if self.colorbar_title == 'auto':
            colors_ = self._data_from_str(colors, data)
            colorbar_title = pd.Series(colors_).name
        else:
            colorbar_title = self.colorbar_title

        # actual ploting:
        marker = marker or {'symbol': 'circle',
                            'line': {'width': 0.5, 'color': 'black'},
                            'colorbar': {'title': colorbar_title,
                                         'titleside': 'right',
                                         'tickfont': self.font_style,
                                         'titlefont': self.font_style}
                            }
        if colors is None:
            marker['color'] = default_color
            marker['showscale'] = False

        nplots = len(data.columns) - int(colors is not None)
        marker_size = marker.get('size') or 5.0 * marker_scale
        text_scale = 1.0 / (nplots ** 0.2)
        tick_scale = 0.9 / (nplots ** 0.3)
        fig = FF.create_scatterplotmatrix(data, index=colors, diag='histogram',
                                          size=marker_size, height=height,
                                          width=width, **kwargs)
        badf = ['xaxis', 'yaxis']
        layout = deepcopy(self.layout)
        scatter_layout = {k: v for (k, v) in layout.items() if k not in badf}
        fig.update({'layout': scatter_layout})

        # update each plot; we don't update the histograms markers as it causes issues:
        for iplot in range(nplots ** 2):
            fig['data'][iplot].update(hoverinfo=self.hoverinfo)
            for ax in ['x', 'y']:
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'] = \
                    self.font_style
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'] = \
                    self.font_style
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'][
                    'family'] = self.font_style['family']
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'][
                    'size'] = self.font_style['size'] * text_scale
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'family'] = self.font_style['family']
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'size'] = self.font_style['size'] * tick_scale
            if iplot % (nplots + 1) != 0:
                fig['data'][iplot].update(marker=marker, text=labels)
            else:
                fig['data'][iplot].update(marker={'color': default_color,
                                                  'line': {'width': 0.5,
                                                           'color': 'black'}})
        if (default_color in ['grey', 'black']) and colors is None:
            fig['layout']['hoverlabel']['font']['color'] = 'white'
        return self.create_plot(fig, return_plot)

    def histogram(self, data=None, cols=None, orientation="vertical",
                  histnorm="", n_bins=None, bins=None, colors=None,
                  bargap=0, return_plot=False):
        """
        Creates a Plotly histogram. If multiple series of data are available,
        will create an overlaid histogram.

        For n_bins, start, end, size, colors, and bargaps, all defaults are
        Plotly defaults.

        Args:
            data (DataFrame or list or [list]): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values or
                list of lists of numerical values.
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
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

        Returns:
            Plotly histogram figure.

        """

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

        dtypes = (list, np.ndarray, tuple)
        if cols is None:
            if isinstance(data, pd.Series):
                cols = [data.name]
                data = pd.DataFrame({cols[0]: data.tolist()})
            elif isinstance(data, pd.DataFrame):
                cols = data.columns.values
            elif isinstance(data[0], dtypes):
                data=pd.DataFrame({'trace{}'.format(i): pd.Series(data[i])
                                   for i, _ in enumerate(data)})
                cols = list(data.keys())
            else:
                data = {'trace1': data}
                cols = ['trace1']

        # Transform all entries to listlike, if given as single entries
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
            if bins[i] is None:
                bins[i] = {}
            else:
                if bins[i].get('size'):
                    if n_bins[i] is not None:
                        raise ValueError('Either set "n_bins" or "bins".')
                    if not bins[i].get('start') or not bins[i].get('end'):
                        warnings.warn('"size" key in bins ignored when "start" '
                                      'or "end" not specified')
                elif n_bins[i] is not None:
                    warnings.warn('"size" not specified in "bins", "n_bins" is '
                                  'ignored. Either set "bins" or "n_bins"')
                if bins[i].get('start') is None != bins[i].get('end') is None:
                    warnings.warn('Both "start" and "end" must be present in '
                                  'bins; otherwise, it is ignored.')

            d = data[col]
            if isinstance(d, np.ndarray):
                if len(d.shape) == 2:
                    d = d.reshape((len(d),))

            if orientation == 'vertical':
                h = go.Histogram(x=d, histnorm=histnorm,
                                 xbins=bins[i],
                                 nbinsx=n_bins[i],
                                 marker=dict(color=colors[i]),
                                 name=col,
                                 hoverinfo=self.hoverinfo,
                                 hoverlabel=self.layout['hoverlabel'])

            elif orientation == 'horizontal':
                h = go.Histogram(y=d, histnorm=histnorm,
                                 ybins=bins[i],
                                 nbinsy=n_bins[i],
                                 marker=dict(color=colors[i]),
                                 name=col,
                                 hoverinfo=self.hoverinfo,
                                 hoverlabel=self.layout['hoverlabel']
                                 )
            else:
                raise ValueError(
                    "The orientation must be 'horizontal' or 'vertical'.")
            hgrams.append(h)

        layout = deepcopy(self.layout)

        layout['hovermode'] = 'x' if orientation == 'vertical' else 'y'
        layout['bargap'] = bargap

        if orientation == 'vertical':
            if not layout['yaxis']['title']:
                layout['yaxis']['title'] = histnorm
        elif orientation == 'horizontal':
            if not layout['xaxis']['title']:
                layout['xaxis']['title'] = histnorm

        if len(hgrams) > 1:
            layout['barmode'] = 'overlay'
            for h in hgrams:
                h['opacity'] = 1.0 / float(len(hgrams)) + 0.2

        fig = {'data': hgrams, 'layout': layout}
        return self.create_plot(fig, return_plot)

    def bar(self, data=None, cols=None, x=None, y=None, labels=None,
            barmode='group', colors=None, bargap=None, return_plot=False):
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
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

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
            if len(x) != len(y):
                raise ValueError('x and y must be of the same length')
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
        for i, _ in enumerate(x):
            barplot = go.Bar(x=x[i], y=y[i], name=labels[i],
                             marker=dict(color=colors[i]),
                             hoverinfo=self.hoverinfo)
            barplots.append(barplot)

        layout = deepcopy(self.layout)
        # Prevent linear default from altering categorical bar plot
        layout['xaxis']['type'] = None
        layout['barmode'] = barmode
        layout['bargap'] = bargap
        fig = {'data': barplots, 'layout': layout}
        return self.create_plot(fig, return_plot)

    def violin(self, data=None, cols=None, use_colorscale=False, rugplot=False,
               group_col=None, groups=None, colorscale=None, return_plot=False):
        """
        Create a violin plot using Plotly.

        Args:
            data: (DataFrame/list) A dataframe containing at least one
                numerical column. Also accepts lists/arrays of numerical
                values, using columns as separate variables (distributions are
                down rows). If None, uses the dataframe passed into the
                constructor.
            cols: ([str]) The labels for the columns of the dataframe to be
                included in the plot. If data is passed as a list/array, pass
                a list of cols to be used as labels for the violins.
            rugplot: (bool) If True, plots the distribution of the data next
                to the violin with a 'rugplot'.
            group_col: (str) Name of the column containing the group for each
                row, if it exists. Used only if there is one entry in cols.
            groups: ([str]): All group names to be included in the violin plot.
                Used only if there is one entry in cols.
            colorscale: (str/tuple/list/dict) either a plotly scale name (Greys,
                YlGnBu, Greens, etc.), an rgb or hex color, a color tuple, a
                list/dict of colors. The color is representative of the median
                value of the violin.
            use_colorscale: (bool) Only applicable if grouping by another
                variable. Will implement a colorscale based on the first 2
                colors of param colors. This means colors must be a list with
                at least 2 colors in it (Plotly colorscales are accepted since
                they map to a list of two rgb colors)
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

        Returns: A Plotly violin plot Figure object.

        """
        if data is None:
            if self.df is None:
                raise ValueError(
                    "Violin plot requires either dataframe labels and a "
                    "dataframe or a list of numerical values.")
            data = self.df

        # No matter the data type, data is converted into a dataframe

        if isinstance(data, pd.Series):
            cols = [data.name]
            data = pd.DataFrame({data.name: data.tolist()})
        elif isinstance(data, (list, np.ndarray)):

            data = np.array(data)
            if data.shape[0] == 1:
                data = np.array([data])

            if not cols:
                cols = ['data{}'.format(i) for i in range(data.shape[1])]
            data = pd.DataFrame(data=data, columns=cols)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError('"data" was set to an unknown datatype. Please'
                             'use a list, a list of lists/numpy array, a series'
                             ' or a DataFrame.')

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
                        'in the dataframe.'.format(j))

        if use_colorscale:
            colorscale = colorscale or self.colorscale
        else:
            colorscale = ['rgb(105,105,105)'] * len(data)

        fig = FF.create_violin(data=data, data_header=cols[0],
                               group_header=group_col,
                               title=self.layout['title'],
                               colors=colorscale, use_colorscale=use_colorscale,
                               group_stats=group_stats, rugplot=rugplot)
        layout = deepcopy(self.layout)
        font_style = deepcopy(self.font_style)
        font_style['size'] = 0.65 * font_style['size']

        violin_layout = {k: v for (k, v) in layout.items() if k != 'xaxis'}
        if 'color' in violin_layout['hoverlabel']['font']:
            violin_layout['hoverlabel']['font'].pop('color')

        if 'bgcolor' in violin_layout['hoverlabel']:
            violin_layout['hoverlabel'].pop('bgcolor')

        fig.update({'layout': violin_layout})

        # Change sizes in all x-axis
        for item in fig['layout']:
            if item.startswith('xaxis'):
                fig['layout'][item].update({'titlefont': self.font_style,
                                            'tickfont': self.font_style})

        if not hasattr(self, 'width'):
            fig['layout']['width'] = 1400
        if not hasattr(self, 'height'):
            fig['layout']['height'] = 1000

        fig['layout']['font'] = font_style
        return self.create_plot(fig, return_plot)

    def parallel_coordinates(self, data=None, cols=None, line=None, precision=2,
                             colors=None, return_plot=False):
        """
        Create a Plotly Parcoords plot from dataframes.

        Args:
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            cols ([str]): A list of strings specifying the columns of the
                dataframe to use.
            colors (str): The name of the column to use for the color bar.
            line (dict): plotly line dict with keys such as "color" or "width"
            precision (int): the number of floating points for columns with
                float data type (2 is recommended for a nice visualization)
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).
        Returns:
            a Plotly parallel coordinates plot.
        """
        # making sure the combination of input args make sense
        if data is None:
            if self.df is None:
                raise ValueError(
                    "Parallel coordinates requires either dataframe labels and a"
                    "dataframe or a list of numerical values.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif isinstance(data, (np.ndarray, list)):
            data = pd.DataFrame(data, columns=cols)

        if cols is None:
            cols = data.columns.values

        if colors is None:
            colors = 'blue'
        else:
            colors = self._data_from_str(colors, data)
        if self.colorbar_title == 'auto':
            colorbar_title = pd.Series(colors).name
        else:
            colorbar_title = self.colorbar_title

        cols = list(cols)
        if pd.Series(colors).name in cols:
            cols.remove(pd.Series(colors).name)

        dimensions = []
        for col in cols:
            if is_numeric_dtype(data[col]) and 'int' not in str(
                    data[col].dtype):
                values = data[col].apply(lambda x: round(x, precision))
            else:
                values = data[col]
            dimensions.append({'label': col, 'values': values.tolist()})

        font_style = deepcopy(self.font_style)
        font_style['size'] = 0.7 * font_style['size']
        line = line or {'color': colors,
                        'colorscale': self.colorscale,
                        'colorbar': {'title': colorbar_title,
                                     'titleside': 'right',
                                     'tickfont': font_style,
                                     'titlefont': font_style,
                                     }}
        par_coords = go.Parcoords(line=line, dimensions=dimensions,
                                  hoverinfo=self.hoverinfo)

        par_coords.tickfont = font_style
        par_coords.labelfont = font_style
        par_coords.rangefont = font_style
        fig = {'data': [par_coords], 'layout': self.layout}
        return self.create_plot(fig, return_plot)

    def heatmap_basic(self, data=None, x_labels=None, y_labels=None,
                      colorscale=None, colorscale_range=None,
                      annotations_text=None, annotations_font_size=20,
                      annotations_color='white', return_plot=False):
        """
        Make a heatmap plot, either using 2D arrays of values, or a dataframe.

        Args:
            data: (array) an array of arrays. For example, in case of a pandas
                dataframe 'df', data=df.values.tolist(). If None, uses the data
                frame passed into the constructor.
            x_labels: (array) an array of strings to label the heatmap columns
            y_labels: (array) an array of strings to label the heatmap rows
            colorscale (str/array): See colorscale in __init__.
            colorscale_range: (array) Sets the minimum (first array item) and
                maximum value (second array item) of the colorscale.
            annotations_text: (array) an array of arrays, with each value being
                a string annotation to the corresponding value in 'data'
            annotations_font_size: (int) size of annotation text
            annotations_color: (str/array) color of annotation text - accepts
                similar formats as other color variables

        Returns: A Plotly heatmap plot Figure object.
        """

        if not data:
            if not self.df:
                raise ValueError("Either data or self.df (set in initializer)"
                                 "must be defined.")
            else:
                data = self.df.as_matrix()
                x_labels = self.df.columns.values.tolist()
                y_labels = self.df.index.values.tolist()

        if not colorscale:
            colorscale = self.colorscale

        if not colorscale_range:
            colorscale_min = None
            colorscale_max = None
        elif len(colorscale_range) == 2:
            colorscale_min = colorscale_range[0]
            colorscale_max = colorscale_range[1]
        else:
            raise ValueError(
                "The field 'colorscale_range' must be a list with two values.")

        font_family = self.font_style['family']
        font_size = self.font_style['size']

        if annotations_text:
            annotations = []

            for n, row in enumerate(data):
                for m, _ in enumerate(row):
                    var = annotations_text[n][m]
                    annotations.append(
                        dict(
                            text=str(var),
                            x=x_labels[m], y=y_labels[n],
                            xref='x1', yref='y1',
                            font=dict(color=annotations_color,
                                      size=annotations_font_size,
                                      family=font_family),
                            showarrow=False)
                    )
        else:
            annotations = []

        if self.colorbar_title == 'auto':
            colorbar_title = pd.Series(data).name
        else:
            colorbar_title = self.colorbar_title

        data = go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            zmin=colorscale_min,
            zmax=colorscale_max,
            colorscale=colorscale or self.colorscale,
            hoverinfo=self.hoverinfo,
            colorbar={
                'title': colorbar_title, 'titleside': 'right',
                'tickfont': {'size': 0.75 * self.ticksize,
                             'family': font_family},
                'titlefont': {'size': font_size,
                              'family': font_family}}
        )

        layout = deepcopy(self.layout)

        # heatmap specific formatting:
        for ax in ['x', 'y']:
            if 'type' in layout['{}axis'.format(ax)]:
                layout['{}axis'.format(ax)].pop('type')
        layout['annotations'] = annotations
        if 'bgcolor' not in layout['hoverlabel']:
            layout['hoverlabel']['bgcolor'] = 'white'
        fig = {'data': [data], 'layout': layout}

        return self.create_plot(fig, return_plot)


    def heatmap_df(self, data=None, cols=None, x_labels=None, x_nqs=6,
                   y_labels=None, y_nqs=4, precision=1, annotation='count',
                   annotation_color='black', colorscale=None, color_range=None,
                   return_plot=False):
        """
        A heatmap which can accept a dataframe as input directly.

        Args:
            data: (dataframe): only the first 3 numerical columns considered
            cols ([str]): A list of strings specifying the columns of the
                dataframe (either data or self.df) to use. Currenly, only 3
                columns is supported. Note that the order in cols matter, the
                first is considered x, second y and the third as z (color)
            x_labels ([str]): labels for the categories in x data (first column)
            x_nqs (int or None): if unique values for x_prop is more than this,
                x_prop is divided into x_nqs quantiles for better presentation
                *if x_labels is set, x_nqs ignored (i.e. x_nqs = len(x_labels))
            y_labels ([str]): similar to x_labels but for the 2nd column in data
            y_nqs (int or None): similar to x_nqs but for the 2nd column in data
            precision (int): number of floating points used for binning/display
            annotation (str or None): mode of annotation. Options are:
                None: no annotations
                "count": the number of data available in each cell displayed
                "value": the actual value of the cell in addition to colorbar
            annotation_color (str): the color of annotation (text inside cells)
            colorscale: see the __init__ doc for colorscale
            color_range ([min, max]): the range of numbers included in colorbar.
                if any number is outside of this range, it will be forced to
                either one. Note that if colorcol_range is set, the colorbar
                ticks will be updated to reflect -min or max+ at the two ends.
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).

        Returns: A Plotly heatmap plot Figure object.
        """
        if data is None:
            if self.df is None:
                raise ValueError("heatmap_df requires dataframe input.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif not isinstance(data, pd.DataFrame):
            raise ValueError('"heatmap_df" only supports dataframes with '
                             'numerical columns. Please use heatmap instead.')
        cols = data.columns.values
        x_prop = cols[0]
        y_prop = cols[1]
        col_prop = cols[2]
        if x_labels is not None:
            x_nqs = len(x_labels)
        if y_labels is not None:
            y_nqs = len(y_labels)

        data = data.sort_values(y_prop, ascending=True)
        if y_nqs is None or len(data[y_prop].unique()) > y_nqs:
            try:
                data['y_bin'] = pd.qcut(data[y_prop], y_nqs, labels=y_labels,
                                        precision=precision).astype(str)
                y_groups = data['y_bin'].unique()
            except BaseException:
                warnings.warn('pd.qcut failed! categorizing on unique values')
                y_groups = data[y_prop].unique()
                data['y_bin'] = data[y_prop]
        else:
            y_groups = data[y_prop].unique()
            data['y_bin'] = data[y_prop]

        data = data.sort_values(x_prop, ascending=True)
        if x_nqs is None or len(data[x_prop].unique()) > x_nqs:
            try:
                data['x_bin'] = pd.qcut(data[x_prop], x_nqs, labels=x_labels,
                                        precision=precision).astype(str)
                x_groups = data['x_bin'].unique()
            except BaseException:
                warnings.warn('pd.qcut failed! categorizing on unique values')
                x_groups = data[x_prop].unique()
                data['x_bin'] = data[x_prop]
        else:
            x_groups = data[x_prop].unique()
            data['x_bin'] = data[x_prop]

        data_ = []
        annotations = []
        annotation_template = {'font': {'color': annotation_color,
                                        'size': 0.7 * self.font_style['size'],
                                        'family': self.font_style['family']},
                               'showarrow': False}
        for y in y_groups:
            temp = data[data['y_bin'].values == y]
            grpd = temp.groupby('x_bin').mean().reset_index()
            gcnt = temp.groupby('x_bin').count().reset_index()
            x_data = []
            for x in x_groups:
                if x in grpd['x_bin'].values:
                    val = grpd[grpd['x_bin'].values == x][col_prop].values[0]
                    count = gcnt[gcnt['x_bin'].values == x][col_prop].values[0]
                    val = str(round(val, precision))
                else:
                    count = 0
                    val = 'N/A'
                x_data.append(val)
                a_d = deepcopy(annotation_template)
                a_d['x'] = x
                a_d['y'] = y
                if annotation is None:
                    a_d['text'] = ''
                elif annotation == 'value':
                    a_d['text'] = val
                elif annotation == 'count':
                    a_d['text'] = count
                else:
                    a_d['text'] = annotation
                annotations.append(a_d)
            data_.append(x_data)

        x_labels = x_labels or x_groups
        y_labels = y_labels or y_groups

        if self.colorbar_title == 'auto':
            colorbar_title = col_prop
        else:
            colorbar_title = self.colorbar_title

        zmin = None
        zmax = None
        if color_range is not None:
            zmin = color_range[0]
            zmax = color_range[1]
        trace = go.Heatmap(z=data_, x=x_labels, y=y_labels, zmin=zmin, zmax=zmax,
                           colorscale=colorscale or self.colorscale,
                           hoverinfo=self.hoverinfo, colorbar={
                'title': colorbar_title, 'titleside': 'right',
                'tickfont': {'size': 0.75 * self.ticksize,
                             'family': self.font_style['family']},
                'titlefont': {'size': self.font_style['size'],
                              'family': self.font_style['family']}
            })

        layout = deepcopy(self.layout)

        # heatmap specific formatting:
        for ax in ['x', 'y']:
            if 'type' in layout['{}axis'.format(ax)]:
                layout['{}axis'.format(ax)].pop('type')
        layout['margin']['l'] += self.ticksize * \
                                 (2 + precision / 10.0) + 35
        if not layout['xaxis'].get('title'):
            warnings.warn('xaxis title was automatically set to x_prop value')
            layout['xaxis']['title'] = x_prop
        if not layout['yaxis'].get('title'):
            warnings.warn('yaxis title was automatically set to y_prop value')
            layout['yaxis']['title'] = y_prop
        layout['annotations'] = annotations
        if 'bgcolor' not in layout['hoverlabel']:
            layout['hoverlabel']['bgcolor'] = 'white'
        fig = {'data': [trace], 'layout': layout}
        return self.create_plot(fig, return_plot)


    def setup_labels(self, labels, data, expected_length=None):
        """
        Set the input labels to the appropriate format to support labeling of
            each data point with one or multiple labels that shows upon hovering
            over the point (Plotly default behavior).
        Args:
            labels (str or [str] or [list]): see the docs for labels in xy
            data (DataFrame or list): A dataframe containing at least
                one numerical column. Also accepts lists of numerical values.
                If None, uses the dataframe passed into the constructor.
            expected_length (int): the expected length of the rows/labels. This
                is len(data) if data is dataframe and length of axes

        Returns ([list]): list of labels each with the expected length

        """
        expected_length = expected_length or len(data)
        if labels is None:
            pass
        elif not isinstance(labels, (list, np.ndarray, pd.Series, pd.Index)):
            labels = [self._data_from_str(labels, data)]
        else:
            if len(list(labels)) ==expected_length and \
                    isinstance(labels[0], (str, tuple)):
                labels = [labels]
            else:
                labels = [self._data_from_str(l) for l in labels]

        # here, labels is expected to be a [list] each list w/ expected_length
        if labels is not None:
            for label in labels:
                if len(list(label)) != expected_length:
                    raise ValueError('the length of this label is not equal to '
                                     'the expected length:\n{}'.format(label))
            labels = ['</br>'.join([str(t) for t in l]) for l in zip(*labels)]
        return labels


    def triangle(self, data=None, cols=None, sum_of_3=1.0, axes_titles=None,
                 labels=None, markers=None, return_plot=False):
        """
        Phase diagram type plot for 3 (and only 3) variables that always add
        to a certain number (e.g. 1 or 100%); regardless the rows are separately
        normalized inside plotly so that they add to 1 as otherwise, triangle
        plot does not make sense.

        Args:
            data: (dataframe): if not set, self.df is used
            cols ([str]): A list of strings specifying the 3 columns of the
                dataframe (either data or self.df) to plot the triangle plot
                for. Note that the order of 3 axes is decided based on the
                order of cols.
            sum_of_3 (int/float): scale the sum of cols to this number.
            axes_titles ([str]): titles of the 3 axes, this overrides the
                dataframe column names. Note that if set, axes_titles must be
                of the length 3. Examples:
                    ['A', 'B', 'X']
                    ['title 1', '', 'title 2] (i.e. no title for the 2nd axis)
            labels (str or [str] or [list]): to set annotation for scatter
                points the same for all traces. Note that, several column
                names can be simultaneously used as labels but it is important
                to understand that when labels is set, it is assumed that all
                traces have the same length as the same labels are assigned to
            markers (None or dict): plotly marker dict with keys such as size,
                symbol, color, line, etc
            return_plot (bool): Returns the dictionary representation of the
                figure if True.

        Returns: A Plotly triangle plot Figure object.
        """
        #TODO: on a big monitor, if the window size is expanded, the ticks fly to the corner of the screen, see why?! and fix
        #TODO: add sizes and colors to follow the same logic in xy (add a setup_sizes similar to setup_labels)
        #TODO: add sizes and colors to follow the same logic in xy (add a setup_colors similar to setup_labels)
        if data is None:
            if self.df is None:
                raise ValueError("triangle requires dataframe input.")
            elif cols is None:
                data = self.df.select_dtypes(include=['float', 'int', 'bool'])
            else:
                data = self.df[cols]
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if cols is None:
            cols = data.columns.values[:3]
        if len(list(cols)) != 3:
            raise ValueError('triangle plot requires 3 and only 3 columns!')

        labels = self.setup_labels(labels, data)

        data = {
            'type': 'scatterternary',
            'mode': 'markers',
            'a': list(data[cols[0]]), # list() to ensure JSON serializable
            'b': list(data[cols[1]]),
            'c': list(data[cols[2]]),
            'text': labels,
            'marker': markers or {
                'symbol': 'circle-open',
                'color': '#DB7365',
                'size': 14,
                'line': {'width': 2}
            }
        }

        axes_titles = axes_titles or cols

        axes = []
        angles = [0, 45, -45]
        for iax in range(3):
            ax = deepcopy(self.layout['xaxis'])
            ax.pop('type', None)
            ax['title'] = axes_titles[iax]
            if iax > 0:
                ax['title'] = '<br>'+ax['title'] # not to overlap w/ axis ticks
            ax['showline'] = True
            ax['showgrid'] = True
            ax['tickangle'] = angles[iax]
            axes.append(ax)

        layout = deepcopy(self.layout)
        layout['ternary'] = {'sum': sum_of_3,
                             'aaxis': axes[0],
                             'baxis': axes[1],
                             'caxis': axes[2]
                             }
        fig = {'data': [data], 'layout': layout}

        return self.create_plot(fig, return_plot)

    # TODO: implement pyramid after finishing triangle (i.e. after labels, sizes and colors are all supported in triangle)
    # def pyramid(self, data=None, cols=None, sizes=None, labels=None,
    #              colors=None, lines=True, normalize=True):
    #     if data is None:
    #         if self.df is None:
    #             raise ValueError("pyramid requires dataframe input.")
    #         elif cols is None:
    #             data = self.df.select_dtypes(include=['float', 'int', 'bool'])
    #         else:
    #             data = self.df[cols]
    #     elif not isinstance(data, pd.DataFrame):
    #         data = pd.DataFrame(data)
    #     cols = data.columns.values
    #
    #     scaler = MinMaxScaler()
    #     for col in cols:
    #         data[col] = scaler.fit_transform(data[col])
    #
    #     if normalize: # TODO: maybe enforce this? because if not normalized, this plot doesn't make much sense
    #         sums = data[cols].as_matrix().sum(axis=1)
    #         for col in data:
    #             data[col] /= sums