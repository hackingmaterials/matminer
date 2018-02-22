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
              'Alireza Faghaninia  <alireza.faghaninia@gmail.com>'


# todo: common function for if then checking data types + automatically ignore non-numerical data
# todo: add tests
# todo: nuke *_plot methods?

class PlotlyFig:
    def __init__(self, df=None, mode='offline', title=None, x_title=None,
                 y_title=None, colorbar_title='auto', x_scale='linear',
                 y_scale='linear', tick_size=25, font_scale=1, font_size=25,
                 font_family='Courier', bg_color="white", font_color='black',
                 colorscale='Viridis', height=None, width=None,
                 resolution_scale=None, margins=100, pad=0, username=None,
                 api_key=None, filename='auto', show_offline_plot=True,
                 hovermode='closest', hoverinfo='x+y+text'):
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
                tick_size: (int) size of ticks in px

            Fonts:
                font_scale (int/float): The relative scale of the font to the
                    rest of the plot
                font_size: (int) size of text of plot title and axis titles
                font_family: (str) The HTML font family to use in browser - for
                    example, "Arial", or "Times New Roman". If multiple passed,
                    the list is an order of preference in case fonts are not
                    found on the system.

            Colors:
                bg_color: (str) Sets the background color. For example, "grey".
                font_color: (str) Sets all font colors. For example, "black".
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

        Returns: None

        """

        # All class attributes are set in init. Only self.layout can change.

        # Fix fonts
        font_size = float(font_size) * float(font_scale)

        title = "" if title is None else title
        y_title = "" if y_title is None else y_title
        x_title = "" if x_title is None else x_title

        # Fix margins
        if not isinstance(margins, (list, tuple, np.ndarray)):
            margins = [margins] * 4
        margins = {'t': margins[0],
                   'b': margins[1] + tick_size + font_size,
                   'r': margins[2],
                   'l': margins[3] + tick_size + font_size,
                   'pad': pad}

        kwargs = {'y_scale': y_scale,
                  'x_scale': x_scale,
                  'colorbar_title': colorbar_title,
                  'font_size': font_size,
                  'font_family': font_family,
                  'font_color': font_color,
                  'hoverinfo': hoverinfo,
                  'hovermode': hovermode,
                  'margin': margins,
                  'pad': pad,
                  'width': width,
                  'height': height,
                  'colorscale': colorscale,
                  'font_scale': font_scale,
                  'tick_size': tick_size,
                  'x_title': x_title,
                  'y_title': y_title,
                  'title': title,
                  'bg_color': bg_color}

        # Remove None entries to prevent Plotly silent errors
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

        # Set all attr dictionary entries which can be None
        pfkwargs = {'df': df,
                    'mode': mode,
                    'filename': filename,
                    'show_offline_plot': show_offline_plot,
                    'username': username,
                    'api_key': api_key,
                    'resolution_scale': resolution_scale}
        kwargs.update(pfkwargs)

        # Fix attributes of PlotlyFig object
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.layout = {}
        font_style = {'size': self.font_size, 'family': self.font_family,
                      'color': self.font_color}
        self.layout['titlefont'] = font_style
        self.layout['legend'] = {'font': font_style}
        self.layout['xaxis'] = {'title': self.x_title, 'type': self.x_scale,
                                'titlefont': font_style, 'tickfont': font_style}
        self.layout['yaxis'] = {'title': self.y_title, 'type': self.y_scale,
                                'titlefont': font_style, 'tickfont': font_style}
        self.layout['plot_bgcolor'] = self.bg_color
        self.layout['paper_bgcolor'] = self.bg_color
        self.layout['hoverlabel'] = {'font': font_style}
        self.layout['title'] = self.title

        optional_fields = ['hovermode', 'margin', 'autosize', 'width', 'height']
        for k in optional_fields:
            if k in kwargs.keys():
                self.layout[k] = kwargs[k]

        if self.mode in ['online', 'static']:
            if not os.path.isfile('~/.plotly/.credentials'):
                if 'username' not in kwargs.keys():
                    raise ValueError(
                        'Field "username" must be filled in online and static '
                        'plotting modes.')
                if 'api_key' not in kwargs.keys():
                    raise ValueError(
                        'Field "api_key" must be filled in online and static'
                        'plotting modes.')
                plotly.tools.set_credentials_file(username=self.username,
                                                  api_key=self.api_key)

        if self.mode == 'static':
            if not self.filename or not self.filename.lower().endswith(
                    ('.png', '.svg', '.jpeg', '.pdf')):
                raise ValueError(
                    'field "filename" must be filled in static plotting mode '
                    'and must have an extension ending in ('
                    '".png", ".svg", ".jpeg", ".pdf")')

        self.plot_counter = 1
        self.font_style = font_style

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
            A Plotly Figure object (if self.plot_mode = 'return')

        """
        if return_plot:
            return fig

        if self.filename == 'auto':
            filename = 'auto_{}'.format(self.plot_counter)
        else:
            filename = self.filename

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
            plotly.plotly.image.save_as(fig, filename=filename,
                                        height=self.height, width=self.width,
                                        scale=self.resolution_scale)
        self.plot_counter += 1


    def data_from_col(self, col, data=None):
        """
        try to get data based on column name in dataframe and return
            informative error if failed.
        Args:
            col (str): column name to look for
            data (pandas.DataFrame): if dataframe try to get col column from it
            allow_str (bool): if True: pass col even if not in data or self.df
        Returns (pd.Series or col itself):
        """
        if isinstance(col, str):
            try:
                return data[col]
            except:
                if col in self.df:
                    return self.df[col]
                else:
                    raise ValueError('"{}" not in the data!'.format(col))
        else:
            return col


    def xy(self, xy_pairs, colors=None, color_range=None, labels=None,
           names=None, sizes=None, modes='markers', markers=None,
           marker_scale=1.0, lines=None, colorscale=None, showlegends=None,
           error_bars = None, normalize_size=True, return_plot=False):
        """
        Make an XY scatter plot, either using arrays of values, or a dataframe.

        Args:
            xy_pairs (tuple or [tuple]): x & y columns of scatter plots
                with possibly different lengths are extracted from this arg
                example: ([1, 2], [3, 4])
                example: [(df['x1'], df['y1']), (df['x2'], df['y2'])]
                example: [('x1', 'y1'), ('x2', 'y2')]
            colors (list or np.ndarray or pd.Series): set the colorscale for
                the colorbar (list of numbers); overwrites marker['color']
            color_range ([min, max]): the range of numbers included in colorbar.
                if any number is outside of this range, it will be forced to
                either one. Note that if colorcol_range is set, the colorbar ticks
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
            modes (str or [str]): trace style; can be 'markers', 'lines' or
                'lines+markers'.
            markers (dict or [dict]): gives the ability to fine tune marker
                of each scatter plot individually if list of dicts passed. Note
                that the key "size" is forbidden in markers. Use sizes arg instead.
            lines (dict or [dict]: similar to markers though only if mode=='lines'
            colorscale (str):  see the colorscale doc in __init__
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
        else:
            assert len(names) == len(xy_pairs)
        if sizes is None:
            sizes = [10 * marker_scale] * len(xy_pairs)
        elif isinstance(sizes, str):
            sizes = [self.data_from_col(sizes)] * len(xy_pairs)
        else:
            if len(sizes) != len(xy_pairs):
                raise ValueError(
                    '"sizes" must be the same length as "xy_pairs"')
            for i, _ in enumerate(sizes):
                sizes[i] = self.data_from_col(sizes[i])

        if error_bars is not None:
            if isinstance(error_bars, str):
                error_bars = [error_bars] * len(xy_pairs)
            error_bars = [self.data_from_col(ebar) for ebar in error_bars]
            if len(error_bars) != len(xy_pairs):
                raise ValueError('"error_bars" must be the same length as "xy_pairs"')

        if normalize_size:
            for i, size in enumerate(sizes):
                if isinstance(sizes[i], (list, np.ndarray, pd.Series)):
                    size = pd.Series(size).fillna(size.min())
                    sizes[i] = ((size - size.min()) / (
                                size.max() - size.min()) + 0.05) * 30 * marker_scale
                    print(sizes[i])

        if isinstance(modes, str):
            modes = [modes] * len(xy_pairs)
        else:
            assert len(modes) == len(xy_pairs)
        if colors is None:
            showscale = False
            colorbar = None
        else:
            showscale = True
            colorbar = self.data_from_col(colors)
            assert isinstance(colorbar, (list, np.ndarray, pd.Series))
            if color_range:
                colorbar = pd.Series(colorbar)
                colorbar[colorbar < color_range[0]] = color_range[0]
                colorbar[colorbar > color_range[1]] = color_range[1]
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

        if self.colorbar_title == 'auto':
            colorbar_title = pd.Series(colorbar).name
        else:
            colorbar_title = self.colorbar_title

        for im, _ in enumerate(markers):
            markers[im]['showscale'] = showscale
            if markers[im].get('size', None) is None:
                markers[im]['size'] = sizes[im]
            else:
                raise ValueError(
                    '"size" must not be set in markers, use sizes argument instead')
            if colorbar is not None:
                markers[im]['color'] = colorbar
                fontd = {'family': self.font_family,
                         'size': 0.75 * self.tick_size}
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

        if error_bars is not None:
            for i, _ in enumerate(traces):
                traces[i].error_y = {'type': 'data', 'array':error_bars[i],
                                     'visible': True}

        fig = {'data': traces, 'layout': layout}
        if showscale:
            fig['layout']['legend']['x'] = 0.9
        return self.create_plot(fig, return_plot)

    def heatmap(self, data=None, cols=None, x_labels=None, x_nqs=6,
                y_labels=None, y_nqs=4, precision=1, annotation='count',
                annotation_color='black', colorscale=None, return_plot=False):
        #todo: Stuff that I think would be good to see in heatmap - alex
        #todo: 1. Ability to take in x_label, y_label, and a matrix like heatmap_plot: I vote for keeping heatmap_plot

        """
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
            return_plot (bool): Returns the dictionary representation of the
                figure if True. If False, prints according to self.mode (set
                with mode in __init__).
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
        elif not isinstance(data, pd.DataFrame):
            raise ValueError('"heatmap" only supports dataframes with numerical'
                             ' columns. Please use heatmap_plot instead.')

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
            except:
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
            except:
                warnings.warn('pd.qcut failed! categorizing on unique values')
                x_groups = data[x_prop].unique()
                data['x_bin'] = data[x_prop]
        else:
            x_groups = data[x_prop].unique()
            data['x_bin'] = data[x_prop]

        data_ = []
        annotations = []
        annotation_template = {'font': {'color': annotation_color,
                                        'size': 0.7 * self.font_size,
                                        'family': self.font_family},
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
                a_d = annotation_template.copy()
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
        trace = go.Heatmap(z=data_, x=x_labels, y=y_labels,
                           colorscale=colorscale or self.colorscale, colorbar={
                'title': colorbar_title, 'titleside': 'right',
                'tickfont': {'size': 0.75 * self.tick_size,
                             'family': self.font_family},
                'titlefont': {'size': self.font_size,
                              'family': self.font_family}
            })
        layout = self.layout.copy()

        # heatmap specific formatting:
        for ax in ['x', 'y']:
            if 'type' in layout['{}axis'.format(ax)]:
                layout['{}axis'.format(ax)].pop('type')
        layout['margin']['l'] += self.tick_size * (2 + precision / 10.0) + 35
        if not layout['xaxis'].get('title'):
            warnings.warn('xaxis title was automatically set to x_prop value')
            layout['xaxis']['title'] = x_prop
        if not layout['yaxis'].get('xaxis'):
            warnings.warn('yaxis title was automatically set to y_prop value')
            layout['yaxis']['title'] = y_prop
        layout['annotations'] = annotations
        fig = {'data': [trace], 'layout': layout}
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

        height = 1000 if not hasattr(self, 'height') else self.height
        width = 1300 if not hasattr(self, 'width') else self.width

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

        labels = self.data_from_col(labels, data)
        if self.colorbar_title == 'auto':
            colors_ = self.data_from_col(colors, data)
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
        scatter_layout = {k: v for (k, v) in self.layout.items() if k not in badf}
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
                    'family'] = self.font_family
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['titlefont'][
                    'size'] = self.font_size * text_scale
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'family'] = self.font_family
                fig['layout']['{}axis{}'.format(ax, iplot + 1)]['tickfont'][
                    'size'] = self.font_size * tick_scale
            if iplot % (nplots + 1) != 0:
                fig['data'][iplot].update(marker=marker, text=labels)
            else:
                fig['data'][iplot].update(marker={'color': default_color,
                                    'line': {'width': 0.5, 'color': 'black'}})
        if (default_color=='gray' or default_color=='black') and colors is None:
            fig['layout']['hoverlabel']['font']['color'] = 'white'
        return self.create_plot(fig, return_plot)


    def histogram(self, data=None, cols=None, orientation="vertical",
                  histnorm="count", n_bins=None, bins=None, colors=None,
                  bargap=0, return_plot=False):
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

        if cols is None:
            if isinstance(data, pd.Series):
                cols = [data.name]
                data = pd.DataFrame({cols[0]: data.tolist()})
            elif isinstance(data, pd.DataFrame):
                cols = data.columns.values
            else:
                data = pd.DataFrame({'trace1': data})
                cols = ['trace1']



        # Transform all entries to listlike, if given as single entries
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
        return self.create_plot(fig, return_plot)

    def violin(self, data=None, cols=None, group_col=None, groups=None,
               title=None, colors=None, use_colorscale=False,
               return_plot=False):
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
                color types (str). If None, uses grey as all colors.
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

        if not colors:
            use_colorscale = False
            colors = ['rgb(105,105,105)'] * data.shape[0]

        fig = FF.create_violin(data=data, data_header=cols[0],
                               group_header=group_col, title=title,
                               colors=colors, use_colorscale=use_colorscale,
                               group_stats=group_stats)


        violin_layout = {k: v for (k, v) in self.layout.items() if k != 'xaxis'}
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
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=cols)

        if cols is None:
            cols = data.columns.values

        if colors is None:
            colors = 'blue'
        else:
            colors = self.data_from_col(colors, data)
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
            dimensions.append({'label': col, 'values': values})

        font_style = self.font_style
        font_style['size'] = 0.65 * font_style['size']
        line = line or {'color': colors,
                        'colorscale': self.colorscale,
                        'colorbar': {'title': colorbar_title,
                                     'titleside': 'right',
                                     'tickfont': font_style,
                                     'titlefont': font_style,
                                     }}
        par_coords = go.Parcoords(line=line, dimensions=dimensions)

        par_coords.tickfont = font_style
        par_coords.labelfont = font_style
        par_coords.rangefont = font_style

        fig = {'data': [par_coords], 'layout': self.layout}
        return self.create_plot(fig, return_plot)

    def heatmap_plot(self, data, x_labels=None, y_labels=None,
                     colorscale='Viridis', colorscale_range=None,
                     annotations_text=None, annotations_font_size=20,
                     annotations_color='white', return_plot=False):
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
            annotations_font_size: (int) size of annotation text
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
                                      size=annotations_font_size,
                                      family=self.font_family),
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
            colorbar=dict(tickfont=dict(size=int(0.75 * self.tick_size),
                                        family=self.font_family))
        )

        data = [trace0]

        # Add annotations
        self.layout['annotations'] = annotations

        fig = dict(data=data, layout=self.layout)
        return self.create_plot(fig, return_plot)


    def xy_plot(self, x_col, y_col, text=None, color='rgba(70, 130, 180, 1)',
                size=6, colorscale='Viridis', legend=None,
                showlegend=False, mode='markers', marker='circle',
                marker_fill='fill', hoverinfo='x+y+text',
                add_xy_plot=None, marker_outline_width=0,
                marker_outline_color='black', linedash='solid',
                linewidth=2, lineshape='linear', error_type=None,
                error_direction=None, error_array=None,
                error_value=None, error_symmetric=True, error_arrayminus=None,
                error_valueminus=None, return_plot=False):
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
        warnings.warn("xy_plot is deprecated. Please use .xy() instead.",
                      DeprecationWarning)

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
                colorbar=dict(tickfont=dict(size=int(0.75 * self.tick_size),
                                            family=self.font_family))
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
                                tickfont=dict(size=int(0.75 * self.tick_size),
                                              family=self.font_family))
                        )
                    )
                )

        # Add legend
        self.layout['showlegend'] = showlegend

        fig = dict(data=data, layout=self.layout)
        return self.create_plot(fig, return_plot)
