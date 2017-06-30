import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
import numpy as np
import warnings
from scipy import stats


__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class PlotlyFig:
    def __init__(self, plot_title=None, x_title=None, y_title=None, hovermode='closest', filename=None,
                 plot_mode='offline', username=None, api_key=None, textsize=30, ticksize=25, fontfamily=None,
                 height=800, width=1000, scale=None, margin_top=100, margin_bottom=80, margin_left=80, margin_right=80,
                 pad=0):
        """
        Class for making Plotly plots

        Args:
            plot_title: (str) title of plot
            x_title: (str) title of x-axis
            y_title: (str) title of y-axis
            hovermode: (str) determines the mode of hover interactions. Can be 'x'/'y'/'closest'/False
            filename: (str) name/filepath of plot file
            plot_mode: (str) (i) 'offline': creates and saves plots on the local disk, (ii) 'notebook': to embed plots
                in a IPython/Jupyter notebook, (iii) 'online': save the plot in your online plotly account (requires
                the fields 'username' and 'api_key' to be set), or (iv) 'static': save a static image of the plot
                locally. Valid image formats are 'png', 'svg', 'jpeg', and 'pdf'. The format is taken as the extension
                of the filename or as the supplied format.
            username: (str) plotly account username
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
            margin_top: (float) Sets the top margin (in px)
            margin_bottom: (float) Sets the bottom margin (in px)
            margin_left: (float) Sets the left margin (in px)
            margin_right: (float) Sets the right margin (in px)
            pad: (float) Sets the amount of padding (in px) between the plotting area and the axis lines

        Returns: None

        """
        self.title = plot_title
        self.x_title = x_title
        self.y_title = y_title
        self.hovermode = hovermode
        self.filename = filename
        self.plot_mode = plot_mode
        self.username = username
        self.api_key = api_key
        self.textsize = textsize
        self.ticksize = ticksize
        self.fontfamily = fontfamily
        self.height = height
        self.width = width
        self.scale = scale

        # Make default layout
        self.layout = dict(
            title=self.title,
            titlefont=dict(size=self.textsize, family=self.fontfamily),
            xaxis=dict(title=self.x_title, titlefont=dict(size=self.textsize, family=self.fontfamily),
                       tickfont=dict(size=self.ticksize, family=self.fontfamily)),
            yaxis=dict(title=self.y_title, titlefont=dict(size=self.textsize, family=self.fontfamily),
                       tickfont=dict(size=self.ticksize, family=self.fontfamily)),
            hovermode=self.hovermode,
            width=self.width,
            height=self.height,
            margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right, pad=pad)
        )

        if self.plot_mode == 'online':
            if not self.username:
                raise ValueError('field "username" must be filled in online plotting mode')
            if not self.api_key:
                raise ValueError('field "api_key" must be filled in online plotting mode')

        elif self.plot_mode == 'static':
            if not self.filename or not self.filename.lower().endswith(('.png', '.svg', '.jpeg', '.pdf')):
                raise ValueError(
                    'field "filename" must be filled in static plotting mode and must have an extension ending in ('
                    '".png", ".svg", ".jpeg", ".pdf")')

    def xy_plot(self, x_col, y_col, text=None, color='rgba(70, 130, 180, 1)', size=6, colorscale='Viridis',
                legend=None, showlegend=False, mode='markers', marker='circle', marker_fill='fill',
                hoverinfo='x+y+text', add_xy_plot=None, marker_outline_width=0, marker_outline_color='black',
                linedash='solid', linewidth=2, lineshape='linear', error_type=None, error_direction=None,
                error_array=None, error_value=None, error_symmetric=True, error_arrayminus=None, error_valueminus=None):
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

        Returns: XY scatter plot

        """
        showscale = False

        if type(color) is not str:
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
                line=dict(width=marker_outline_width, color=marker_outline_color, colorscale=colorscale),
                symbol=marker,
                colorbar=dict(tickfont=dict(size=int(0.75 * self.ticksize), family=self.fontfamily))
            ),
            line=dict(dash=linedash, width=linewidth, shape=lineshape)
        )

        # Add error bars
        if error_type:
            if error_direction is None:
                raise ValueError("The field 'error_direction' must be populated if 'err_type' is specified")
            if error_type == 'data':
                if error_symmetric:
                    trace0['error_' + error_direction] = dict(type=error_type, array=error_array)
                else:
                    if not error_arrayminus:
                        raise ValueError("Please specify error bar lengths in the negative direction")
                    trace0['error_' + error_direction] = dict(type=error_type, array=error_array,
                                                              arrayminus=error_arrayminus)
            elif error_type == 'constant' or error_type == 'percent':
                if error_symmetric:
                    trace0['error_' + error_direction] = dict(type=error_type, value=error_value)
                else:
                    if not error_valueminus:
                        raise ValueError("Please specify error bar lengths in the negative direction")
                    trace0['error_' + error_direction] = dict(type=error_type, value=error_value,
                                                              valueminus=error_valueminus)
            else:
                raise ValueError("Invalid error bar type. Please choose from 'data'/'constant'/'percent'.")

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
                            colorscale=colorscale, # colorscale is fixed to that of the main plot
                            showscale=showscale, # showscale is fixed to that of the main plot
                            line=dict(width=marker_outline_width, color=marker_outline_color, colorscale=colorscale),
                            symbol=plot_data['marker'],
                            colorbar=dict(tickfont=dict(size=int(0.75 * self.ticksize), family=self.fontfamily))
                        )
                    )
                )

        # Add legend
        self.layout['showlegend'] = showlegend

        fig = dict(data=data, layout=self.layout)

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)

    def heatmap_plot(self, data, x_labels=None, y_labels=None, colorscale='Viridis', colorscale_range=None,
                     annotations_text=None, annotations_text_size=20, annotations_color='white'):
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

        Returns: heatmap plot

        """
        if not colorscale_range:
            colorscale_min = None
            colorscale_max = None
        elif len(colorscale_range) == 2:
            colorscale_min = colorscale_range[0]
            colorscale_max = colorscale_range[1]
        else:
            raise ValueError("The field 'colorscale_range' must be a list with two values.")

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
                            font=dict(color=annotations_color, size=annotations_text_size, family=self.fontfamily),
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
            colorbar=dict(tickfont=dict(size=int(0.75 * self.ticksize), family=self.fontfamily))
        )

        data = [trace0]

        # Add annotations
        self.layout['annotations'] = annotations

        fig = dict(data=data, layout=self.layout)

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)

    def violin_plot(self, data, data_col=None, group_col=None, title=None, height=800, width=1000,
                    colors=None, use_colorscale=False, groups=None):
        """
        Create a violin plot using Plotly.

        Args:
            data: (list/array) accepts either a list of numerical values, a list of dictionaries all with identical keys
                and at least one column of numeric values, or a pandas dataframe with at least one column of numbers
            data_col: (str) the header of the data column to be used from an inputted pandas dataframe. Not applicable
                if 'data' is a list of numeric values
            group_col: (str) applicable if grouping data by a variable. 'group_header' must be set to the name of the
                grouping variable
            title: (str) the title of the violin plot
            height: (float) the height of the violin plot
            width: (float) the width of the violin plot
            colors: (str/tuple/list/dict) either a plotly scale name (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu,
                Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis), an rgb or hex
                color, a color tuple, a list of colors or a dictionary. An rgb color is of the form 'rgb(x, y, z)'
                where x, y and z belong to the interval [0, 255] and a color tuple is a tuple of the form (a, b, c)
                where a, b and c belong to [0, 1]. If colors is a list, it must contain valid color types as its
                members. If colors is a dictionary, its keys must represent group names, and corresponding values must
                be valid color types (str).
            use_colorscale: (bool) Only applicable if grouping by another variable. Will implement a colorscale based
                on the first 2 colors of param colors. This means colors must be a list with at least 2 colors in it
                (Plotly colorscales are accepted since they map to a list of two rgb colors)
            groups: (list) list of group names (strings). This field is used when the same colorscale is required to be
                used for all violins.

        Returns: a Plotly violin plot

        """
        if groups and isinstance(data, pd.DataFrame):
            use_colorscale = True
            group_stats = {}
            groupby_data = data.groupby([group_col])

            for group in groups:
                data_from_group = groupby_data.get_group(group)[data_col]
                stat = np.median(data_from_group)
                group_stats[group] = stat

        else:
            group_stats = None

        # Filter out groups from dataframe that have only 1 row.
        if isinstance(data, pd.DataFrame):
            group_value_counts = data[group_col].value_counts().to_dict()

            for j in group_value_counts:
                if group_value_counts[j] == 1:
                    data = data[data[group_col] != j]
                    warnings.warn('Omitting rows with group = ' + str(j) + ' which have only one row in the dataframe.')

        fig = FF.create_violin(data=data, data_header=data_col, group_header=group_col, title=title, height=height,
                               width=width, colors=colors, use_colorscale=use_colorscale, group_stats=group_stats)

        # Cannot add x-axis title as the above object populates it with group names.
        fig.update(dict(
            layout=dict(
            title=self.title,
            titlefont=dict(size=self.textsize, family=self.fontfamily),
            yaxis=dict(title=self.y_title, titlefont=dict(size=self.textsize, family=self.fontfamily),
                       tickfont=dict(size=self.ticksize, family=self.fontfamily)),
            )
        ))

        # Change sizes in all x-axis
        for item in fig['layout']:
            if item.startswith('xaxis'):
                fig['layout'][item].update(
                    dict(
                        titlefont=dict(size=self.textsize, family=self.fontfamily),
                        tickfont=dict(size=self.ticksize, family=self.fontfamily)
                    )
                )

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)

    def scatter_matrix(self, dataframe, select_columns=None, index_colname=None, diag_kind='scatter',
                       marker_size=10, height=800, width=1000, marker_outline_width=0, marker_outline_color='black'):
        """
        Create a scatter matrix plot from dataframes using Plotly.

        Args:
            dataframe: (array) array of the data with column headers
            select_columns: (list) names/headers of columns to plot from the dataframe
            index_colname: (str) name of the index column in data array
            diag_kind: (str) sets the chart type for the main diagonal plots (default='scatter')
                Choose from 'scatter'/'box'/'histogram'
            marker_size: (float) sets the marker size (in px)
            height: (int/float) sets the height of the chart
            width: (int/float) sets the width of the chart
            marker_outline_width: (int) thickness of marker outline (currently affects the outline of histograms too
                when "diag_kind" = 'histogram')
            marker_outline_color: (str/array) color of marker outline - accepts similar formats as other color variables

        Returns: a Plotly scatter matrix plot

        """
        df = dataframe[select_columns] if select_columns else dataframe
        fig = FF.create_scatterplotmatrix(df, index=index_colname, diag=diag_kind, size=marker_size,
                                              height=height, width=width)

        # Add outline to markers
        for trace in fig['data']:
            trace['marker']['line'] = dict(width=marker_outline_width, color=marker_outline_color)

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)

    def histogram(self, x, histnorm="", x_start=None, x_end=None, bin_size=1, color='rgba(70, 130, 180, 1)', bargap=0):
        """
        Create a histogram using Plotly

        Args:
            x: (list) sample data
            histnorm: (str) Specifies the type of normalization used for this histogram trace. If "", the span of each
                bar corresponds to the number of occurrences (i.e. the number of data points lying inside the bins). If
                "percent", the span of each bar corresponds to the percentage of occurrences with respect to the total
                number of sample points (here, the sum of all bin area equals 100%). If "density", the span of each bar
                corresponds to the number of occurrences in a bin divided by the size of the bin interval (here, the
                sum of all bin area equals the total number of sample points). If "probability density", the span of
                each bar corresponds to the probability that an event will fall into the corresponding bin (here, the
                sum of all bin area equals 1)
            x_start: (float) starting value for x-axis bins. Note: after some testing, this variable does not seem to
                be read by Plotly when set to 0 for the latest version of Plotly as of this commit (Nov'16).
            x_end: (float) end value for x-axis bins
            bin_size: (float) step in-between value of each x axis bin
            color: (str/array) in the format of a (i) color name (eg: "red"), or (ii) a RGB tuple,
                (eg: "rgba(255, 0, 0, 0.8)"), where the last number represents the marker opacity/transparency, which
                must be between 0.0 and 1.0., or (iii) hexagonal code (eg: "FFBAD2")
            bargap: (float) gap between bars

        Returns: a Plotly histogram plot

        """
        if not x_start:
            x_start = min(x)

        if not x_end:
            x_end = max(x)

        trace0 = go.Histogram(x=x, histnorm=histnorm, xbins=dict(start=x_start, end=x_end, size=bin_size),
                              marker=dict(color=color))

        data = [trace0]

        self.layout['hovermode'] = 'x'
        self.layout['bargap'] = bargap

        fig = dict(data=data, layout=self.layout)

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)

    def bar_chart(self, x, y):
        """
        Create a bar chart using Plotly

        Args:
            x: (list/numpy array/Pandas series of numbers, strings, or datetimes) sets the x coordinates
            y: (list/numpy array/Pandas series of numbers, strings, or datetimes) sets the y coordinates

        Returns: a Plotly bar chart

        """

        trace0 = go.Bar(x=x, y=y)

        data = [trace0]

        fig = dict(data=data, layout=self.layout)

        if self.plot_mode == 'offline':
            if self.filename:
                plotly.offline.plot(fig, filename=self.filename)
            else:
                plotly.offline.plot(fig)

        elif self.plot_mode == 'notebook':
            plotly.offline.init_notebook_mode()  # run at the start of every notebook; version 1.9.4 required
            plotly.offline.iplot(fig)

        elif self.plot_mode == 'online':
            plotly.tools.set_credentials_file(username=self.username, api_key=self.api_key)
            if self.filename:
                plotly.plotly.plot(fig, filename=self.filename, sharing='public')
            else:
                plotly.plotly.plot(fig, sharing='public')

        elif self.plot_mode == 'static':
            plotly.plotly.image.save_as(fig, filename=self.filename, height=self.height, width=self.width,
                                        scale=self.scale)
