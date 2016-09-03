import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class PlotlyPlot:
    def __init__(self, plot_title=None, x_title=None, y_title=None, mode='markers', hovermode='closest',
                 filename=None, plot_mode='offline', username=None, api_key=None):
        """
        Class for making Plotly plots

        Args:
            plot_title: (str) title of plot
            x_title: (str) title of x-axis
            y_title: (str) title of y-axis
            mode: (str) marker style; can be 'markers'/'lines'/'lines+markers'
            hovermode: (str) can be 'x'/'y'/'closest'
            filename: (str) name of plot file
            plot_mode: (str) (i) 'offline': creates and saves plots on the local disk, (ii) 'notebook': to embed plots
                in a IPython/Jupyter notebook, or (iii) 'online': save the plot in your online plotly account (requires
                the fields 'username' and 'api_key' to be set)
            username: (str) plotly account username
            api_key: (str) plotly account API key

        Returns: None

        """
        self.title = plot_title
        self.x_title = x_title
        self.y_title = y_title
        self.mode = mode
        self.hovermode = hovermode
        self.filename = filename
        self.plot_mode = plot_mode
        self.username = username
        self.api_key = api_key

        if self.plot_mode == 'online':
            if not self.username:
                raise ValueError('field "username" must be filled in online plotting mode')
            if not self.api_key:
                raise ValueError('field "api_key" must be filled in online plotting mode')

    def xy_plot(self, x_col, y_col, text=None, color='rgba(70, 130, 180, 1)', size=6, colorscale='Viridis'):
        """
        Make an XY scatter plot, either using arrays of values, or a dataframe.

        Args:
            x_col: (array) x-axis values
            y_col: (array) y-axis values
            text: (str/array) text to use when hovering over points; a single string, or an array of strings, or a
                dataframe column containing text strings
            color: (str/array) in the format of a (i) color name (eg: "red"), or (ii) a RGB tuple,
                (eg: "rgba(255, 0, 0, 0.8)"), where the last number represents the marker opacity/transparency, which
                must be between 0.0 and 1.0., or (iii) hexagonal code (eg: "FFBAD2"), or (iv) name of a dataframe
                numeric column to set the marker color scale to
            size: (int/array) marker size in the format of (i) a constant integer size, or (ii) name of a dataframe
                numeric column to set the marker size scale to
            colorscale: (str) Sets the colorscale. The colorscale must be an array containing arrays mapping a
                normalized value to an rgb, rgba, hex, hsl, hsv, or named color string. At minimum, a mapping for the
                lowest (0) and highest (1) values are required. For example, `[[0, 'rgb(0,0,255)',
                [1, 'rgb(255,0,0)']]`. Alternatively, `colorscale` may be a palette name string of the following list:
                Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
                Blackbody, Earth, Electric, Viridis

        Returns: XY scatter plot

        """
        showscale = False

        if type(color) is not str:
            showscale = True

        if type(size) is not int:
            size_max = max(size)
            size_min = min(size)
            size = (size - size_min) / (size_max - size_min) * 100

        fig = {
            'data': [
                {
                    'x': x_col,
                    'y': y_col,
                    'text': text,
                    'mode': self.mode,
                    'marker': {
                        'size': size,
                        'color': color,
                        'colorscale': colorscale,
                        'showscale': showscale
                    }
                },
            ],
            'layout': {
                'title': self.title,
                'xaxis': {'title': self.x_title},
                'yaxis': {'title': self.y_title},
                'hovermode': self.hovermode
            }
        }

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

    def heatmap_plot(self, data, x_labels=None, y_labels=None, colorscale='Viridis'):
        """
        Make a heatmap plot, either using 2D arrays of values, or a dataframe.

        Args:
            data: (array) an array of arrays. For example, in case of a pandas dataframe 'df', data=df.values.tolist()
            x_labels: (array) an array of strings to label the heatmap columns
            y_labels: (array) an array of strings to label the heatmap rows
            colorscale: (str) Sets the colorscale. The colorscale must be an array containing arrays mapping a
                normalized value to an rgb, rgba, hex, hsl, hsv, or named color string. At minimum, a mapping for the
                lowest (0) and highest (1) values are required. For example, `[[0, 'rgb(0,0,255)',
                [1, 'rgb(255,0,0)']]`. Alternatively, `colorscale` may be a palette name string of the following list:
                Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
                Blackbody, Earth, Electric, Viridis

        Returns: heatmap plot

        """
        fig = {
            'data': [go.Heatmap
                (
                    z=data,
                    colorscale=colorscale,
                    x=x_labels,
                    y=y_labels
                ),
            ],
            'layout': {
                'title': self.title,
                'xaxis': {'title': self.x_title},
                'yaxis': {'title': self.y_title},
                'hovermode': self.hovermode
            }
        }

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

    def violin_plot(self, data, data_col=None, group_col=None, title=None, height=450, width=600,
                    colors=None, use_colorscale=False, group_stats=None):
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
            colors: (str/tuple/list/dict) either a plotly scale name, an rgb or hex color, a color tuple, a list of
                colors or a dictionary. An rgb color is of the form 'rgb(x, y, z)' where x, y and z belong to the
                interval [0, 255] and a color tuple is a tuple of the form (a, b, c) where a, b and c belong to [0, 1].
                If colors is a list, it must contain valid color types as its members. If colors is a dictionary, its
                keys must represent group names, and corresponding values must be valid color types (str).
            use_colorscale: (bool) Only applicable if grouping by another variable. Will implement a colorscale based
                on the first 2 colors of param colors. This means colors must be a list with at least 2 colors in it
                (Plotly colorscales are accepted since they map to a list of two rgb colors)
            group_stats: (dict) a dictioanry where each key is a unique value from the group_header column in data.
                Each value must be a number and will be used to color the violin plots if a colorscale is being used

        Returns: a Plotly violin plot

        """
        fig = FF.create_violin(data=data, data_header=data_col, group_header=group_col, title=title, height=height,
                               width=width, colors=colors, use_colorscale=use_colorscale, group_stats=group_stats)

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


