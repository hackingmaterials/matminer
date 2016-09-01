import plotly

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class ScatterPlot:
    def __init__(self, plot_title=None, x_title=None, y_title=None, mode='markers', hovermode='closest',
                 filename=None, plot_mode='offline', username=None, api_key=None):
        """

        Args:
            plot_title:
            x_title:
            y_title:
            mode: 'markers'/'lines'/'lines+markers'
            hovermode:
            filename:
            plot_mode:
            username:
            api_key:

        Returns:

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

    def plot_dataframe(self, x_col, y_col, dataframe=None, text=None, color='rgba(70, 130, 180, 1)', size=6):
        """


        Args:
            dataframe:
            x_col:
            y_col:
            text: (str/array)
            color_col: (str) name of a dataframe numeric column to set the marker color scale to
            color_str: (str) in the format of a (i) color name (eg: "red"), or (ii) a RGB tuple,
            (eg: "rgba(255, 0, 0, 0.8)"), where the last number represents the marker opacity/transparency, which must
            be between 0.0 and 1.0., or (iii) hexagonal code (eg: "FFBAD2").
            size_col: (str) name of a dataframe numeric column to set the marker sizes to
            size_int: (int) marker size

        Returns:

        """
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
                        'colorscale': 'Viridis',
                        'showscale': True
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
