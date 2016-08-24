import plotly

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class ScatterPlot:

    def __init__(self, plot_title=None, x_title=None, y_title=None, mode='markers', hovermode='closest', filename=None):
        self.title = plot_title
        self.x_title = x_title
        self.y_title = y_title
        self.mode = mode
        self.hovermode = hovermode
        self.filename = filename

    def plot_dataframe(self, dataframe, x_col, y_col, text_col=None):
        fig = {
            'data': [
                {
                    'x': dataframe[x_col],
                    'y': dataframe[y_col],
                    'text': dataframe[text_col],
                    'mode': self.mode},
            ],
            'layout': {
                'title': self.title,
                'xaxis': {'title': self.x_title},
                'yaxis': {'title': self.y_title},
                'hovermode': self.hovermode
            }
        }

        if self.filename:
            plotly.offline.plot(fig, filename=self.filename)
        else:
            plotly.offline.plot(fig)
