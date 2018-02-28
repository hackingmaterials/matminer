"""
PlotlyFig examples of heatmap plots.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from matminer import PlotlyFig


def plot_simple_heatmap_df():
    """
    Very basic example shows how heatmap_df takes a dataframe and returns
    an overview heatmap of the data with the help of pandas.qcut

    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = [2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
    c = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df = pd.DataFrame(data=np.asarray([a, b, c]).T,
                      columns=['var a', 'var b', 'var c'])
    pf = PlotlyFig(colorscale='Oregon')
    pf.heatmap_df(df, x_labels=['low','high'], y_labels=['q1','q2','q3','q4'])


def plot_basic_heatmap():
    """
    Very basic heatmap plot when the data is already in the right format.
    Duplicate example; see https://plot.ly/python/heatmaps/ for more info

    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    pf = PlotlyFig(filename='heatmap_basic')
    z=[[1, 20, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]]
    x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    y=['Morning', 'Afternoon', 'Evening']
    pf.heatmap_basic(z, x_labels=x, y_labels=y)


def plot_boston_dataset_heatmap_df():
    """
    This example uses sklearn boston dataset to plot a heatmap based on the
    binned data to see if there is a relationship between the Median house
    value w/ air NOX concentration and CHAS (Charles River dummy variable)

    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    boston = datasets.load_boston()
    df_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df_boston['Median value'] = boston.target

    pf = PlotlyFig(fontscale=0.8, filename='boston', colorscale='RdBu')
    pf.heatmap_df(df_boston[['NOX', 'CHAS', 'Median value']],
               x_nqs=4,
               y_labels=['otherwise', 'tract bounds river'])


if __name__ == '__main__':
    plot_simple_heatmap_df()
    plot_basic_heatmap()
    plot_boston_dataset_heatmap_df()