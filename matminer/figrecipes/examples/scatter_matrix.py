import numpy as np
import pandas as pd
from matminer.datasets.dataframe_loader import load_elastic_tensor
from matminer.figrecipes.plot import PlotlyFig

"""
This script shows some basic examples of scatter_matrix using figrecipes in matminer.
"""

def plot_scatter_matrix():
    """
    A few different scatter matrix plots using elastic dataset in matminer.
    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    df = load_elastic_tensor()
    pf = PlotlyFig(df)

    # basic matrix:
    pf.scatter_matrix(cols=['K_VRH', 'G_VRH', 'nsites', 'volume'])

    # with colorscale and labels:
    pf.scatter_matrix(cols=['K_VRH', 'G_VRH', 'nsites', 'volume'],
                      colors='nsites',
                      labels='material_id',
                      colorscale='Picnic')

    # with all the numerical columns included (note the change in sizes):
    pf = PlotlyFig(filename='scatter_matrix_elastic', fontscale=0.6)
    pf.scatter_matrix(df, marker_scale=0.6)

if __name__ == '__main__':
    plot_scatter_matrix()