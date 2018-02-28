"""
PlotlyFig examples of histogram plots.
"""

from matminer.datasets.dataframe_loader import load_elastic_tensor, \
    load_dielectric_constant
from matminer.figrecipes.plot import PlotlyFig


def basic_histogram():
    """
    Here we plot a basic histogram showing the distribution of band gaps
    in the matminer dielectric constant dataset, originally taken from Petousis
    et al., 2017.
    """
    df = load_dielectric_constant()
    pf = PlotlyFig(title="Distribution of Band Gaps in the Dielectric Constant "
                         "Dataset",
                   x_title="Bang Gap (eV)",
                   hoverinfo='y')
    pf.histogram(df['band_gap'])


def advanced_histogram():
    """
    This is a work in progress
    """

    df = load_elastic_tensor()
    pf = PlotlyFig(df, title="Various Histograms")
    pf.histogram(cols=['G_Reuss', 'G_VRH', 'G_Voigt'], bins={'size': 10})


if __name__ == "__main__":
    basic_histogram()
    advanced_histogram()
