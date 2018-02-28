"""
This script shows some basic examples of bar plots using figrecipes in
matminer.
"""

from matminer import PlotlyFig
from matminer.datasets.dataframe_loader import load_dielectric_constant


def basic_bar():
    pf = PlotlyFig()
    pf.bar(x = ['var a', 'var b', 'var c'], y = [1, 2, 3])


def advanced_bar():
    """
    Compare the number of sites in the unit cell and eij_max of the first 5
    samples from the piezoelectric_tensor dataset.
    """
    # Format the general layout of our figure with 5 samples
    pf = PlotlyFig(df=load_dielectric_constant().iloc[:5],
                   title='Comparison of 5 materials band gaps and n')
    # Plot!
    colors = ['red', 'orange', 'yellow', 'blue', 'green']
    pf.bar(cols=['n', 'band_gap'], labels='formula', colors=colors)


if __name__ == "__main__":
    basic_bar()
    advanced_bar()