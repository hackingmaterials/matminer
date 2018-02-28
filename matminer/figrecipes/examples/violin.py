"""
PlotlyFig examples of bar plots.
"""

from matminer import PlotlyFig
from matminer.datasets.dataframe_loader import load_elastic_tensor


df = load_elastic_tensor()

def simple_violin():
    pf = PlotlyFig(df, title="Distribution of Elastic Constant Averages",
                   colorscale='Reds')
    pf.violin(cols=['K_Reuss', 'K_Voigt', 'G_Reuss', 'G_Voigt'],
              use_colorscale=True)


if __name__ == "__main__":
    simple_violin()
