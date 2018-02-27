"""
PlotlyFig examples of parallel coordinates plots.
"""

from matminer.datasets.dataframe_loader import load_elastic_tensor
from matminer import PlotlyFig


def basic_parallel_coordinates():
    df = load_elastic_tensor()
    pf = PlotlyFig(df, title="Elastic tensor dataset", colorscale='Jet')
    pf.parallel_coordinates(colors='volume')


if __name__ == "__main__":
    basic_parallel_coordinates()