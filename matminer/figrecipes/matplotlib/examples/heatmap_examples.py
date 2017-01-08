from figrecipes.matplotlib.plots import HeatMapPlot

__author__ = 'Anubhav Jain <ajain@lbl.gov>'

"""
Note - be sure to copy/modify matplotlibrc
"""

def heatmap_ex1():
    # HEAT MAP
    data = [[1, 2], [3, 4], [5, 6]]
    hmp = HeatMapPlot(data, ['I', 'II'], ['A', 'B', 'C'])
    hmp.plot()


if __name__ == "__main__":
    heatmap_ex1()
