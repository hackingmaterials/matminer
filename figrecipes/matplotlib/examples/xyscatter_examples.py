from figrecipes.matplotlib.plots import XYPlot

__author__ = 'Anubhav Jain <ajain@lbl.gov>'

"""
Note - be sure to copy/modify matplotlibrc
"""

def xyscatter_ex1():
    data = [{'x': [1, 2, 3], 'y': [1, 4, 9]}, {'x': [1, 2], 'y': [7, 8]}, {'x': [3, 4], 'y': [6, 3]}, {'x': [5, 6], 'y': [6, 3]}]
    sp = XYPlot(data)
    sp.plot()

if __name__ == "__main__":
    xyscatter_ex1()