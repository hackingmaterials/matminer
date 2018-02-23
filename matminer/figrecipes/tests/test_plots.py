# Tests will go here for PlotlyFig

from matminer.figrecipes.plot import PlotlyFig
import pandas as pd


if __name__ == "__main__":

    a = [1, 2, 3]
    b = [1, 4, 9]

    df = pd.DataFrame({'a': a, 'b': b})
    
    pf = PlotlyFig()
    pf.xy(