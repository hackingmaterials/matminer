"""
PlotlyFig testing. Most tests are run by comparing generated Plotly figures
with pre-generated json files. Some are just ensuring Plotly does not throw
errors.
"""
from monty.json import MontyEncoder

__author__ = "Alex Dunn <ardunn@lbl.gov>"

import os
import unittest
import json
import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest
from matminer.figrecipes.plot import PlotlyFig

a = [1.6, 2.1, 3]
b = [1, 4.2, 9]
c = [14, 15, 17]
ah = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bh = [2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
ch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
np.random.seed(23)
xlabels = ['low', 'med', 'high']
ylabels = ['worst', 'mediocre', 'best']
pfkwargs = {'mode': 'offline', 'colorbar_title': 'auto', 'y_scale': 'linear',
            'x_scale': 'linear', 'ticksize': 25, 'fontscale': 0.9,
            'fontsize': 25, 'fontfamily': 'Courier', 'bgcolor': 'white',
            'colorscale': 'Viridis', 'margins': 120, 'pad': 0,
            'filename': 'offline_plot', 'show_offline_plot': True,
            'hovermode': 'closest','hoverinfo': 'x+y+text'}


def refresh_json(open_plots=False):
    """
    For developer use. Refresh the json files and open plots to see if they
    look good. Use this function to set the current PlotlyFig build outputs
    as the true values of the tests.

    Args:
        open_plots (bool): If True, opens all plots generated. Useful if you
            want to check the current build outputs to make sure they look good.
            If False, just generates the json files and quits.
    """

    pf = PlotlyFig(**pfkwargs)
    xys = pf.xy([(a, b)], return_plot=True)
    xym = pf.xy([(a, b), (b, a)], return_plot=True)
    xy_colors = pf.xy([(a, b), (a, c), (c, c)],
                    modes=['markers', 'markers+lines', 'lines'],
                    colors=[c, 'red', 'blue'], return_plot=True)
    hmb = pf.heatmap_basic([a, b, c], xlabels, ylabels, return_plot=True)
    his = pf.histogram(a + b + c, n_bins=5, return_plot=True)
    bar = pf.bar(x=a, y=b, labels=xlabels, return_plot=True)
    pcp = pf.parallel_coordinates([a, b], cols=xlabels, return_plot=True)

    # plotly figures need to be converted jsonable data
    xys['data'] = [p.to_plotly_json() for p in xys['data']]
    xym['data'] = [p.to_plotly_json() for p in xym['data']]
    xy_colors['data'] = [p.to_plotly_json() for p in xy_colors['data']]
    hmb['data'] = [p.to_plotly_json() for p in hmb['data']]
    his['data'] = [p.to_plotly_json() for p in his['data']]
    bar['data'] = [p.to_plotly_json() for p in bar['data']]
    pcp['data'] = [p.to_plotly_json() for p in pcp['data']]

    # Layout is compared for the plots which always convert to dataframes,
    # as dataframes are not easily encoded by json.dump
    vio = pf.violin([a, b, c, b, a, c, b], cols=xlabels, return_plot=True)
    scm = pf.scatter_matrix([a, b, c], return_plot=True)

    # plotly layout needs to be converted jsonable data
    vio = {'layout': vio['layout'].to_plotly_json()}
    scm = {'layout': scm['layout'].to_plotly_json()}

    df = pd.DataFrame(data=np.asarray([ah, bh, ch]).T,
                      columns=['ah', 'bh', 'ch'])
    x_labels = ['low', 'high']
    y_labels = ['small', 'large']
    # TODO: this plot was not JSON serializable, use a different serialization method for all plots
    hmdf = pf.heatmap_df(df, x_labels=x_labels, y_labels=y_labels, return_plot=True)
    hmdf['data'] = [p.to_plotly_json() for p in hmdf['data']]

    df = pd.DataFrame(np.random.rand(50, 3), columns=list('qwe'))
    triangle = pf.triangle(df[['q', 'w', 'e']], return_plot=True)

    fnamedict = {"xys": xys, "xym": xym, "xy_colors":xy_colors,
                 "hmb": hmb, "his": his, "bar": bar,
                 "pcp": pcp, "vio": vio, "scm": scm,
                 'triangle': triangle,
                 'hmdf': hmdf
                 }

    for fname, obj in fnamedict.items():
        if fname in ["vio", "scm"]:
            obj = obj['layout']

        with open("template_{}.json".format(fname), "w") as f:
            json.dump(obj, f, cls=MontyEncoder)

    if open_plots:
        for obj in fnamedict.values():
            pf.create_plot(obj, return_plot=False)


class PlotlyFigTest(PymatgenTest):
    def setUp(self):
        self.pf = PlotlyFig(**pfkwargs)
        self.base_dir = os.path.dirname(os.path.realpath(__file__))

    def fopen(self, fname):
        fname = self.base_dir + "/" + fname
        with open(fname, 'r') as f:
            return json.load(f)

    def test_xy(self):
        # Single trace
        xys_test = self.pf.xy([(a, b)], return_plot=True)
        xys_test['data'] = [p.to_plotly_json() for p in xys_test['data']]
        xys_true = self.fopen("template_xys.json")
        self.assertTrue(xys_test == xys_true)

        # Multi trace
        xym_test = self.pf.xy([(a, b), (b, a)], return_plot=True)
        xym_test['data'] = [p.to_plotly_json() for p in xym_test['data']]
        xym_true = self.fopen("template_xym.json")
        self.assertTrue(xym_test == xym_true)

        xy_colors_test = self.pf.xy([(a, b), (a, c), (c, c)],
                      modes=['markers', 'markers+lines', 'lines'],
                      colors=[c, 'red', 'blue'], return_plot=True)
        xy_colors_test['data'] = [p.to_plotly_json() for p in xy_colors_test['data']]
        xy_colors_true = self.fopen("template_xy_colors.json")
        self.assertTrue(xy_colors_test == xy_colors_true)

    def test_heatmap_basic(self):
        hmb_test = self.pf.heatmap_basic([a, b, c], xlabels, ylabels,
                                         return_plot=True)
        hmb_test['data'] = [p.to_plotly_json() for p in hmb_test['data']]
        hmb_true = self.fopen("template_hmb.json")
        self.assertTrue(hmb_test == hmb_true)

    def test_histogram(self):
        his_test = self.pf.histogram(a + b + c, n_bins=5, return_plot=True)
        his_test['data'] = [p.to_plotly_json() for p in his_test['data']]
        his_true = self.fopen("template_his.json")
        self.assertTrue(his_test == his_true)

    def test_bar(self):
        bar_test = self.pf.bar(x=a, y=b, labels=xlabels, return_plot=True)
        bar_test['data'] = [p.to_plotly_json() for p in bar_test['data']]
        bar_true = self.fopen("template_bar.json")
        self.assertTrue(bar_test == bar_true)

    def test_parallel_coordinates(self):
        pcp_test = self.pf.parallel_coordinates([a, b], cols=xlabels,
                                                return_plot=True)
        pcp_test['data'] = [p.to_plotly_json() for p in pcp_test['data']]
        pcp_true = self.fopen("template_pcp.json")
        self.assertTrue(pcp_test == pcp_true)

    def test_violin(self):
        vio_test = self.pf.violin(
            [a, b, c, b, a, c, b], cols=xlabels, return_plot=True)['layout']
        vio_test = vio_test.to_plotly_json()
        vio_true = self.fopen("template_vio.json")
        self.assertTrue(vio_test == vio_true)

    def test_scatter_matrix(self):
        scm_test = self.pf.scatter_matrix([a, b, c], return_plot=True)['layout']
        scm_test = scm_test.to_plotly_json()
        scm_true = self.fopen("template_scm.json")
        self.assertTrue(scm_test == scm_true)

    def test_heatmap_df(self):

        df = pd.DataFrame(data=np.asarray([ah, bh, ch]).T,
                          columns=['ah', 'bh', 'ch'])
        x_labels = ['low', 'high']
        y_labels = ['small', 'large']
        hmdf_test = self.pf.heatmap_df(df, x_labels=x_labels,
                                       y_labels=y_labels,
                                       return_plot=True)
        hmdf_true = self.fopen("template_hmdf.json")
        self.assertTrue(hmdf_test, hmdf_true)

    def test_trianlge(self):
        df = pd.DataFrame(np.random.rand(50, 3), columns=list('qwe'))
        triangle_test = self.pf.triangle(df[['q', 'w', 'e']], return_plot=True)
        triangle_true = self.fopen("template_triangle.json")
        self.assertTrue(triangle_test, triangle_true)


if __name__ == "__main__":
    #refresh_json(open_plots=True)
    unittest.main()
