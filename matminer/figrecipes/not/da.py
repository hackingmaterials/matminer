# Tests will go here for PlotlyFig

from matminer import PlotlyFig
from pymatgen.util.testing import PymatgenTest
import pandas as pd
import numpy as np
import json



a = [1.6, 2.1, 3]
b = [1, 4.2, 9]
c = [14, 15, 17]
xlabels = ['low', 'med', 'high']
ylabels = ['worst', 'mediocre', 'best']



def refresh(open_plots=False):
    """
    For developer use. Refresh the json files and open plots to see if they
    look good. Use this function to write the "truth" values for future tests
    using the outputs from current PlotlyFig output.
    """

    pf = PlotlyFig()
    xys = pf.xy([(a, b)], return_plot=True)
    xym = pf.xy([(a, b), (b, a)], return_plot=True)
    hmb = pf.heatmap_basic([a, b, c], xlabels, ylabels, return_plot=True)
    his = pf.histogram(a + b + c, n_bins=5, return_plot=True)
    bar = pf.bar(x=a, y=b, labels=xlabels, return_plot=True)
    pcp = pf.parallel_coordinates([a, b], cols=xlabels, return_plot=True)

    # Layout is compared for the plots which always convert to dataframes,
    # as dataframes are not easily encoded by json.dump
    vio = pf.violin([a, b, c, b, a, c, b], cols=xlabels, return_plot=True)
    scm = pf.scatter_matrix([a, b, c], return_plot=True)

    fnamedict = {"xys": xys, "xym": xym, "hmb": hmb, "his": his, "bar": bar,
                 "pcp": pcp, "vio": vio, "scm": scm}

    for fname, obj in fnamedict.items():
        if obj in [vio, scm]:
            obj = obj['layout']
        json.dump(obj, open("template_{}.json".format(fname), "w"))

    if open_plots:
        for obj in fnamedict.values():
            pf.create_plot(obj, return_plot=False)



# class StructureFeaturesTest(PymatgenTest):
#     def setUp(self):
#         self.pf = PlotlyFig(show_offline_plot=False)
#
#     def test_xy(self):
#         # Single trace
#         xys_test = self.pf.xy([(a, b)], return_plot=True)
#         xys_true = json.load(open("template_xys.json", "r"))
#         self.assertEqual(xys_test, xys_true)
#
#         # Multi trace
#         xym_test = self.pf.xy([(a, b), (b, a)], return_plot=True)
#         xym_true = json.load(open("template_xym.json", "r"))
#         self.assertEqual(xym_test, xym_true)
#
#     def test_heatmap_basic(self):
#         hmb_test = pf.heatmap_basic([a, b, c], xlabels, ylabels, return_plot=True)
#         hmb_true = json.load(open("template_hmb.json", "r"))
#         self.assertEqual(hmb_test, hmb_true)
#
#     def test_histogram(self):
#         his_test = self.pf.histogram(a + b + c, n_bins=5, return_plot=True)
#         his_true = json.load(open("template_his.json", "r"))
#         self.assertEqual(his_test, his_true)
#
#     def test_bar(self):
#         bar_test = self.pf.bar(x=a, y=b, labels=xlabels, return_plot=True)
#         bar_true = json.load(open("template_bar.json", "r"))
#         self.assertEqual(bar_test, bar_true)
#
#     def test_parallel_coordinates(self):
#         pcp_test = self.pf.parallel_coordinates([a, b], cols=xlabels, return_plot=True)
#         pcp_true = json.load(open("template_pcp.json", "r"))
#         self.assertEqual(pcp_test, pcp_true)
#
#     def test_violin(self):
#         vio_test = self.pf.violin([a, b, c, b, a, c, b], cols=xlabels,return_plot=True)['layout']
#         vio_true = json.load(open("template_vio.json", "r"))
#         self.assertEqual(vio_test, vio_true)
#
#     def scatter_matrix(self):
#         scm_test = self.pf.scatter_matrix([a, b, c], return_plot=True)['layout']
#         scm_true = json.load(open("template_scm.json"))
#         self.assertEqual(scm_test, scm_true)
#
#     def test_heatmap_df(self):
#         pass


if __name__ == "__main__":

    # refresh(open_plots=True)
    df = pd.DataFrame(data=np.asarray([a, b, c]), columns=xlabels)
    pf = PlotlyFig()
    pf.heatmap_df(data=df)









