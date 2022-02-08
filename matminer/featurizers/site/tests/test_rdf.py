import unittest

import numpy as np
import pandas as pd

from matminer.featurizers.site.rdf import (
    AngularFourierSeries,
    GaussianSymmFunc,
    GeneralizedRadialDistributionFunction,
)
from matminer.featurizers.site.tests.base import SiteFeaturizerTest
from matminer.featurizers.utils.grdf import Gaussian


class RDFTests(SiteFeaturizerTest):
    def test_gaussiansymmfunc(self):
        data = pd.DataFrame({"struct": [self.cscl], "site": [0]})
        gsf = GaussianSymmFunc()
        gsfs = gsf.featurize_dataframe(data, ["struct", "site"])
        self.assertAlmostEqual(gsfs["G2_0.05"][0], 5.0086817867593822)
        self.assertAlmostEqual(gsfs["G2_4.0"][0], 1.2415138042932981)
        self.assertAlmostEqual(gsfs["G2_20.0"][0], 0.00696)
        self.assertAlmostEqual(gsfs["G2_80.0"][0], 0.0)
        self.assertAlmostEqual(gsfs["G4_0.005_1.0_1.0"][0], 2.6399416897128658)
        self.assertAlmostEqual(gsfs["G4_0.005_1.0_-1.0"][0], 0.90049182882301426)
        self.assertAlmostEqual(gsfs["G4_0.005_4.0_1.0"][0], 1.1810690738596332)
        self.assertAlmostEqual(gsfs["G4_0.005_4.0_-1.0"][0], 0.033850556557100071)

    def test_grdf(self):
        f1 = Gaussian(1, 0)
        f2 = Gaussian(1, 1)
        f3 = Gaussian(1, 5)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        # test fit, transform, and featurize dataframe for both run modes GRDF mode
        grdf = GeneralizedRadialDistributionFunction(bins=[f1, f2, f3], mode="GRDF")
        grdf.fit(s_tuples)
        features = grdf.transform(s_tuples)
        self.assertArrayAlmostEqual(
            features,
            [[4.4807e-06, 0.00031, 0.02670], [3.3303e-06, 0.00026, 0.01753]],
            3,
        )
        features = grdf.featurize_dataframe(pd.DataFrame(s_tuples), [0, 1])
        self.assertArrayEqual(
            list(features.columns.values),
            [
                0,
                1,
                "Gaussian center=0 width=1",
                "Gaussian center=1 width=1",
                "Gaussian center=5 width=1",
            ],
        )

        # pairwise GRDF mode
        grdf = GeneralizedRadialDistributionFunction(bins=[f1, f2, f3], mode="pairwise_GRDF")
        grdf.fit(s_tuples)
        features = grdf.transform(s_tuples)
        self.assertArrayAlmostEqual(features[0], [4.4807e-06, 3.1661e-04, 0.0267], 3)
        self.assertArrayAlmostEqual(
            features[1],
            [2.1807e-08, 6.1119e-06, 0.0142, 3.3085e-06, 2.5898e-04, 0.0032],
            3,
        )
        features = grdf.featurize_dataframe(pd.DataFrame(s_tuples), [0, 1])
        self.assertArrayEqual(
            list(features.columns.values),
            [
                0,
                1,
                "site2 0 Gaussian center=0 width=1",
                "site2 1 Gaussian center=0 width=1",
                "site2 0 Gaussian center=1 width=1",
                "site2 1 Gaussian center=1 width=1",
                "site2 0 Gaussian center=5 width=1",
                "site2 1 Gaussian center=5 width=1",
            ],
        )

        # test preset
        grdf = GeneralizedRadialDistributionFunction.from_preset("gaussian")
        grdf.featurize(self.sc, 0)
        self.assertArrayEqual(
            [bin.name() for bin in grdf.bins],
            [f"Gaussian center={i} width=1.0" for i in np.arange(10.0)],
        )

    def test_afs(self):
        f1 = Gaussian(1, 0)
        f2 = Gaussian(1, 1)
        f3 = Gaussian(1, 5)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        # test transform,and featurize dataframe
        afs = AngularFourierSeries(bins=[f1, f2, f3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(
            features,
            [
                [
                    -1.0374e-10,
                    -4.3563e-08,
                    -2.7914e-06,
                    -4.3563e-08,
                    -1.8292e-05,
                    -0.0011,
                    -2.7914e-06,
                    -0.0011,
                    -12.7863,
                ],
                [
                    -1.7403e-11,
                    -1.0886e-08,
                    -3.5985e-06,
                    -1.0886e-08,
                    -6.0597e-06,
                    -0.0016,
                    -3.5985e-06,
                    -0.0016,
                    -3.9052,
                ],
            ],
            3,
        )
        features = afs.featurize_dataframe(pd.DataFrame(s_tuples), [0, 1])
        self.assertArrayEqual(
            list(features.columns.values),
            [
                0,
                1,
                "AFS (Gaussian center=0 width=1, Gaussian center=0 width=1)",
                "AFS (Gaussian center=0 width=1, Gaussian center=1 width=1)",
                "AFS (Gaussian center=0 width=1, Gaussian center=5 width=1)",
                "AFS (Gaussian center=1 width=1, Gaussian center=0 width=1)",
                "AFS (Gaussian center=1 width=1, Gaussian center=1 width=1)",
                "AFS (Gaussian center=1 width=1, Gaussian center=5 width=1)",
                "AFS (Gaussian center=5 width=1, Gaussian center=0 width=1)",
                "AFS (Gaussian center=5 width=1, Gaussian center=1 width=1)",
                "AFS (Gaussian center=5 width=1, Gaussian center=5 width=1)",
            ],
        )

        # test preset
        afs = AngularFourierSeries.from_preset("gaussian")
        afs.featurize(self.sc, 0)
        self.assertArrayEqual(
            [bin.name() for bin in afs.bins],
            [f"Gaussian center={i} width=0.5" for i in np.arange(0, 10, 0.5)],
        )

        afs = AngularFourierSeries.from_preset("histogram")
        afs.featurize(self.sc, 0)
        self.assertArrayEqual(
            [bin.name() for bin in afs.bins],
            [f"Histogram start={i} width=0.5" for i in np.arange(0, 10, 0.5)],
        )


if __name__ == "__main__":
    unittest.main()
