import unittest

import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN, JmolNN, CrystalNN

from matminer.featurizers.site.chemical import (
    EwaldSiteEnergy,
    ChemicalSRO,
    LocalPropertyDifference,
    SiteElementalProperty,
)
from matminer.featurizers.site.tests.base import SiteFeaturizerTest


class ChemicalSiteTests(SiteFeaturizerTest):
    def test_chemicalSRO(self):
        df_sc = pd.DataFrame({"struct": [self.sc], "site": [0]})
        df_cscl = pd.DataFrame({"struct": [self.cscl], "site": [0]})
        vnn = ChemicalSRO.from_preset("VoronoiNN", cutoff=6.0)
        vnn.fit(df_sc[["struct", "site"]])
        vnn_csros = vnn.featurize_dataframe(df_sc, ["struct", "site"])
        self.assertAlmostEqual(vnn_csros["CSRO_Al_VoronoiNN"][0], 0.0)
        vnn = ChemicalSRO(VoronoiNN(), includes="Cs")
        vnn.fit(df_cscl[["struct", "site"]])
        vnn_csros = vnn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(vnn_csros["CSRO_Cs_VoronoiNN"][0], 0.0714285714)
        vnn = ChemicalSRO(VoronoiNN(), excludes="Cs")
        vnn.fit(df_cscl[["struct", "site"]])
        vnn_csros = vnn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(vnn_csros["CSRO_Cl_VoronoiNN"][0], -0.0714285714)
        jmnn = ChemicalSRO.from_preset("JmolNN", el_radius_updates={"Al": 1.55})
        jmnn.fit(df_sc[["struct", "site"]])
        jmnn_csros = jmnn.featurize_dataframe(df_sc, ["struct", "site"])
        self.assertAlmostEqual(jmnn_csros["CSRO_Al_JmolNN"][0], 0.0)
        jmnn = ChemicalSRO.from_preset("JmolNN")
        jmnn.fit(df_cscl[["struct", "site"]])
        jmnn_csros = jmnn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(jmnn_csros["CSRO_Cs_JmolNN"][0], -0.5)
        self.assertAlmostEqual(jmnn_csros["CSRO_Cl_JmolNN"][0], -0.5)
        mdnn = ChemicalSRO.from_preset("MinimumDistanceNN")
        mdnn.fit(df_cscl[["struct", "site"]])
        mdnn_csros = mdnn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(mdnn_csros["CSRO_Cs_MinimumDistanceNN"][0], 0.5)
        self.assertAlmostEqual(mdnn_csros["CSRO_Cl_MinimumDistanceNN"][0], -0.5)
        monn = ChemicalSRO.from_preset("MinimumOKeeffeNN")
        monn.fit(df_cscl[["struct", "site"]])
        monn_csros = monn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(monn_csros["CSRO_Cs_MinimumOKeeffeNN"][0], 0.5)
        self.assertAlmostEqual(monn_csros["CSRO_Cl_MinimumOKeeffeNN"][0], -0.5)
        mvnn = ChemicalSRO.from_preset("MinimumVIRENN")
        mvnn.fit(df_cscl[["struct", "site"]])
        mvnn_csros = mvnn.featurize_dataframe(df_cscl, ["struct", "site"])
        self.assertAlmostEqual(mvnn_csros["CSRO_Cs_MinimumVIRENN"][0], 0.5)
        self.assertAlmostEqual(mvnn_csros["CSRO_Cl_MinimumVIRENN"][0], -0.5)
        # test fit + transform
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit(df_cscl[["struct", "site"]])  # dataframe
        vnn_csros = vnn.transform(df_cscl[["struct", "site"]].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit(df_cscl[["struct", "site"]].values)  # np.array
        vnn_csros = vnn.transform(df_cscl[["struct", "site"]].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit([[self.cscl, 0]])  # list
        vnn_csros = vnn.transform([[self.cscl, 0]])
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        # test fit_transform
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn_csros = vnn.fit_transform(df_cscl[["struct", "site"]].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)

    def test_ewald_site(self):
        ewald = EwaldSiteEnergy(accuracy=4)

        # Set the charges
        for s in [self.sc, self.cscl]:
            s.add_oxidation_state_by_guess()

        # Run the sc-Al structure
        self.assertArrayAlmostEqual(ewald.featurize(self.sc, 0), [0])

        # Run the cscl-structure
        #   Compared to a result computed using GULP
        self.assertAlmostEqual(ewald.featurize(self.cscl, 0), ewald.featurize(self.cscl, 1))
        self.assertAlmostEqual(ewald.featurize(self.cscl, 0)[0], -6.98112443 / 2, 3)

        # Re-run the Al structure to make sure it is accurate
        #  This is to test the caching feature
        self.assertArrayAlmostEqual(ewald.featurize(self.sc, 0), [0])

    def test_local_prop_diff(self):
        f = LocalPropertyDifference()

        # Test for Al, all features should be zero
        features = f.featurize(self.sc, 0)
        self.assertArrayAlmostEqual(features, [0])

        # Change the property to Number, compute for B1
        f.set_params(properties=["Number"])
        for i in range(2):
            features = f.featurize(self.b1, i)
            self.assertArrayAlmostEqual(features, [1])

    def test_site_elem_prop(self):
        f = SiteElementalProperty.from_preset("seko-prb-2017")

        # Make sure it does the B1 structure correctly
        feat_labels = f.feature_labels()
        feats = f.featurize(self.b1, 0)
        self.assertAlmostEqual(1, feats[feat_labels.index("site Number")])

        feats = f.featurize(self.b1, 1)
        self.assertAlmostEqual(2, feats[feat_labels.index("site Number")])

        # Test the citations
        citations = f.citations()
        self.assertEqual(1, len(citations))
        self.assertIn("Seko2017", citations[0])


if __name__ == "__main__":
    unittest.main()
