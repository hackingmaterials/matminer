import unittest

import pandas as pd

from matminer.featurizers.site.external import SOAP
from matminer.featurizers.site.tests.base import SiteFeaturizerTest


class ExternalSiteTests(SiteFeaturizerTest):
    def test_SOAP(self):
        def n_soap_feat(soaper):
            return soaper.soap.get_number_of_features()

        # Test individual samples
        soap = SOAP(rcut=3.0, nmax=4, lmax=2, sigma=1, periodic=True)
        soap.fit([self.diamond])
        v = soap.featurize(self.diamond, 0)
        self.assertEqual(len(v), n_soap_feat(soap))

        soap.fit([self.ni3al])
        v = soap.featurize(self.ni3al, 0)
        self.assertEqual(len(v), n_soap_feat(soap))

        # Test dataframe fitting
        df = pd.DataFrame({"s": [self.diamond, self.ni3al, self.nacl], "idx": [0, 1, 0]})
        soap.fit(df["s"])
        df = soap.featurize_dataframe(df, ["s", "idx"])
        self.assertTupleEqual(df.shape, (3, n_soap_feat(soap) + 2))

        # Check that only the first has carbon features
        carbon_label = df["SOAP_29"]
        self.assertTrue(carbon_label[0] != 0)
        self.assertTrue(carbon_label[1] == 0)
        self.assertTrue(carbon_label[2] == 0)


if __name__ == "__main__":
    unittest.main()
