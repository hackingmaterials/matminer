import unittest

import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.packing import AtomicPackingEfficiency
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class PackingFeaturesTest(CompositionFeaturesTest):
    def test_ape(self):
        f = AtomicPackingEfficiency()
        ef = ElementFraction()
        ef.set_n_jobs(1)

        # Test the APE calculation routines
        self.assertAlmostEqual(1.11632, f.get_ideal_radius_ratio(15))
        self.assertAlmostEqual(0.154701, f.get_ideal_radius_ratio(2))
        self.assertAlmostEqual(1.65915, f.get_ideal_radius_ratio(27))
        self.assertAlmostEqual(15, f.find_ideal_cluster_size(1.116)[0])
        self.assertAlmostEqual(3, f.find_ideal_cluster_size(0.1)[0])
        self.assertAlmostEqual(24, f.find_ideal_cluster_size(2)[0])

        # Test the nearest neighbor lookup tool
        nn_lookup = f.create_cluster_lookup_tool([Element("Cu"), Element("Zr")])

        #  Check that the table gets the correct structures
        stable_clusters = [
            Composition("CuZr10"),
            Composition("Cu6Zr6"),
            Composition("Cu8Zr5"),
            Composition("Cu13Zr1"),
            Composition("Cu3Zr12"),
            Composition("Cu8Zr8"),
            Composition("Cu12Zr5"),
            Composition("Cu17Zr"),
        ]
        ds, _ = nn_lookup.kneighbors(ef.featurize_many(stable_clusters), n_neighbors=1)
        self.assertArrayAlmostEqual([[0]] * 8, ds)
        self.assertEqual(8, nn_lookup._fit_X.shape[0])

        # Swap the order of the clusters, make sure it gets the same list
        nn_lookup_swapped = f.create_cluster_lookup_tool([Element("Zr"), Element("Cu")])
        self.assertArrayAlmostEqual(sorted(nn_lookup._fit_X.tolist()), sorted(nn_lookup_swapped._fit_X.tolist()))

        # Make sure we had a cache hit
        self.assertEqual(1, f._create_cluster_lookup_tool.cache_info().misses)
        self.assertEqual(1, f._create_cluster_lookup_tool.cache_info().hits)

        # Change the tolerance, see if it changes the results properly
        f.threshold = 0.002
        nn_lookup = f.create_cluster_lookup_tool([Element("Cu"), Element("Zr")])
        self.assertEqual(2, nn_lookup._fit_X.shape[0])
        ds, _ = nn_lookup.kneighbors(
            ef.featurize_many([Composition("CuZr10"), Composition("Cu3Zr12")]),
            n_neighbors=1,
        )
        self.assertArrayAlmostEqual([[0]] * 2, ds)

        # Make sure we had a cache miss
        self.assertEqual(2, f._create_cluster_lookup_tool.cache_info().misses)
        self.assertEqual(1, f._create_cluster_lookup_tool.cache_info().hits)

        # Compute the distances from Cu50Zr50
        mean_dists = f.compute_nearest_cluster_distance(Composition("CuZr"))
        self.assertArrayAlmostEqual([0.424264, 0.667602, 0.800561], mean_dists, decimal=6)

        # Compute the optimal APE for Cu50Zr50
        self.assertArrayAlmostEqual(
            [0.000233857, 0.003508794],
            f.compute_simultaneous_packing_efficiency(Composition("Cu50Zr50")),
        )

        # Test the dataframe calculator
        df = pd.DataFrame({"comp": [Composition("CuZr")]})
        df = f.featurize_dataframe(df, "comp")

        self.assertEqual(6, len(df.columns))
        self.assertIn("dist from 5 clusters |APE| < 0.002", df.columns)

        self.assertAlmostEqual(0.003508794, df["mean abs simul. packing efficiency"][0])

        # Make sure it works with composition that do not match any efficient clusters
        feat = f.compute_nearest_cluster_distance(Composition("Al"))
        self.assertArrayAlmostEqual([1] * 3, feat)


if __name__ == "__main__":
    unittest.main()
