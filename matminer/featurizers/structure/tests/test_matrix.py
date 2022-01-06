import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Molecule
from sklearn.exceptions import NotFittedError

from matminer.featurizers.structure.matrix import (
    CoulombMatrix,
    OrbitalFieldMatrix,
    SineCoulombMatrix,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class MatrixStructureFeaturesTest(StructureFeaturesTest):
    def test_coulomb_matrix(self):
        # flat
        cm = CoulombMatrix(flatten=True)
        df = pd.DataFrame({"s": [self.diamond, self.nacl]})
        with self.assertRaises(NotFittedError):
            df = cm.featurize_dataframe(df, "s")
        df = cm.fit_featurize_dataframe(df, "s")
        labels = cm.feature_labels()
        self.assertListEqual(labels, ["coulomb matrix eig 0", "coulomb matrix eig 1"])
        self.assertArrayAlmostEqual(df[labels].iloc[0], [49.169453, 24.546758], decimal=5)
        self.assertArrayAlmostEqual(df[labels].iloc[1], [153.774731, 452.894322], decimal=5)

        # matrix
        species = ["C", "C", "H", "H"]
        coords = [[0, 0, 0], [0, 0, 1.203], [0, 0, -1.06], [0, 0, 2.263]]
        acetylene = Molecule(species, coords)
        morig = CoulombMatrix(flatten=False).featurize(acetylene)
        mtarget = [
            [36.858, 15.835391290, 2.995098235, 1.402827813],
            [15.835391290, 36.858, 1.4028278132103624, 2.9950982],
            [2.9368896127, 1.402827813, 0.5, 0.159279959],
            [1.4028278132, 2.995098235, 0.159279959, 0.5],
        ]
        self.assertAlmostEqual(int(np.linalg.norm(morig - np.array(mtarget))), 0)
        m = CoulombMatrix(diag_elems=False, flatten=False).featurize(acetylene)[0]
        self.assertAlmostEqual(m[0][0], 0.0)
        self.assertAlmostEqual(m[1][1], 0.0)
        self.assertAlmostEqual(m[2][2], 0.0)
        self.assertAlmostEqual(m[3][3], 0.0)

    def test_sine_coulomb_matrix(self):
        # flat
        scm = SineCoulombMatrix(flatten=True)
        df = pd.DataFrame({"s": [self.sc, self.ni3al]})
        with self.assertRaises(NotFittedError):
            df = scm.featurize_dataframe(df, "s")
        df = scm.fit_featurize_dataframe(df, "s")
        labels = scm.feature_labels()
        self.assertEqual(labels[0], "sine coulomb matrix eig 0")
        self.assertArrayAlmostEqual(df[labels].iloc[0], [235.740418, 0.0, 0.0, 0.0], decimal=5)
        self.assertArrayAlmostEqual(
            df[labels].iloc[1],
            [232.578562, 1656.288171, 1403.106576, 1403.106576],
            decimal=5,
        )

        # matrix
        scm = SineCoulombMatrix(flatten=False)
        sin_mat = scm.featurize(self.diamond)
        mtarget = [[36.8581, 6.147068], [6.147068, 36.8581]]
        self.assertAlmostEqual(np.linalg.norm(sin_mat - np.array(mtarget)), 0.0, places=4)
        scm = SineCoulombMatrix(diag_elems=False, flatten=False)
        sin_mat = scm.featurize(self.diamond)[0]
        self.assertEqual(sin_mat[0][0], 0)
        self.assertEqual(sin_mat[1][1], 0)

    def test_orbital_field_matrix(self):
        ofm_maker = OrbitalFieldMatrix(flatten=False)
        ofm = ofm_maker.featurize(self.diamond)[0]
        mtarget = np.zeros((32, 32))
        mtarget[1][1] = 1.4789015  # 1.3675444
        mtarget[1][3] = 1.4789015  # 1.3675444
        mtarget[3][1] = 1.4789015  # 1.3675444
        mtarget[3][3] = 1.4789015  # 1.3675444 if for a coord# of exactly 4
        for i in range(32):
            for j in range(32):
                if i not in [1, 3] and j not in [1, 3]:
                    self.assertEqual(ofm[i, j], 0.0)
        mtarget = np.matrix(mtarget)
        self.assertAlmostEqual(np.linalg.norm(ofm - mtarget), 0.0, places=4)

        ofm_maker = OrbitalFieldMatrix(True, flatten=False)
        ofm = ofm_maker.featurize(self.diamond)[0]
        mtarget = np.zeros((39, 39))
        mtarget[1][1] = 1.4789015
        mtarget[1][3] = 1.4789015
        mtarget[3][1] = 1.4789015
        mtarget[3][3] = 1.4789015
        mtarget[1][33] = 1.4789015
        mtarget[3][33] = 1.4789015
        mtarget[33][1] = 1.4789015
        mtarget[33][3] = 1.4789015
        mtarget[33][33] = 1.4789015
        mtarget = np.matrix(mtarget)
        self.assertAlmostEqual(np.linalg.norm(ofm - mtarget), 0.0, places=4)

        ofm_flat = OrbitalFieldMatrix(period_tag=False, flatten=True)
        self.assertEqual(len(ofm_flat.feature_labels()), 1024)

        ofm_flat = OrbitalFieldMatrix(period_tag=True, flatten=True)
        self.assertEqual(len(ofm_flat.feature_labels()), 1521)
        ofm_vector = ofm_flat.featurize(self.diamond)
        for ix in [40, 42, 72, 118, 120, 150, 1288, 1320]:
            self.assertAlmostEqual(ofm_vector[ix], 1.4789015345821415)


if __name__ == "__main__":
    unittest.main()
