from pymatgen.core import Lattice, Structure
from pymatgen.util.testing import PymatgenTest


class SiteFeaturizerTest(PymatgenTest):
    def setUp(self):
        self.sc = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            [
                "Al",
            ],
            [[0, 0, 0]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
        )
        self.cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"],
            [[0.45, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
        )
        self.b1 = Structure(
            Lattice([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            ["H", "He"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
        )
        self.diamond = Structure(
            Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264], [0, 0, 2.528]]),
            ["C0+", "C0+"],
            [[2.554, 1.806, 4.423], [0.365, 0.258, 0.632]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=True,
            site_properties=None,
        )
        self.nacl = Structure(
            Lattice([[3.485, 0, 2.012], [1.162, 3.286, 2.012], [0, 0, 4.025]]),
            ["Na1+", "Cl1-"],
            [[0, 0, 0], [2.324, 1.643, 4.025]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=True,
            site_properties=None,
        )
        self.ni3al = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            [
                "Al",
            ]
            + ["Ni"] * 3,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
            site_properties=None,
        )

    def tearDown(self):
        del self.sc
        del self.cscl
