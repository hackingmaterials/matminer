from unittest import TestCase

from monty.json import MontyEncoder
from pandas import DataFrame

import json

from pymatgen.core.structure import IStructure

from matminer.utils.conversions import dict_to_object, \
    structure_to_oxidstructure, \
    str_to_composition, json_to_object, structure_to_composition, \
    composition_to_oxidcomposition, structure_to_istructure
from pymatgen import Composition, Lattice, Structure, Element


class TestConversions(TestCase):

    def test_str_to_composition(self):
        d = {'comp_str': ["Fe2", "MnO2"]}

        df = DataFrame(data=d)
        df["composition"] = str_to_composition(df["comp_str"])
        self.assertEqual(df["composition"].tolist(), [Composition("Fe2"),
                                                      Composition("MnO2")])

        df["composition_red"] = str_to_composition(df["comp_str"], reduce=True)
        self.assertEqual(df["composition_red"].tolist(), [Composition("Fe"),
                                                           Composition("MnO2")])

    def test_structure_to_composition(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        df = DataFrame(data={'structure': [struct]})

        df["composition"] = structure_to_composition(df["structure"])
        self.assertEqual(df["composition"].tolist()[0], Composition("Si2"))

        df["composition_red"] = structure_to_composition(df["structure"],
                                                         reduce=True)
        self.assertEqual(df["composition_red"].tolist()[0], Composition("Si"))

    def test_dict_to_object(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        d = {'structure_dict': [struct.as_dict(), struct.as_dict()]}
        df = DataFrame(data=d)

        df["structure"] = dict_to_object(df["structure_dict"])
        self.assertEqual(df["structure"].tolist()[0], struct)
        self.assertEqual(df["structure"].tolist()[1], struct)


    def test_json_to_object(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        struct_json = json.dumps(struct, cls=MontyEncoder)

        d = {'structure_json': [struct_json]}
        df = DataFrame(data=d)

        df["structure"] = json_to_object(df["structure_json"])
        self.assertEqual(df["structure"].tolist()[0], struct)

    def test_structure_to_oxidstructure(self):
        cscl = Structure(Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                         ["Cl", "Cs"], [[0.45, 0.5, 0.5], [0, 0, 0]])
        d = {'structure': [cscl]}
        df = DataFrame(data=d)

        df["struct_oxid"] = structure_to_oxidstructure(df["structure"])
        self.assertEqual(df["struct_oxid"].tolist()[0][0].specie.oxi_state, -1)
        self.assertEqual(df["struct_oxid"].tolist()[0][1].specie.oxi_state, +1)

        df["struct_oxid2"] = structure_to_oxidstructure(df["structure"], oxi_states_override={"Cl": [-2], "Cs": [+2]})
        self.assertEqual(df["struct_oxid2"].tolist()[0][0].specie.oxi_state, -2)
        self.assertEqual(df["struct_oxid2"].tolist()[0][1].specie.oxi_state, +2)

        # original is preserved
        self.assertEqual(df["structure"].tolist()[0][0].specie, Element("Cl"))

        # test in-place
        structure_to_oxidstructure(df["structure"], inplace=True)
        self.assertEqual(df["structure"].tolist()[0][0].specie.oxi_state, -1)

    def test_composition_to_oxidcomposition(self):
        df = DataFrame(data={"composition": [Composition("Fe2O3")]})
        df["composition_oxid"] = composition_to_oxidcomposition(df["composition"])
        self.assertEqual(df["composition_oxid"].tolist()[0], Composition({"Fe3+": 2, "O2-":3}))

    def test_to_istructure(self):
        cscl = Structure(Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl", "Cs"], [[0.45, 0.5, 0.5], [0, 0, 0]])
        df = DataFrame({"structure": [cscl]})

        # Run the conversion
        df["istructure"] = structure_to_istructure(df["structure"])

        # Make sure the new structure is an IStructure, and equal
        #  to the original structure
        self.assertIsInstance(df["istructure"][0], IStructure)
        self.assertEqual(df["istructure"][0], df["structure"][0])
