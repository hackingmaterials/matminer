from unittest import TestCase

from monty.json import MontyEncoder
from pandas import DataFrame

import json

from matminer.utils.conversions import dict_to_object, struct_to_oxidstruct, \
    str_to_composition, json_to_object, structure_to_composition
from pymatgen import Composition, Lattice, Structure, Element


class TestConversions(TestCase):

    def test_str_to_composition(self):
        d = {'comp_str': ["Fe", "MnO2"]}

        df = DataFrame(data=d)
        df["composition"] = str_to_composition(df["comp_str"])
        self.assertEqual(df["composition"].tolist(), [Composition("Fe"),
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


    def struct_to_oxidstruct(self):
        cscl = Structure(Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                         ["Cl", "Cs"], [[0.45, 0.5, 0.5], [0, 0, 0]])
        d = {'structure': [cscl]}
        df = DataFrame(data=d)

        df["struct_oxid"] = struct_to_oxidstruct(df["structure"])
        self.assertEqual(df["struct_oxid"].tolist()[0][0].specie.oxi_state, -1)
        self.assertEqual(df["struct_oxid"].tolist()[0][1].specie.oxi_state, +1)

        df["struct_oxid2"] = struct_to_oxidstruct(df["structure"], oxi_states_override={"Cl": [-2], "Cs": [+2]})
        self.assertEqual(df["struct_oxid2"].tolist()[0][0].specie.oxi_state, -2)
        self.assertEqual(df["struct_oxid2"].tolist()[0][1].specie.oxi_state, +2)

        # original is preserved
        self.assertEqual(df["structure"].tolist()[0][0].specie, Element("Cl"))

        # test in-place
        struct_to_oxidstruct(df["structure"], inplace=True)
        self.assertEqual(df["structure"].tolist()[0][0].specie.oxi_state, -1)
