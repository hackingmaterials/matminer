import json
import math
import unittest

from monty.json import MontyEncoder
from unittest import TestCase
from pandas import DataFrame, MultiIndex

from pymatgen.core.structure import IStructure
from pymatgen.core import Composition, Lattice, Structure, Element
from pymatgen.settings import SETTINGS

from matminer.featurizers.conversions import (
    StrToComposition, StructureToComposition, StructureToIStructure,
    DictToObject, JsonToObject, StructureToOxidStructure,
    CompositionToOxidComposition, CompositionToStructureFromMP)


class TestConversions(TestCase):

    def test_conversion_overwrite(self):
        # Test with overwrite
        d = {'comp_str': ["Fe2", "MnO2"]}
        df = DataFrame(data=d)

        stc = StrToComposition(target_col_id='comp_str', overwrite_data=False)
        with self.assertRaises(ValueError):
            df = stc.featurize_dataframe(df, 'comp_str', inplace=True)

        with self.assertRaises(ValueError):
            df = stc.featurize_dataframe(df, 'comp_str', inplace=False)

        stc = StrToComposition(target_col_id='comp_str', overwrite_data=True)

        dfres_ipt = df.copy()
        stc.featurize_dataframe(dfres_ipt, 'comp_str', inplace=True)
        self.assertListEqual(dfres_ipt.columns.tolist(), ["comp_str"])

        dfres_ipf = stc.featurize_dataframe(df, 'comp_str', inplace=False)
        self.assertListEqual(dfres_ipf.columns.tolist(), ["comp_str"])

    def test_str_to_composition(self):
        d = {'comp_str': ["Fe2", "MnO2"]}

        df = DataFrame(data=d)
        df = StrToComposition().featurize_dataframe(df, 'comp_str')

        self.assertEqual(df["composition"].tolist(),
                         [Composition("Fe2"), Composition("MnO2")])

        stc = StrToComposition(reduce=True, target_col_id='composition_red')
        df = stc.featurize_dataframe(df, 'comp_str')

        self.assertEqual(df["composition_red"].tolist(),
                         [Composition("Fe"), Composition("MnO2")])

    def test_structure_to_composition(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        df = DataFrame(data={'structure': [struct]})

        stc = StructureToComposition()
        df = stc.featurize_dataframe(df, 'structure')
        self.assertEqual(df["composition"].tolist()[0], Composition("Si2"))

        stc = StructureToComposition(reduce=True,
                                     target_col_id='composition_red')
        df = stc.featurize_dataframe(df, 'structure')
        self.assertEqual(df["composition_red"].tolist()[0], Composition("Si"))

    def test_dict_to_object(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        d = {'structure_dict': [struct.as_dict(), struct.as_dict()]}
        df = DataFrame(data=d)

        dto = DictToObject(target_col_id='structure')
        df = dto.featurize_dataframe(df, 'structure_dict')
        self.assertEqual(df["structure"].tolist()[0], struct)
        self.assertEqual(df["structure"].tolist()[1], struct)

        # test dynamic target_col_id setting
        df = DataFrame(data=d)
        dto = DictToObject()
        df = dto.featurize_dataframe(df, 'structure_dict')
        self.assertEqual(df["structure_dict_object"].tolist()[0], struct)
        self.assertEqual(df["structure_dict_object"].tolist()[1], struct)

    def test_json_to_object(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        struct_json = json.dumps(struct, cls=MontyEncoder)

        d = {'structure_json': [struct_json]}
        df = DataFrame(data=d)

        jto = JsonToObject(target_col_id='structure')
        df = jto.featurize_dataframe(df, 'structure_json')
        self.assertEqual(df["structure"].tolist()[0], struct)

        # test dynamic target_col_id setting
        df = DataFrame(data=d)
        jto = JsonToObject()
        df = jto.featurize_dataframe(df, 'structure_json')
        self.assertEqual(df["structure_json_object"].tolist()[0], struct)

    def test_structure_to_oxidstructure(self):
        cscl = Structure(Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                         ["Cl", "Cs"], [[0.45, 0.5, 0.5], [0, 0, 0]])
        d = {'structure': [cscl]}
        df = DataFrame(data=d)

        sto = StructureToOxidStructure()
        df = sto.featurize_dataframe(df, 'structure')
        self.assertEqual(df["structure_oxid"].tolist()[0][0].specie.oxi_state,
                         -1)
        self.assertEqual(df["structure_oxid"].tolist()[0][1].specie.oxi_state,
                         +1)

        sto = StructureToOxidStructure(target_col_id='structure_oxid2',
                                       oxi_states_override={"Cl": [-2],
                                                            "Cs": [+2]})
        df = sto.featurize_dataframe(df, 'structure')
        self.assertEqual(df["structure_oxid2"].tolist()[0][0].specie.oxi_state,
                         -2)
        self.assertEqual(df["structure_oxid2"].tolist()[0][1].specie.oxi_state,
                         +2)

        # original is preserved
        self.assertEqual(df["structure"].tolist()[0][0].specie, Element("Cl"))

        # test in-place
        sto = StructureToOxidStructure(target_col_id=None, overwrite_data=True)
        df = sto.featurize_dataframe(df, 'structure')
        self.assertEqual(df["structure"].tolist()[0][0].specie.oxi_state, -1)

        # test error handling
        test_struct = Structure([5, 0, 0, 0, 5, 0, 0, 0, 5], ['Sb', 'F', 'O'],
                                [[0, 0, 0], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]])
        df = DataFrame(data={'structure': [test_struct]})
        sto = StructureToOxidStructure(return_original_on_error=False,
                                       max_sites=2)
        self.assertRaises(ValueError, sto.featurize_dataframe, df,
                          'structure')

        # check non oxi state structure returned correctly
        sto = StructureToOxidStructure(return_original_on_error=True,
                                       max_sites=2)
        df = sto.featurize_dataframe(df, 'structure')
        self.assertEqual(df["structure_oxid"].tolist()[0][0].specie,
                         Element("Sb"))

    def test_composition_to_oxidcomposition(self):
        df = DataFrame(data={"composition": [Composition("Fe2O3")]})
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition')
        self.assertEqual(df["composition_oxid"].tolist()[0],
                         Composition({"Fe3+": 2, "O2-": 3}))

        # test error handling
        df = DataFrame(data={"composition": [Composition("Fe2O3")]})
        cto = CompositionToOxidComposition(
            return_original_on_error=False, max_sites=2)
        self.assertRaises(ValueError, cto.featurize_dataframe, df,
                          'composition')

        # check non oxi state structure returned correctly
        cto = CompositionToOxidComposition(
            return_original_on_error=True, max_sites=2)
        df = cto.featurize_dataframe(df, 'composition')
        self.assertEqual(df["composition_oxid"].tolist()[0],
                         Composition({"Fe": 2, "O": 3}))

    def test_to_istructure(self):
        cscl = Structure(Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                         ["Cl", "Cs"], [[0.45, 0.5, 0.5], [0, 0, 0]])
        df = DataFrame({"structure": [cscl]})

        # Run the conversion
        sti = StructureToIStructure()
        df = sti.featurize_dataframe(df, 'structure')

        # Make sure the new structure is an IStructure, and equal
        # to the original structure
        self.assertIsInstance(df["istructure"][0], IStructure)
        self.assertEqual(df["istructure"][0], df["structure"][0])

    def test_conversion_multiindex(self):
        d = {'comp_str': ["Fe2", "MnO2"]}

        df_1lvl = DataFrame(data=d)

        df_1lvl = StrToComposition().featurize_dataframe(
            df_1lvl, 'comp_str', multiindex=True)
        self.assertEqual(df_1lvl[("StrToComposition", "composition")].tolist(),
                         [Composition("Fe2"), Composition("MnO2")])

        df_2lvl = DataFrame(data=d)
        df_2lvl.columns = MultiIndex.from_product((["custom"],
                                                   df_2lvl.columns.values))

        df_2lvl = StrToComposition().featurize_dataframe(
            df_2lvl, ("custom", "comp_str"), multiindex=True)
        self.assertEqual(df_2lvl[("StrToComposition", "composition")].tolist(),
                         [Composition("Fe2"), Composition("MnO2")])

        df_2lvl = DataFrame(data=d)
        df_2lvl.columns = MultiIndex.from_product((["custom"],
                                                   df_2lvl.columns.values))

        sto = StrToComposition(target_col_id='test')
        df_2lvl = sto.featurize_dataframe(
            df_2lvl, ("custom", "comp_str"), multiindex=True)
        self.assertEqual(df_2lvl[("StrToComposition", "test")].tolist(),
                         [Composition("Fe2"), Composition("MnO2")])

        # if two level multiindex provided as target, it should be written there
        # here we test converting multiindex in place
        df_2lvl = DataFrame(data=d)
        df_2lvl.columns = MultiIndex.from_product((["custom"],
                                                   df_2lvl.columns.values))

        sto = StrToComposition(target_col_id=None, overwrite_data=True)

        df_2lvl = sto.featurize_dataframe(
            df_2lvl, ("custom", "comp_str"), multiindex=True, inplace=False)
        self.assertEqual(df_2lvl[("custom", "comp_str")].tolist(),
                         [Composition("Fe2"), Composition("MnO2")])

        # Try inplace multiindex conversion with return errors
        df_2lvl = DataFrame(data=d)
        df_2lvl.columns = MultiIndex.from_product((["custom"],
                                                   df_2lvl.columns.values))

        sto = StrToComposition(target_col_id=None, overwrite_data=True)
        df_2lvl = sto.featurize_dataframe(
            df_2lvl, ("custom", "comp_str"), multiindex=True,
            return_errors=True, ignore_errors=True)

        self.assertTrue(
            all(df_2lvl[("custom", "StrToComposition Exceptions")].isnull()))

    def test_conversion_multiindex_dynamic(self):
        # test dynamic target_col_id setting with multiindex

        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice([[3.8401979337, 0.00, 0.00],
                           [1.9200989668, 3.3257101909, 0.00],
                           [0.00, -2.2171384943, 3.1355090603]])
        struct = Structure(lattice, ["Si"] * 2, coords)
        d = {'structure_dict': [struct.as_dict(), struct.as_dict()]}
        df_2lvl = DataFrame(data=d)
        df_2lvl.columns = MultiIndex.from_product((["custom"],
                                                   df_2lvl.columns.values))

        dto = DictToObject()
        df_2lvl = dto.featurize_dataframe(df_2lvl, ('custom', 'structure_dict'),
                                          multiindex=True)
        new_col_id = ('DictToObject', 'structure_dict_object')
        self.assertEqual(df_2lvl[new_col_id].tolist()[0], struct)
        self.assertEqual(df_2lvl[new_col_id].tolist()[1], struct)

    @unittest.skipIf(not SETTINGS.get("PMG_MAPI_KEY", ""),
                     "PMG_MAPI_KEY not in environment variables.")
    def test_composition_to_structurefromMP(self):
        df = DataFrame(data={"composition": [Composition("Fe2O3"),
                                             Composition("N9Al34Fe234")]})

        cto = CompositionToStructureFromMP()
        df = cto.featurize_dataframe(df, 'composition')
        structures = df["structure"].tolist()
        self.assertTrue(isinstance(structures[0], Structure))
        self.assertGreaterEqual(len(structures[0]), 5)  # has at least 5 sites
        self.assertTrue(math.isnan(structures[1]))
