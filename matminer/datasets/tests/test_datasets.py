import os
import unittest

import numpy as np
import requests
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype
from pymatgen.core.structure import Composition, Structure

from matminer.datasets.dataset_retrieval import load_dataset
from matminer.datasets.tests.base import DatasetTest, do_complete_test


class DataSetsTest(DatasetTest):
    # Runs tests common to all datasets,
    # makes it quicker to write tests for new datasets
    def universal_dataset_check(
        self,
        dataset_name,
        object_headers=None,
        numeric_headers=None,
        bool_headers=None,
        test_func=None,
    ):

        # "Hard" integrity checks that take a long time.
        # These tests only run if the MATMINER_DATASET_FULL_TEST
        # environment variable is set to True
        if do_complete_test:
            # Get rid of dataset if it's on the disk already
            data_path = os.path.join(
                self.dataset_dir,
                dataset_name + "." + self.dataset_dict[dataset_name]["file_type"],
            )
            if os.path.exists(data_path):
                os.remove(data_path)

            # Test that dataset can be downloaded
            load_dataset(dataset_name)
            self.assertTrue(os.path.exists(data_path))

            # Test that data is now available and has all its elements
            df = load_dataset(dataset_name, download_if_missing=False)
            self.assertEqual(len(df), self.dataset_dict[dataset_name]["num_entries"])

            # Test all columns are there
            self.assertEqual(
                sorted(list(df)),
                sorted(header for header in self.dataset_dict[dataset_name]["columns"].keys()),
            )

            # Test each column for appropriate type
            if object_headers is None:
                object_headers = []
            if numeric_headers is None:
                numeric_headers = []
            if bool_headers is None:
                bool_headers = []

            df = load_dataset(dataset_name, download_if_missing=False)
            if object_headers:
                self.assertTrue(is_object_dtype(df[object_headers].values))
            if numeric_headers:
                self.assertTrue(is_numeric_dtype(df[numeric_headers].values))
            if bool_headers:
                self.assertTrue(is_bool_dtype(df[bool_headers].values))

            # Make sure all columns are accounted for
            column_headers = object_headers + numeric_headers + bool_headers
            self.assertEqual(sorted(list(df)), sorted(column_headers))

            # Run tests unique to the dataset
            if test_func is not None:
                test_func(df)

        # "Soft" check that just makes sure the dataset download page is active
        # This runs when on a system with the CI environment var present
        # (e.g. when running a continuous integration VCS system)
        else:
            download_page = requests.head(self.dataset_dict[dataset_name]["url"])
            self.assertTrue(download_page.ok)


class MatminerDatasetsTest(DataSetsTest):
    """
    All datasets hosted with matminer are tested here, excluding matbench
    datasets.
    """

    def test_elastic_tensor_2015(self):
        object_headers = [
            "material_id",
            "formula",
            "structure",
            "compliance_tensor",
            "elastic_tensor",
            "elastic_tensor_original",
            "cif",
            "poscar",
        ]

        numeric_headers = [
            "nsites",
            "space_group",
            "volume",
            "elastic_anisotropy",
            "G_Reuss",
            "G_VRH",
            "G_Voigt",
            "K_Reuss",
            "K_VRH",
            "K_Voigt",
            "poisson_ratio",
            "kpoint_density",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            tensor_headers = [
                "compliance_tensor",
                "elastic_tensor",
                "elastic_tensor_original",
            ]
            for c in tensor_headers:
                self.assertEqual(type(df[c][0]), np.ndarray)

        self.universal_dataset_check(
            "elastic_tensor_2015",
            object_headers,
            numeric_headers,
            test_func=_unique_tests,
        )

    def test_piezoelectric_tensor(self):
        object_headers = [
            "material_id",
            "formula",
            "structure",
            "point_group",
            "v_max",
            "piezoelectric_tensor",
            "cif",
            "meta",
            "poscar",
        ]

        numeric_headers = ["nsites", "space_group", "volume", "eij_max"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            self.assertEqual(type(df["piezoelectric_tensor"][0]), np.ndarray)

        self.universal_dataset_check(
            "piezoelectric_tensor",
            object_headers,
            numeric_headers,
            test_func=_unique_tests,
        )

    def test_dielectric_constant(self):
        object_headers = [
            "material_id",
            "formula",
            "structure",
            "e_electronic",
            "e_total",
            "cif",
            "meta",
            "poscar",
        ]

        numeric_headers = [
            "nsites",
            "space_group",
            "volume",
            "band_gap",
            "n",
            "poly_electronic",
            "poly_total",
        ]

        bool_headers = ["pot_ferroelectric"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check(
            "dielectric_constant",
            object_headers,
            numeric_headers,
            bool_headers,
            test_func=_unique_tests,
        )

    def test_flla(self):
        object_headers = ["material_id", "formula", "structure"]

        numeric_headers = [
            "e_above_hull",
            "nsites",
            "formation_energy",
            "formation_energy_per_atom",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check("flla", object_headers, numeric_headers, test_func=_unique_tests)

    def test_castelli_perovskites(self):
        object_headers = ["structure", "formula"]

        numeric_headers = [
            "fermi level",
            "fermi width",
            "e_form",
            "mu_b",
            "vbm",
            "cbm",
            "gap gllbsc",
        ]

        bool_headers = ["gap is direct"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check(
            "castelli_perovskites",
            object_headers,
            numeric_headers,
            bool_headers=bool_headers,
            test_func=_unique_tests,
        )

    def test_boltztrap_mp(self):
        object_headers = ["structure", "formula", "mpid"]

        numeric_headers = ["pf_n", "pf_p", "s_n", "s_p", "m_n", "m_p"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check("boltztrap_mp", object_headers, numeric_headers, test_func=_unique_tests)

    def test_phonon_dielectric_mp(self):
        object_headers = ["structure", "formula", "mpid"]

        numeric_headers = ["eps_electronic", "eps_total", "last phdos peak"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check(
            "phonon_dielectric_mp",
            object_headers,
            numeric_headers,
            test_func=_unique_tests,
        )

    def test_glass_ternary_hipt(self):
        object_headers = ["formula", "system", "processing", "phase"]

        numeric_headers = ["gfa"]

        self.universal_dataset_check(
            "glass_ternary_hipt",
            object_headers,
            numeric_headers,
        )

    def test_double_perovskites_gap(self):
        object_headers = ["formula", "a_1", "b_1", "a_2", "b_2"]

        numeric_headers = ["gap gllbsc"]

        self.universal_dataset_check(
            "double_perovskites_gap",
            object_headers,
            numeric_headers,
        )

    def test_double_perovskites_gap_lumo(self):
        object_headers = ["atom"]

        numeric_headers = ["lumo"]

        self.universal_dataset_check(
            "double_perovskites_gap_lumo",
            object_headers,
            numeric_headers,
        )

    def test_mp_all_20181018(self):
        object_headers = ["mpid", "formula", "structure", "initial structure"]

        numeric_headers = [
            "e_hull",
            "gap pbe",
            "mu_b",
            "elastic anisotropy",
            "bulk modulus",
            "shear modulus",
            "e_form",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)

        self.universal_dataset_check("mp_all_20181018", object_headers, numeric_headers, test_func=_unique_tests)

    def test_mp_nostruct_20181018(self):
        object_headers = ["mpid", "formula"]

        numeric_headers = [
            "e_hull",
            "gap pbe",
            "mu_b",
            "elastic anisotropy",
            "bulk modulus",
            "shear modulus",
            "e_form",
        ]

        self.universal_dataset_check(
            "mp_nostruct_20181018",
            object_headers,
            numeric_headers,
        )

    def test_glass_ternary_landolt(self):
        object_headers = ["phase", "formula", "processing"]

        numeric_headers = ["gfa"]

        self.universal_dataset_check(
            "glass_ternary_landolt",
            object_headers,
            numeric_headers,
        )

    def test_citrine_thermal_conductivity(self):
        object_headers = ["k-units", "formula", "k_condition", "k_condition_units"]

        numeric_headers = ["k_expt"]

        self.universal_dataset_check("citrine_thermal_conductivity", object_headers, numeric_headers)

    def test_wolverton_oxides(self):
        object_headers = [
            "atom a",
            "formula",
            "atom b",
            "lowest distortion",
            "mu_b",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
        ]

        numeric_headers = ["e_form", "e_hull", "vpa", "gap pbe", "e_form oxygen"]

        self.universal_dataset_check("wolverton_oxides", object_headers, numeric_headers)

    def test_heusler_magnetic(self):
        object_headers = ["formula", "heusler type", "struct type"]

        numeric_headers = [
            "num_electron",
            "latt const",
            "tetragonality",
            "e_form",
            "pol fermi",
            "mu_b",
            "mu_b saturation",
        ]

        self.universal_dataset_check("heusler_magnetic", object_headers, numeric_headers)

    def test_steel_strength(self):
        object_headers = ["formula"]

        numeric_headers = [
            "c",
            "mn",
            "si",
            "cr",
            "ni",
            "mo",
            "v",
            "n",
            "nb",
            "co",
            "w",
            "al",
            "ti",
            "yield strength",
            "tensile strength",
            "elongation",
        ]

        self.universal_dataset_check("steel_strength", object_headers, numeric_headers)

    def test_jarvis_ml_dft_training(self):
        object_headers = ["jid", "mpid", "structure", "composition"]

        numeric_headers = [
            "e mass_x",
            "e mass_y",
            "e mass_z",
            "epsilon_x opt",
            "epsilon_y opt",
            "epsilon_z opt",
            "e_exfol",
            "e_form",
            "shear modulus",
            "hole mass_x",
            "hole mass_y",
            "hole mass_z",
            "bulk modulus",
            "mu_b",
            "gap tbmbj",
            "epsilon_x tbmbj",
            "epsilon_y tbmbj",
            "epsilon_z tbmbj",
            "gap opt",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            self.assertEqual(type(df["composition"][0]), Composition)

        self.universal_dataset_check(
            "jarvis_ml_dft_training",
            object_headers,
            numeric_headers,
            test_func=_unique_tests,
        )

    def test_jarvis_dft_3d(self):
        object_headers = [
            "jid",
            "mpid",
            "structure",
            "composition",
            "structure initial",
        ]

        numeric_headers = [
            "epsilon_x opt",
            "epsilon_y opt",
            "epsilon_z opt",
            "e_form",
            "shear modulus",
            "bulk modulus",
            "gap tbmbj",
            "epsilon_x tbmbj",
            "epsilon_y tbmbj",
            "epsilon_z tbmbj",
            "gap opt",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            self.assertEqual(type(df["composition"][0]), Composition)

        self.universal_dataset_check("jarvis_dft_3d", object_headers, numeric_headers, test_func=_unique_tests)

    def test_jarvis_dft_2d(self):
        object_headers = [
            "jid",
            "mpid",
            "structure",
            "composition",
            "structure initial",
        ]

        numeric_headers = [
            "epsilon_x opt",
            "epsilon_y opt",
            "epsilon_z opt",
            "exfoliation_en",
            "e_form",
            "gap tbmbj",
            "epsilon_x tbmbj",
            "epsilon_y tbmbj",
            "epsilon_z tbmbj",
            "gap opt",
        ]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            self.assertEqual(type(df["composition"][0]), Composition)

        self.universal_dataset_check("jarvis_dft_2d", object_headers, numeric_headers, test_func=_unique_tests)

    def test_glass_binary(self):
        object_headers = ["formula"]

        numeric_headers = ["gfa"]

        self.universal_dataset_check("glass_binary", object_headers, numeric_headers)

    def test_glass_binary_v2(self):
        object_headers = ["formula"]

        numeric_headers = ["gfa"]

        self.universal_dataset_check("glass_binary_v2", object_headers, numeric_headers)

    def test_m2ax(self):
        object_headers = ["formula"]

        numeric_headers = [
            "a",
            "c",
            "d_mx",
            "d_ma",
            "c11",
            "c12",
            "c13",
            "c33",
            "c44",
            "bulk modulus",
            "shear modulus",
            "elastic modulus",
        ]

        self.universal_dataset_check("m2ax", object_headers, numeric_headers)

    def test_expt_gap(self):
        object_headers = ["formula"]

        numeric_headers = ["gap expt"]

        self.universal_dataset_check("expt_gap", object_headers, numeric_headers)

    def test_expt_formation_enthalpy(self):
        object_headers = ["formula", "pearson symbol", "space group", "mpid"]

        numeric_headers = ["oqmdid", "e_form expt", "e_form mp", "e_form oqmd"]

        self.universal_dataset_check("expt_formation_enthalpy", object_headers, numeric_headers)

    def test_brgoch_superhard_training(self):
        object_headers = [
            "formula",
            "material_id",
            "structure",
            "composition",
            "brgoch_feats",
        ]

        numeric_headers = ["shear_modulus", "bulk_modulus"]

        bool_headers = ["suspect_value"]

        def _unique_tests(df):
            self.assertEqual(type(df["structure"][0]), Structure)
            self.assertEqual(type(df["composition"][0]), Composition)
            self.assertTrue(isinstance(df["brgoch_feats"][0], dict))

        self.universal_dataset_check(
            "brgoch_superhard_training",
            object_headers,
            numeric_headers,
            bool_headers,
            test_func=_unique_tests,
        )

    def test_expt_gap_kingsbury(self):
        object_headers = ["formula", "likely_mpid"]

        numeric_headers = ["expt_gap"]
        self.universal_dataset_check("expt_gap_kingsbury", object_headers, numeric_headers)

    def test_expt_formation_enthalpy_kingsbury(self):

        object_headers = ["formula", "likely_mpid", "phaseinfo", "reference"]

        numeric_headers = ["expt_form_e", "uncertainty"]

        self.universal_dataset_check(
            "expt_formation_enthalpy_kingsbury",
            object_headers,
            numeric_headers,
        )

    def test_ricci_boltztrap_mp_tabular(self):
        object_headers = ["structure", "task", "functional", "pretty_formula", "is_metal"]

        numeric_headers = [
            "ΔE [eV]",
            "V [Å³]",
            "S.p [µV/K]",
            "S.n [µV/K]",
            "Sᵉ.p.v [µV/K]",
            "Sᵉ.p.T [K]",
            "Sᵉ.p.c [cm⁻³]",
            "Sᵉ.n.v [µV/K]",
            "Sᵉ.n.T [K]",
            "Sᵉ.n.c [cm⁻³]",
            "σ.p [1/Ω/m/s]",
            "σ.n [1/Ω/m/s]",
            "PF.p [µW/cm/K²/s]",
            "PF.n [µW/cm/K²/s]",
            "σᵉ.p.v [1/Ω/m/s]",
            "σᵉ.p.T [K]",
            "σᵉ.p.c [cm⁻³]",
            "σᵉ.n.v [1/Ω/m/s]",
            "σᵉ.n.T [K]",
            "σᵉ.n.c [cm⁻³]",
            "PFᵉ.p.v [µW/cm/K²/s]",
            "PFᵉ.p.T [K]",
            "PFᵉ.p.c [cm⁻³]",
            "PFᵉ.n.v [µW/cm/K²/s]",
            "PFᵉ.n.T [K]",
            "PFᵉ.n.c [cm⁻³]",
            "κₑ.p [W/K/m/s]",
            "κₑ.n [W/K/m/s]",
            "κₑᵉ.p.v [W/K/m/s]",
            "κₑᵉ.p.T [K]",
            "κₑᵉ.p.c [cm⁻³]",
            "κₑᵉ.n.v [W/K/m/s]",
            "κₑᵉ.n.T [K]",
            "κₑᵉ.n.c [cm⁻³]",
            "mₑᶜ.p.ε̄ [mₑ]",
            "mₑᶜ.p.ε₁ [mₑ]",
            "mₑᶜ.p.ε₂ [mₑ]",
            "mₑᶜ.p.ε₃ [mₑ]",
            "mₑᶜ.n.ε̄ [mₑ]",
            "mₑᶜ.n.ε₁ [mₑ]",
            "mₑᶜ.n.ε₂ [mₑ]",
            "mₑᶜ.n.ε₃ [mₑ]",
        ]

        self.universal_dataset_check("ricci_boltztrap_mp_tabular", object_headers, numeric_headers)

    def test_superconductivity2018(self):
        object_headers = ["composition"]
        numeric_headers = ["Tc"]

        self.universal_dataset_check("superconductivity2018", object_headers, numeric_headers)


class MatbenchDatasetsTest(DataSetsTest):
    """
    Matbench datasets are tested here.
    """

    def test_matbench_v0_1(self):
        structure_key = "structure"
        composition_key = "composition"
        config_regression = {
            "matbench_dielectric": ["n", structure_key],
            "matbench_expt_gap": ["gap expt", composition_key],
            "matbench_jdft2d": ["exfoliation_en", structure_key],
            "matbench_log_gvrh": ["log10(G_VRH)", structure_key],
            "matbench_log_kvrh": ["log10(K_VRH)", structure_key],
            "matbench_mp_e_form": ["e_form", structure_key],
            "matbench_perovskites": ["e_form", structure_key],
            "matbench_phonons": ["last phdos peak", structure_key],
            "matbench_steels": ["yield strength", composition_key],
        }

        config_classification = {
            "matbench_expt_is_metal": ["is_metal", composition_key],
            "matbench_glass": ["gfa", composition_key],
            "matbench_mp_is_metal": ["is_metal", structure_key],
        }

        clf = "classification"
        reg = "regression"
        config = {clf: config_classification, reg: config_regression}

        for problem_type, problems_config in config.items():
            for ds, ds_config in problems_config.items():
                object_headers = [ds_config[1]]
                if problem_type == clf:
                    numeric_headers = None
                    bool_headers = [ds_config[0]]
                else:
                    numeric_headers = [ds_config[0]]
                    bool_headers = None

                self.universal_dataset_check(ds, object_headers, numeric_headers, bool_headers)


if __name__ == "__main__":
    unittest.main()
