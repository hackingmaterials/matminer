import unittest
import os

from matminer.datasets.utils import _load_dataset_dict

# Set global flag based on environment variable
# specifying whether or not to run full test or partial test
_dataset_test_env_var = os.environ.get("MATMINER_DATASET_FULL_TEST", "False")
do_complete_test = (_dataset_test_env_var.upper() == "TRUE")


class DataSetTest(unittest.TestCase):
    def setUp(self):
        self.dataset_names = [
            'flla',
            'elastic_tensor_2015',
            'piezoelectric_tensor',
            'dielectric_constant',
            'castelli_perovskites',
            'boltztrap_mp',
            'phonon_dielectric_mp',
            'glass_ternary_hipt',
            'double_perovskites_gap',
            'double_perovskites_gap_lumo',
            'mp_all',
            'mp_nostruct',
            'glass_ternary_landolt',
            'citrine_thermal_conductivity',
            'wolverton_oxides',
            'heusler_magnetic',
            'steel_strength',
            'jarvis_ml_dft_training',
            'jarvis_dft_2d',
            'jarvis_dft_3d',
            'glass_binary',
            'glass_binary_v2',
            'm2ax',
            'expt_gap',
            'expt_formation_enthalpy',
            'brgoch_superhard_training'
        ]
        self.dataset_attributes = [
            'file_type',
            'url',
            'hash',
            'reference',
            'description',
            'columns',
            'bibtex_refs',
            'num_entries'
        ]

        self.dataset_dict = _load_dataset_dict()

        # current directory, for storing and discarding test_dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # directory where in-use datasets should be stored,
        # either at MATMINER_DATA env var or under matminer/datasets/
        self.dataset_dir = os.environ.get(
            "MATMINER_DATA",
            os.path.abspath(os.path.join(current_dir, os.pardir))
        )

        # Shared set up for test_validate_dataset & test_fetch_external_dataset
        self._path = os.path.join(current_dir, "test_dataset.csv")
        self._url = "https://ndownloader.figshare.com/files/13039562"
        self._hash = "c487f59ce0d48505c36633b4b202027" \
                     "d0c915474b081e8fb0bde8d5474ee59a1"


if __name__ == "__main__":
    unittest.main()
