from unittest import TestCase
import unittest
import os
import json
import csv
from matminer.featurizers.utils.cgcnn import CIFDataWrapper, \
    CrystalGraphConvNetWrapper, appropriate_kwargs, AtomCustomArrayInitializer
from pymatgen.core import Structure, Lattice
try:
    import cgcnn
    import torch
except ImportError:
    torch = None
    cgcnn = None


class TestCGCNNWrappers(TestCase):
    @unittest.skipIf(not (torch and cgcnn),
                     "pytorch or cgcnn not installed.")
    def setUp(self):
        self.sc = Structure(Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
                            ["Al"], [[0, 0, 0]], validate_proximity=False,
                            to_unit_cell=False, coords_are_cartesian=False)

        self.target = torch.Tensor([1, 0, 1])

    def test_get_cgcnn_data(self):
        id_prop_data, _, struct_list = self._get_cgcnn_data()
        self.assertEqual(len(id_prop_data), len(struct_list))

    def test_cifdatawrapper(self):
        cifdatawrapper = CIFDataWrapper([self.sc], [0], {13: [-1, -1]})
        self.assertEqual(len(cifdatawrapper), 1)
        self.assertEqual(len(cifdatawrapper[0]), 3)
        self.assertEqual(len(cifdatawrapper[0][0]), 3)

    def test_crystal_graph_convnet_wrapper(self):
        model = CrystalGraphConvNetWrapper(2, 41, classification=True)
        state_dict = model.state_dict()
        self.assertEqual(state_dict['embedding.weight'].size(),
                         torch.Size([64, 2]))
        self.assertEqual(state_dict['embedding.bias'].size(),
                         torch.Size([64]))
        self.assertEqual(state_dict['convs.0.fc_full.weight'].size(),
                         torch.Size([128, 169]))
        self.assertEqual(state_dict['convs.1.bn1.weight'].size(),
                         torch.Size([128]))
        self.assertEqual(state_dict['convs.2.bn2.bias'].size(),
                         torch.Size([64]))
        self.assertEqual(state_dict['conv_to_fc.weight'].size(),
                         torch.Size([128, 64]))
        self.assertEqual(state_dict['fc_out.weight'].size(),
                         torch.Size([2, 128]))

    def test_appropriate_kwargs(self):
        init_dict = {'task': 'classification', 'no_task': True}
        arange_dict = appropriate_kwargs(init_dict, self._get_cgcnn_data)
        self.assertEqual(set(arange_dict.keys()), {'task'})
        self.assertEqual(arange_dict['task'], 'classification')

    def test_atom_custom_array_initializer(self):
        atom_initializer = AtomCustomArrayInitializer({13: [-1, -1]})
        self.assertEqual(set(atom_initializer.get_atom_fea(13)), {-1.})

    @staticmethod
    def _get_cgcnn_data(task="classification"):
        """
        Get cgcnn sample data.
        Args:
            task (str): Classification or regression,
                        decided which sample data to return.

        Returns:
            id_prop_data (list): List of property data.
            elem_embedding (list): List of element features.
            struct_list (list): List of structure object.
        """
        if task == "classification":
            cgcnn_data_path = os.path.join(os.path.dirname(cgcnn.__file__),
                                           "..", "data", "sample-classification")
        else:
            cgcnn_data_path = os.path.join(os.path.dirname(cgcnn.__file__),
                                           "..", "data", "sample-regression")

        struct_list = list()
        cif_list = list()
        with open(os.path.join(cgcnn_data_path, "id_prop.csv")) as f:
            reader = csv.reader(f)
            id_prop_data = [row[1] for row in reader]
        with open(os.path.join(cgcnn_data_path, "atom_init.json")) as f:
            elem_embedding = json.load(f)

        for file in os.listdir(cgcnn_data_path):
            if file.endswith('.cif'):
                cif_list.append(int(file[:-4]))
                cif_list = sorted(cif_list)
        for cif_name in cif_list:
            crystal = Structure.from_file(os.path.join(cgcnn_data_path,
                                                       '{}.cif'.format(
                                                           cif_name)))
            struct_list.append(crystal)
        return id_prop_data, elem_embedding, struct_list
