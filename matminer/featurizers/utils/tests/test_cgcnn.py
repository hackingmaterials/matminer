from unittest import TestCase

import torch
from matminer.featurizers.utils.cgcnn import DatasetWrapper, \
    CrystalGraphConvNetWrapper, appropriate_kwargs, get_cgcnn_data, mae, \
    class_eval, AtomCustomArrayInitializer, AverageMeter, Normalizer
from pymatgen.core import Structure, Lattice


class TestPropertyStats(TestCase):
    def setUp(self):
        self.sc = Structure(Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
                            ["Al"], [[0, 0, 0]], validate_proximity=False,
                            to_unit_cell=False, coords_are_cartesian=False)

        self.target = torch.Tensor([1, 0, 1])

    def test_datasetwrapper(self):
        datasetwrapper = DatasetWrapper([self.sc], [0], {13: [-1, -1]})
        self.assertEqual(datasetwrapper.strcs, [self.sc])
        self.assertEqual(len(datasetwrapper), 1)
        self.assertEqual(len(datasetwrapper[0]), 3)
        self.assertEqual(len(datasetwrapper[0][0]), 3)

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
        arange_dict = appropriate_kwargs(init_dict, get_cgcnn_data)
        self.assertEqual(set(arange_dict.keys()), {'task'})
        self.assertEqual(arange_dict['task'], 'classification')

    def test_get_cgcnn_data(self):
        id_prop_data, _, struct_list = get_cgcnn_data()
        self.assertEqual(len(id_prop_data), len(struct_list))

    def test_mae(self):
        mae_result = mae(torch.Tensor([1]), torch.Tensor([1]))
        self.assertEqual(mae_result[0], 0)

    def test_class_eval(self):
        prediction = torch.Tensor([[0, 1], [1, 1], [2, 1]])
        accuracy, precision, recall, fscore, auc_score = \
            class_eval(prediction, self.target)
        self.assertEqual(accuracy, 2/3)
        self.assertEqual(precision, 1.)
        self.assertEqual(recall, 0.5)
        self.assertEqual(fscore, 2/3)
        self.assertEqual(auc_score, 0.5)

    def test_atom_custom_array_initializer(self):
        atom_initializer = AtomCustomArrayInitializer({13: [-1, -1]})
        self.assertEqual(set(atom_initializer.get_atom_fea(13)), {-1.})

    def test_average_meter(self):
        atom_initializer = AverageMeter()
        self.assertEqual(atom_initializer.val, 0)
        self.assertEqual(atom_initializer.avg, 0)
        self.assertEqual(atom_initializer.sum, 0)
        self.assertEqual(atom_initializer.count, 0)

    def test_normalizer(self):
        target_norm = [0.5774, -1.1547,  0.5774]
        target_array = self.target.numpy()
        normalizer = Normalizer(self.target)
        norm_value = normalizer.norm(self.target).numpy()
        self.assertAlmostEqual(norm_value[0], target_norm[0], 4)
        self.assertAlmostEqual(norm_value[1], target_norm[1], 4)
        self.assertAlmostEqual(norm_value[2], target_norm[2], 4)

        denorm_value = normalizer.denorm(torch.Tensor(target_norm)).numpy()
        self.assertAlmostEqual(denorm_value[0], target_array[0], 4)
        self.assertAlmostEqual(denorm_value[1], target_array[1], 4)
        self.assertAlmostEqual(denorm_value[2], target_array[2], 4)

        state_dict = normalizer.state_dict()
        self.assertAlmostEqual(state_dict['mean'].numpy(), 2/3)
        self.assertAlmostEqual(state_dict['std'].numpy(), 0.5774, 4)

