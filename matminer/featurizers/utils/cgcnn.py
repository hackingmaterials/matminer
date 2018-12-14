import inspect
import functools
import warnings
import numpy as np
import random
from monty.dev import requires

try:
    import torch
    from torch.utils.data import Dataset
    import cgcnn
    import cgcnn.data as cgcnn_data
    from cgcnn.data import AtomInitializer
    from cgcnn.model import CrystalGraphConvNet
except ImportError:
    torch, Dataset = None, object
    cgcnn, cgcnn_data = None, None
    AtomInitializer, CrystalGraphConvNet = object, object


class CIFDataWrapper(Dataset):
    """
    Wrapper for a dataset containing pymatgen Structure objects.
    This is modified from CGCNN repo's CIFData for wrapping dataset where the
    structures are stored in CIF files.
    As we already have X as an iterable of pymatgen Structure objects, we can
    use this wrapper instead of CIFData.
    """
    @requires(torch and cgcnn,
              "CIFDataWrapper requires pytorch and cgcnn to be installed with "
              "Python bindings. Please get it at http://pytorch.org and "
              "https://github.com/txie-93/cgcnn.")
    def __init__(self, X, y, atom_init_fea, max_num_nbr=12, radius=8,
                 dmin=0, step=0.2, random_seed=123):
        """
        Args:
            X (Series/list): An iterable of pymatgen Structure objects.
            y (Series/list): target property that CGCNN is to predict.
            atom_init_fea (dict): A dict of {atom type: atom feature}.
            max_num_nbr (int): The max number of every atom's neighbors.
            radius (float): Cutoff radius for searching neighbors.
            dmin (int): The minimum distance for constructing GaussianDistance.
            step (float): The step size for constructing GaussianDistance.
            random_seed (int): Random seed for shuffling the dataset.
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.target_data = list(zip(range(len(y)), y))
        random.seed(random_seed)
        random.shuffle(self.target_data)
        self.structures = X
        self.ari = AtomCustomArrayInitializer(atom_init_fea)
        self.gdf = \
            cgcnn_data.GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.target_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        atom_idx, target = self.target_data[idx]
        crystal = self.structures[atom_idx]
        atom_fea = np.vstack(
            [self.ari.get_atom_fea(crystal[i].specie.number)
             for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius,
                                             include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    '{} not find enough neighbors to build graph. '
                    'If it happens frequently, consider increase '
                    'radius.'.format(atom_idx))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, atom_idx


class CrystalGraphConvNetWrapper(CrystalGraphConvNet):
    """
    Wrapper for CrystalGraphConvNet in the CGCNN repo and add extract_feature
    function to extract the feature vector after pooling layer of CGCNN model
    as features for the structures.
    Please see the CrystalGraphConvNet in the CGCNN repo for more details
    """
    @requires(torch and cgcnn,
              "CrystalGraphConvNetWrapper requires pytorch and cgcnn to be "
              "installed with Python bindings. Please get it at "
              "http://pytorch.org and https://github.com/txie-93/cgcnn.")
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Args:
            orig_atom_fea_len (int): Number of atom features in the input.
            nbr_fea_len (int): Number of bond features.
            atom_fea_len (int): Number of hidden atom features
                in the convolutional layers.
            n_conv (int): Number of convolutional layers.
            h_fea_len (int): Number of hidden features after pooling.
            n_h (int): Number of hidden layers after pooling.
            classification (bool): Classification task or regression task.
        """
        super(CrystalGraphConvNetWrapper, self).__init__(
            orig_atom_fea_len=orig_atom_fea_len, nbr_fea_len=nbr_fea_len,
            atom_fea_len=atom_fea_len, n_conv=n_conv, h_fea_len=h_fea_len,
            n_h=n_h, classification=classification)

    def extract_feature(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Extract the feature vector after pooling layer of CGCNN model as
        features for the structures.

        Args:
            atom_fea (Variable(torch.Tensor)): shape (N, orig_atom_fea_len)
              Atom features from atom type.
            nbr_fea (Variable(torch.Tensor)): shape (N, M, nbr_fea_len)
              Bond features of each atom's M neighbors.
            nbr_fea_idx (torch.LongTensor): shape (N, M)
              Indices of M neighbors of each atom.
            crystal_atom_idx (list of torch.LongTensor):  length N0
              Mapping from the crystal idx to atom idx.

        Returns:
            feature (list): deep learning feature

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        feature = self.pooling(atom_fea, crystal_atom_idx)
        return feature


class AtomCustomArrayInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Args:
        elem_embedding_file (str): The path to the .json file
    """
    @requires(torch and cgcnn,
              "AtomCustomArrayInitializer requires pytorch and cgcnn to be "
              "installed with Python bindings. Please get it at "
              "http://pytorch.org and https://github.com/txie-93/cgcnn.")
    def __init__(self, elem_embedding):
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomArrayInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def appropriate_kwargs(kwargs, func):
    """
    Auto get the appropriate kwargs according to those allowed by the func.
    Args:
        kwargs (dict): kwargs.
        func (object): function object.

    Returns:
        filtered_dict (dict): filtered kwargs.

    """
    sig = inspect.signature(func)
    filter_keys = [param.name for param in sig.parameters.values()
                   if param.kind == param.POSITIONAL_OR_KEYWORD and
                   param.name in kwargs.keys()]
    appropriate_dict = {filter_key: kwargs[filter_key]
                        for filter_key in filter_keys}
    return appropriate_dict
