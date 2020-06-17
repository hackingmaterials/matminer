import functools

import numpy as np
from matminer.utils.data import MatscholarElementData

try:
    import torch
    from torch.utils.data import Dataset
    from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

    from roost.utils import Normalizer
    from roost.roost.model import Roost
except ImportError:
    torch, Dataset = None, object
    L1Loss, MSELoss, CrossEntropyLoss = None, None, None
    Normalizer, Roost = object, object


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(self, comp, targets=None, task="regression", emb="matscholar"):
        """
        """
        self.comp = comp

        if targets is not None:
            self.targets = targets
        else:
            self.targets = np.full_like(comp, np.nan).ravel()

        if emb == "matscholar":
            self.elem_features = MatscholarElementData()
        else:
            raise NotImplementedError(
                "Currently only the matscholar embedding is implemented"
            )

        self.task = task
        if self.task == "regression":
            self.n_targets = 1
        elif self.task == "classification":
            if targets is not None:
                self.n_targets = np.max(self.targets) + 1
            else:
                self.n_targets = 0

    def __len__(self):
        return len(self.comp)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """

        Returns
        -------
        atom_weights: torch.Tensor shape (M, 1)
            weights of atoms in the material
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the material
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indicies
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbour indicies
        target: torch.Tensor shape (1,)
            target value for material
        cry_id: torch.Tensor shape (1,)
            input id for the material
        """
        # NOTE tracking entries can be done at the level of matminer therefore
        # cry_id and composition not needed but must exist for the code to work
        cry_id = None
        composition = None
        elems, weights = zip(*self.comp[idx].element_composition.items())
        target = self.targets[idx]
        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elems) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_elemental_embedding(elem) for elem in elems]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elems)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elems) - 1
        for i, _ in enumerate(elems):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        if self.task == "regression":
            targets = torch.Tensor([float(target)])
        elif self.task == "classification":
            targets = torch.LongTensor([target])

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            composition,
            cry_id,
        )


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for (
        i,
        ((atom_weights, atom_fea, self_fea_idx, nbr_fea_idx), target, comp, cry_id),
    ) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
        ),
        torch.stack(batch_target, dim=0),
        batch_comp,
        batch_cry_ids,
    )


def init_roost(
    model_name,
    run_id,
    task,
    robust,
    loss,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device,
    n_targets,
    elem_emb_len,
    elem_fea_len,
    n_graph,
    elem_heads,
    elem_gate,
    elem_msg,
    cry_heads,
    cry_gate,
    cry_msg,
    out_hidden,
    fine_tune=None,
):

    if fine_tune is not None:
        print(f"Use material_nn and output_nn from '{fine_tune}' as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device)
        model = Roost(**checkpoint["model_params"], device=device,)
        model.load_state_dict(checkpoint["state_dict"])

        if model.model_params["robust"] == True:
            raise NotImplementedError("Robust losses not supported within Matminer")

    else:
        model = Roost(
            n_targets=n_targets,
            elem_emb_len=elem_emb_len,
            elem_fea_len=elem_fea_len,
            n_graph=n_graph,
            elem_heads=elem_heads,
            elem_gate=elem_gate,
            elem_msg=elem_msg,
            cry_heads=cry_heads,
            cry_gate=cry_gate,
            cry_msg=cry_msg,
            out_hidden=out_hidden,
            task=task,
            robust=robust,
            device=device,
        )

        model.to(device)

    # Select Optimiser
    if optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [])

    # Select Task and Loss Function
    if task == "classification":
        normalizer = None
        criterion = CrossEntropyLoss()

    elif task == "regression":
        normalizer = Normalizer()
        if loss == "L1":
            criterion = L1Loss()
        elif loss == "L2":
            criterion = MSELoss()
        else:
            raise NameError("Only L1 or L2 losses are allowed for regression tasks")

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model, criterion, optimizer, scheduler, normalizer
