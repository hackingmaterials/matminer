import inspect
import json
import os
import functools
import warnings
import csv
import numpy as np
import time
import random
import torch
import cgcnn
from cgcnn.data import AtomInitializer, GaussianDistance
from cgcnn.model import CrystalGraphConvNet
from pymatgen.core import Structure
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    """
    Dataset object for torch load data.
    """
    def __init__(self, X, y, atom_init_fea, radius=8, max_num_nbr=12,
                 dmin=0, step=0.2, random_seed=123):
        """
        Args:
            X (Series/list): An iterable of pymatgen Structure objects.
            y (Series/list) : X's target property that CGCNN aims to predict.
            atom_init_fea (dict) : Dict of {atom_id: atom_features}.
            radius (int): Maximum inter-atomic distance.
            max_num_nbr (int): The max number of every atom's neighbors.
            dmin (int): Minimum inter-atomic distance.
            step (float): Step size for the Gaussian filter.
            random_seed (int): random seed.
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.target_data = list(zip(range(len(y)), y))
        random.seed(random_seed)
        random.shuffle(self.target_data)
        self._strcs = X
        self.ari = AtomCustomArrayInitializer(atom_init_fea)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    @property
    def strcs(self):
        return self._strcs

    def __len__(self):
        return len(self.target_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        atom_idx, target = self.target_data[idx]
        crystal = self._strcs[atom_idx]
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
    Create a crystal graph convolutional neural network for predicting total
    material properties.

    Wrapper for CrystalGraphConvNet, add get features ability.
    """
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
        Deep Learning feature extraction.

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


def appropriate_kwargs(kwargs, func):
    """
    Auto get the appropriate kwargs dict due to func parameters
    Args:
        kwargs (dict): Kwargs dict.
        func (object):  function object.

    Returns:
        filtered_dict (dict): Appropriate kwargs dict

    """
    sig = inspect.signature(func)
    filter_keys = [param.name for param in sig.parameters.values()
                   if param.kind == param.POSITIONAL_OR_KEYWORD and
                   param.name in kwargs.keys()]
    appropriate_dict = {filter_key: kwargs[filter_key]
                        for filter_key in filter_keys}
    return appropriate_dict


def train(train_loader, model, criterion, optimizer, epoch, normalizer,
          task='regression', cuda=False, print_freq=10):
    """
    Train a cgcnn model.

    Args:
        train_loader (DataLoader): DataLoader object for training data.
        model: Cgcnn model object.
        criterion: Loss function object.
        optimizer: Optimizer object.
        epoch (int): Epoch number.
        normalizer (Normalizer): Normalizer object.
        task (str): Task type, classification or regression
        cuda (bool): Use cuda or not. If True, use cuda, otherwise, don't use.
        print_freq (int): Result print frequency.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            input_var = (Variable(input[0].cuda(async=True)),
                         Variable(input[1].cuda(async=True)),
                         input[2].cuda(async=True),
                         [crys_idx.cuda(async=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if cuda:
            target_var = Variable(target_normed.cuda(async=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, mae_errors=mae_errors)
                      )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, accu=accuracies,
                       prec=precisions, recall=recalls, f1=fscores,
                       auc=auc_scores))


def validate(val_loader, model, criterion, normalizer, output_path,
             test=False, task='regression', cuda=False, print_freq=10):
    """
    Train a cgcnn model.

    Args:
        val_loader (DataLoader): DataLoader object for validate data.
        model (CrystalGraphConvNetWrapper): Cgcnn model.
        criterion (object): Loss function.
        normalizer (Normalizer): Normalizer object.
        output_path (str): Output path.
        test (bool): Whether predict test data and save it to disk.
        task (str): Task type, classification or regression
        cuda (bool): Use cuda or not. If True, use cuda, otherwise, don't use.
        print_freq (int): Result print frequency.

    Returns:
        avg (float): Return average of mean absolute error if task is
                     regression, otherwise, return average or auc score.
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    if task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if cuda:
            input_var = (Variable(input[0].cuda(async=True), volatile=True),
                         Variable(input[1].cuda(async=True), volatile=True),
                         input[2].cuda(async=True),
                         [crys_idx.cuda(async=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0], volatile=True),
                         Variable(input[1], volatile=True),
                         input[2],
                         input[3])
        if task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if cuda:
            target_var = Variable(target_normed.cuda(async=True),
                                  volatile=True)
        else:
            target_var = Variable(target_normed, volatile=True)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Args:
        prediction(torch.Tensor (N, 1)): Predict tensor.
        target(torch.Tensor (N, 1)): Target tensor.
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    """
    Evaluate deep learning result.

    Args:
        prediction(torch.Tensor (N, 2)): Predict tensor.
        target(torch.Tensor (N, 1)): Target tensor.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        fscore (float): F score.
        auc_score (float): AUC (Area Under the curve) score.
    """
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AtomCustomArrayInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Args:
        elem_embedding_file (str): The path to the .json file
    """
    def __init__(self, elem_embedding):
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomArrayInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

