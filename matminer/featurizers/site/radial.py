from __future__ import division

import numpy as np

from matminer.featurizers.utils.grdf import Gaussian, Histogram
from matminer.featurizers.base import BaseFeaturizer


class GeneralizedRadialDistributionFunction(BaseFeaturizer):
    """
    Compute the general radial distribution function (GRDF) for a site.

    The GRDF is a radial measure of crystal order around a site. There are two
    featurizing modes:

    1. GRDF: (recommended) - n_bins length vector
        In GRDF mode, The GRDF is computed by considering all sites around a
        central site (i.e., no sites are omitted when computing the GRDF). The
        features output from this mode will be vectors with length n_bins.

    2. pairwise GRDF: (advanced users) - n_bins x n_sites matrix
        In this mode, GRDFs are are still computed around a central site, but
        only one other site (and their translational equivalents) are used to
        compute a GRDF (e.g. site 1 with site 2 and the translational
        equivalents of site 2). This results in a a n_sites x n_bins matrix of
        features. Requires `fit` for determining the max number of sites for

    The GRDF is a generalization of the partial radial distribution function
    (PRDF). In contrast with the PRDF, the bins of the GRDF are not mutually-
    exclusive and need not carry a constant weight of 1. The PRDF is a case of
    the GRDF when the bins are rectangular functions. Examples of other
    functions to use with the GRDF are Gaussian, trig, and Bessel functions.

    See :func:`~matminer.featurizers.utils.grdf` for a full list of available binning functions.

    There are two preset conditions:
        gaussian: bin functions are gaussians
        histogram: bin functions are rectangular functions

    Args:
        bins:   ([AbstractPairwise]) List of pairwise binning functions. Each of these functions
            must implement the AbstractPairwise class.
        cutoff: (float) maximum distance to look for neighbors
        mode:   (str) the featurizing mode. supported options are:
                    'GRDF' and 'pairwise_GRDF'
    """

    def __init__(self, bins, cutoff=20.0, mode='GRDF'):
        self.bins = bins
        self.cutoff = cutoff

        if mode not in ['GRDF', 'pairwise_GRDF']:
            raise AttributeError('{} is not a valid GRDF mode. try '
                                 '"GRDF" or "pairwise_GRDF"'.format(mode))
        else:
            self.mode = mode

        self.fit_labels = None

    def fit(self, X, y=None, **fit_kwargs):
        """
        Determine the maximum number of sites in X to assign correct feature
        labels

        Args:
            X - [list of tuples], training data
                tuple values should be (struc, idx)
        Returns:
            self
        """

        max_sites = max([len(X[i][0]._sites) for i in range(len(X))])
        self.fit_labels = ['site2 {} {}'.format(i, bin.name()) for bin in self.bins
                           for i in range(max_sites)]
        return self

    def featurize(self, struct, idx):
        """
        Get GRDF of the input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.

        Returns:
            Flattened list of GRDF values. For each run mode the list order is:
                GRDF:          bin#
                pairwise GRDF: site2# bin#
            The site2# corresponds to a pymatgen site index and bin#
            corresponds to one of the bin functions
        """

        if not struct.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get list of neighbors by site
        # Indexing is [site#][neighbor#][pymatgen Site, distance, site index]
        sites = struct._sites
        central_site = sites[idx]
        neighbors_lst = struct.get_neighbors(central_site, self.cutoff,
                                             include_index=True)
        sites = range(0, len(sites))

        # Generate lists of pairwise distances according to run mode
        if self.mode == 'GRDF':
            # Make a single distance collection
            distance_collection = [[neighbor[1] for neighbor in neighbors_lst]]
        else:
            # Make pairwise distance collections for pairwise GRDF
            distance_collection = [
                [neighbor[1] for neighbor in neighbors_lst
                    if neighbor[2] == site_idx] for site_idx in sites]

        # compute bin counts for each list of pairwise distances
        bin_counts = []
        for values in distance_collection:
            bin_counts.append([sum(bin(values)) for bin in self.bins])

        # Compute "volume" of each bin to normalize GRDFs
        volumes = [bin.volume(self.cutoff) for bin in self.bins]

        # normalize the bin counts by the bin volume to compute features
        features = []
        for values in bin_counts:
            features.extend(np.array(values) / np.array(volumes))

        return features

    def feature_labels(self):
        if self.mode == 'GRDF':
            return [bin.name() for bin in self.bins]
        else:
            if self.fit_labels:
                return self.fit_labels
            else:
                raise AttributeError('the fit method must be called first, to '
                                     'determine the correct feature labels.')

    @staticmethod
    def from_preset(preset, width=1.0, spacing=1.0, cutoff=10, mode='GRDF'):
        """
        Preset bin functions for this featurizer. Example use:
            >>> GRDF = GeneralizedRadialDistributionFunction.from_preset('gaussian')
            >>> GRDF.featurize(struct, idx)

        Args:
            preset (str): shape of bin (either 'gaussian' or 'histogram')
            width (float): bin width. std dev for gaussian, width for histogram
            spacing (float): the spacing between bin centers
            cutoff (float): maximum distance to look for neighbors
            mode (str): featurizing mode. either 'GRDF' or 'pairwise_GRDF'
        """

        # Generate bin functions
        if preset == "gaussian":
            bins = []
            for center in np.arange(0., cutoff, spacing):
                bins.append(Gaussian(width, center))
        elif preset == "histogram":
            bins = []
            for start in np.arange(0, cutoff, spacing):
                bins.append(Histogram(start, width))
        else:
            raise ValueError('Not a valid preset condition.')
        return GeneralizedRadialDistributionFunction(bins, cutoff=cutoff, mode=mode)

    def citations(self):
        return ['@article{PhysRevB.95.144110, title = {Representation of compo'
                'unds for machine-learning prediction of physical properties},'
                ' author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, '
                'Keita and Takahashi, Akira and Tanaka, Isao},'
                'journal = {Phys. Rev. B}, volume = {95}, issue = {14}, '
                'pages = {144110}, year = {2017}, publisher = {American Physic'
                'al Society}, doi = {10.1103/PhysRevB.95.144110}}']

    def implementors(self):
        return ["Maxwell Dylla", "Saurabh Bajaj", "Logan Williams"]


class AGNIFingerprints(BaseFeaturizer):
    """
    Product integral of RDF and Gaussian window function, from `Botu et al <http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908>`_.

    Integral of the product of the radial distribution function and a
    Gaussian window function. Originally used by
    `Botu et al <http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908>`_ to fit empiricial
    potentials. These features come in two forms: atomic fingerprints and
    direction-resolved fingerprints.
    Atomic fingerprints describe the local environment of an atom and are
    computed using the function:
    :math:`A_i(\eta) = \sum\limits_{i \\ne j} e^{-(\\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`i` is the index of the atom, :math:`j` is the index of a neighboring atom, :math:`\eta` is a scaling function,
    :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`, and :math:`f(r)` is a cutoff function where
    :math:`f(r) = 0.5[\cos(\\frac{\pi r_{ij}}{R_c}) + 1]` if :math:`r < R_c` and :math:`0` otherwise.
    The direction-resolved fingerprints are computed using
    :math:`V_i^k(\eta) = \sum\limits_{i \\ne j} \\frac{r_{ij}^k}{r_{ij}} e^{-(\\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`r_{ij}^k` is the :math:`k^{th}` component of :math:`\\bold{r}_i - \\bold{r}_j`.
    Parameters:
    TODO: Differentiate between different atom types (maybe as another class)
    """

    def __init__(self, directions=(None, 'x', 'y', 'z'), etas=None,
                 cutoff=8):
        """
        Args:
            directions (iterable): List of directions for the fingerprints. Can
                be one or more of 'None`, 'x', 'y', or 'z'
            etas (iterable of floats): List of which window widths to compute
            cutoff (float): Cutoff distance (Angstroms)
        """
        self.directions = directions
        self.etas = etas
        if self.etas is None:
            self.etas = np.logspace(np.log10(0.8), np.log10(16), 8)
        self.cutoff = cutoff

    def featurize(self, struct, idx):
        # Get all neighbors of this site
        my_site = struct[idx]
        neighbors = struct.get_neighbors(my_site, self.cutoff)
        sites = [n[0] for n in neighbors]
        dists = np.array([n[1] for n in neighbors])

        # If one of the features is direction-dependent, compute the :math:`(r_i - r_j) / r_{ij}`
        if any([x in self.directions for x in ['x', 'y', 'z']]):
            disps = np.array(
                [my_site.coords - s.coords for s in sites]) / dists[:,np.newaxis]

        # Compute the cutoff function
        cutoff_func = 0.5 * (np.cos(np.pi * dists / self.cutoff) + 1)

        # Compute "e^(r/eta) * cutoff_func" for each eta
        windowed = np.zeros((len(dists), len(self.etas)))
        for i, eta in enumerate(self.etas):
            windowed[:, i] = np.multiply(
                np.exp(-1 * np.power(np.true_divide(dists, eta), 2)),
                cutoff_func)

        # Compute the fingerprints
        output = []
        for d in self.directions:
            if d is None:
                output.append(np.sum(windowed, axis=0))
            else:
                if d == 'x':
                    proj = [1., 0., 0.]
                elif d == 'y':
                    proj = [0., 1., 0.]
                elif d == 'z':
                    proj = [0., 0., 1.]
                else:
                    raise Exception('Unrecognized direction')
                output.append(
                    np.sum(windowed * np.dot(disps, proj)[:, np.newaxis],
                           axis=0))

        # Return the results
        return np.hstack(output)

    def feature_labels(self):
        labels = []
        for d in self.directions:
            for e in self.etas:
                if d is None:
                    labels.append('AGNI eta=%.2e' % e)
                else:
                    labels.append('AGNI dir=%s eta=%.2e' % (d, e))
        return labels

    def citations(self):
        return ["@article{Botu2015, author = {Botu, Venkatesh and Ramprasad, Rampi},doi = {10.1002/qua.24836}," \
               "journal = {International Journal of Quantum Chemistry},number = {16},pages = {1074--1083}," \
               "title = {{Adaptive machine learning framework to accelerate ab initio molecular dynamics}}," \
               "volume = {115},year = {2015}}"]

    def implementors(self):
        return ['Logan Ward']

