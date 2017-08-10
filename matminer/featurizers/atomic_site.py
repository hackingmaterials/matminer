"""Features that describe the local environment of a single atom

The `featurize` function takes two arguments:
    strc (Structure): Object representing the structure containing the site of interest
    site (int): Index of the site to be featurized

We have to use two options because the Site object does not hold a pointer back to its structure. To run
:code:`featurize_dataframe`, you must pass the column names for both the site and the structure. For example:

.. code:: python
    f = AGNIFingerprints()
    f.featurize_dataframe(data, ['site', 'structure'])
"""

import numpy as np

from .base import BaseFeaturizer


class AGNIFingerprints(BaseFeaturizer):
    """Integral of the product of the radial distribution function and a Gaussian window function.

    Originally used by [Botu *et al*](http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908) to fit empiricial potentials,
    these features come in two forms: atomic fingerprints, and direction-resolved fingerprints.

    Atomic fingerprints describe the local environment of an atom and are computed using the function:

    :math:`A_i(\eta) = \sum\limits_{i \ne j} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`

    where :math:`i` is the index of the atom, :math:`j` is the index of a neighboring atom, :math:`\eta` is a scaling function,
    :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`, and :math:`f(r)` is a cutoff function where
    :math:`f(r) = 0.5[cos(\frac{\pi r_{ij}}{R_c}) + 1]` if :math:`r < R_c:math:` and 0 otherwise.

    The direction-resolved fingerprints are computed using

    :math:`V_i^k(\eta) = \sum\limits_{i \ne j} \frac{r_{ij}^k}{r_{ij}} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`

    where :math:`r_{ij}^k` is the :math:`k^{th}` component of :math:`\bold{r}_i - \bold{r}_j`.

    Parameters:
            directions (iterable): List of directions for the fingerprints. Can be `none`, 'x', 'y', or 'z'
            etas (iterable of floats): List of which window widths to compute
            cutoff (float): Cutoff distance

    TODO: Differentiate between different atom types (maybe as another class)
    """

    def __init__(self, directions=(None, 'x', 'y', 'z'), etas=np.logspace(np.log10(0.8), np.log10(16), 8),
                 cutoff=8):
        self.directions = directions
        self.etas = etas
        self.cutoff = cutoff

    @property
    def directions(self):
        return self.__directions

    @directions.setter
    def directions(self, dirs):
        for d in dirs:
            if d not in [None, 'x', 'y', 'z']:
                raise Exception('Direction not `None`, x, y, or z: z')
        self.__directions = dirs

    def featurize(self, strc, site):
        # Get all neighbors of this site
        my_site = strc[site]
        sites, dists = zip(*strc.get_neighbors(my_site, self.cutoff))

        # Convert dists to a ndarray
        dists = np.array(dists)

        # If one of the features is direction-dependent, compute the :math:`(r_i - r_j) / r_{ij}`
        if any([x in self.directions for x in ['x', 'y', 'z']]):
            disps = np.array([my_site.coords - s.coords for s in sites]) / dists[:, np.newaxis]

        # Compute the cutoff function
        cutoff_func = 0.5 * (np.cos(np.pi * dists / self.cutoff) + 1)

        # Compute "e^(r/eta) * cutoff_func" for each eta
        windowed = np.zeros((len(dists), len(self.etas)))
        for i, eta in enumerate(self.etas):
            windowed[:, i] = np.multiply(np.exp(-1 * np.power(np.divide(dists, eta), 2)), cutoff_func)

        # Compute the fingerprints
        output = []
        for d in self.directions:
            if d is None:
                output.append(np.sum(windowed, axis=0))
            else:
                if d is 'x':
                    proj = [1., 0., 0.]
                elif d is 'y':
                    proj = [0., 1., 0.]
                elif d is 'z':
                    proj = [0., 0., 1.]
                else:
                    raise Exception('Unrecognized direction')
                output.append(np.sum(windowed * np.dot(disps, proj)[:, np.newaxis], axis=0))

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
        return "@article{Botu2015, author = {Botu, Venkatesh and Ramprasad, Rampi},doi = {10.1002/qua.24836}," \
               "journal = {International Journal of Quantum Chemistry},number = {16},pages = {1074--1083}," \
               "title = {{Adaptive machine learning framework to accelerate ab initio molecular dynamics}}," \
               "volume = {115},year = {2015}}"

    def implementors(self):
        return ['Logan Ward']
