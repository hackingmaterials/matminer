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
from pymatgen.analysis.structure_analyzer import OrderParameters
from math import sqrt

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


class OPSiteFingerprint(BaseFeaturizer):
    """
    Local structure order parameters computed from the neighbor
    environment of a site. For each order parameter, we determine
    the neighbor shell that complies with the expected
    coordination number. For example, we find the 4 nearest
    neighbors for the tetrahedral OP, the 6 nearest for the
    octahedral OP, and the 8 nearest neighbors for the bcc OP.
    If we don't find such a shell, the OP is either set to zero
    or evaluated with the shell of the next largest observed
    coordination number.

    Args:
        dr (float): width for binning neighors in unit of relative
                    distances (= distance/nearest neighbor
                    distance).  The binning is necessary to make the
                    neighbor-finding step robust agains small numerical
                    variations in neighbor distances (default: 0.1).
        dist_exp (boolean): exponent for distance factor to multiply
                            order parameters with that penalizes (large)
                            variations in distances in a given motif.
                            0 will switch the option off
                            (default: 2).
        zero_ops (boolean): set an OP to zero if there is no neighbor
                            shell that complies with the expected
                            coordination number of a given OP
                            (e.g., CN=4 for tetrahedron;
                            default: True).
    """
    def __init__(self, optypes=None, dr=0.1, dist_exp=2, zero_ops=True):
        self.optypes = {
            1: ["sgl_bd"],
            2: ["bent180", "bent45", "bent90", "bent135"],
            3: ["tri_plan", "tet", "T"],
            4: ["sq_plan", "sq", "tet", "see_saw", "tri_pyr"],
            5: ["pent_plan", "sq_pyr", "tri_bipyr"],
            6: ["oct", "pent_pyr"],
            7: ["hex_pyr", "pent_bipyr"],
            8: ["bcc", "hex_bipyr"],
            9: ["q2", "q4", "q6"],
            10: ["q2", "q4", "q6"],
            11: ["q2", "q4", "q6"],
            12: ["cuboct", "q2", "q4", "q6"]} if optypes is None \
            else optypes.copy()
        self.dr = dr
        self.idr = 1.0 / dr
        self.dist_exp = dist_exp
        self.zero_ops = zero_ops
        self.ops = {}
        for cn, t_list in self.optypes.items():
            self.ops[cn] = []
            for t in t_list:
                if t[:4] == 'bent':
                    self.ops[cn].append(OrderParameters(
                        [t[:4]], parameters=[{'TA': float(t[4:])/180.0, \
                                              'IGW_TA':1.0/0.0667}]))
                else:
                    self.ops[cn].append(OrderParameters([t]))

    def featurize(self, struct, idx):
        """
        Get OP fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            opvals (numpy array): order parameters of target site.
        """
        opvals = []
        s = struct.sites[idx]
        neigh_dist = []
        r = 6
        while len(neigh_dist) < 12:
            r += 1.0
            neigh_dist = struct.get_neighbors(s, r)
        # Smoothen distance, but use relative distances.
        dmin = min([d for n, d in neigh_dist])
        neigh_dist = [[n, d / dmin] for n, d in neigh_dist]
        for j in range(len(neigh_dist)):
            neigh_dist[j][1] = float(
                int(neigh_dist[j][1] * self.idr + 0.5)) * self.dr
        d_sorted = []
        for n, d in neigh_dist:
            if d not in d_sorted:
                d_sorted.append(d)
        d_sorted = sorted(d_sorted)
    
        # Do q_sgl_bd separately.
        if self.optypes[1][0] == "sgl_bd":
            site_list = [s]
            for n, dn in neigh_dist:
                site_list.append(n)
            opval = self.ops[1][0].get_order_parameters(
                site_list, 0,
                indices_neighs=[j for j in range(1,len(site_list))])
            opvals.append(opval[0])
    
        prev_cn = 0
        prev_site_list = None
        prev_d_fac = None
        dmin = min(d_sorted)
        for d in d_sorted:
            this_cn = 0
            site_list = [s]
            this_av_inv_drel = 0.0
            for n, dn in neigh_dist:
                if dn <= d:
                    this_cn += 1
                    site_list.append(n)
                    this_av_inv_drel += (1.0 / (dn / dmin))
            this_av_inv_drel = this_av_inv_drel / float(this_cn)
            d_fac = this_av_inv_drel ** self.dist_exp
            for cn in range(max(2, prev_cn+1), min(this_cn+1, 13)):
                # Set all OPs of non-CN-complying neighbor environments
                # to zero if applicable.
                if self.zero_ops and cn != this_cn:
                    for it in range(len(self.optypes[cn])):
                        opvals.append(0)
                    continue

                # Set all (remaining) OPs.    
                for it in range(len(self.optypes[cn])):
                    opval = self.ops[cn][it].get_order_parameters(
                        site_list, 0,
                        indices_neighs=[j for j in range(1,len(site_list))])
                    if opval[0] is None:
                        opval[0] = 0
                    else:
                        opval[0] = d_fac * opval[0]
                    opvals.append(opval[0])
            prev_site_list = site_list
            prev_cn = this_cn
            prev_d_fac = d_fac
            if prev_cn >= 12:
                break
    
        return np.array(opvals)

    def feature_labels(self):
        labels = []
        for cn, li in self.optypes.items():
            for e in li:
                labels.append(e)
        return labels

    def citations(self):
        return ('@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}')

    def implementors(self):
        return (['Nils E. R. Zimmermann'])

