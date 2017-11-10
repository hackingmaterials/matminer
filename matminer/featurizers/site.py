from __future__ import division

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
import math

from collections import defaultdict

from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.structure_analyzer import OrderParameters
from matminer.featurizers.stats import PropertyStats

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

    def __init__(self, directions=(None, 'x', 'y', 'z'), etas=None,
                 cutoff=8):
        self.directions = directions
        self.etas = etas
        if self.etas is None:
            self.etas = np.logspace(np.log10(0.8), np.log10(16), 8)
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
            windowed[:, i] = np.multiply(np.exp(-1 * np.power(np.true_divide(dists, eta), 2)), cutoff_func)

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
        ddr (float): variation of width for finding stable OP values.
        ndr (int): number of width variations for each variaton direction
                   (e.g., ndr = 0 only uses the input dr, whereas
                   ndr=1 tests dr = dr - ddr, dr, and dr + ddr.
        dop (float): binning width to compute histogram for each OP
                     if ndr > 0.
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
    def __init__(self, optypes=None, dr=0.1, ddr=0.01, ndr=1, dop=0.001,
                 dist_exp=2, zero_ops=True):
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
        self.ddr = ddr
        self.ndr = ndr
        self.dop = dop
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
        idop = 1.0 / self.dop
        opvals = {}
        s = struct.sites[idx]
        neigh_dist = []
        r = 6
        while len(neigh_dist) < 12:
            r += 1.0
            neigh_dist = struct.get_neighbors(s, r)
        # Smoothen distance, but use relative distances.
        dmin = min([d for n, d in neigh_dist])
        neigh_dist = [[n, d / dmin] for n, d in neigh_dist]
        neigh_dist_alldrs = {}
        d_sorted_alldrs = {}
        for i in range(-self.ndr, self.ndr+1):
            opvals[i] = []
            this_dr = self.dr + float(i) * self.ddr
            this_idr = 1.0 / this_dr
            neigh_dist_alldrs[i] = []
            for j in range(len(neigh_dist)):
                neigh_dist_alldrs[i].append([neigh_dist[j][0],
                    (float(int(neigh_dist[j][1] * this_idr \
                    + 0.5)) + 0.5) * this_dr])
            d_sorted_alldrs[i] = []
            for n, d in neigh_dist_alldrs[i]:
                if d not in d_sorted_alldrs[i]:
                    d_sorted_alldrs[i].append(d)
            d_sorted_alldrs[i] = sorted(d_sorted_alldrs[i])

        # Do q_sgl_bd separately.
        if self.optypes[1][0] == "sgl_bd":
            for i in range(-self.ndr, self.ndr+1):
                site_list = [s]
                for n, dn in neigh_dist_alldrs[i]:
                    site_list.append(n)
                opval = self.ops[1][0].get_order_parameters(
                    site_list, 0,
                    indices_neighs=[j for j in range(1,len(site_list))])
                opvals[i].append(opval[0])
    
        for i in range(-self.ndr, self.ndr+1):
            prev_cn = 0
            prev_site_list = None
            prev_d_fac = None
            dmin = min(d_sorted_alldrs[i])
            for d in d_sorted_alldrs[i]:
                this_cn = 0
                site_list = [s]
                this_av_inv_drel = 0.0
                for j, [n, dn] in enumerate(neigh_dist_alldrs[i]):
                    if dn <= d:
                        this_cn += 1
                        site_list.append(n)
                        this_av_inv_drel += (1.0 / (neigh_dist[j][1]))
                this_av_inv_drel = this_av_inv_drel / float(this_cn)
                d_fac = this_av_inv_drel ** self.dist_exp
                for cn in range(max(2, prev_cn+1), min(this_cn+1, 13)):
                    # Set all OPs of non-CN-complying neighbor environments
                    # to zero if applicable.
                    if self.zero_ops and cn != this_cn:
                        for it in range(len(self.optypes[cn])):
                            opvals[i].append(0)
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
                        if self.optypes[cn][it] == 'bcc':
                            opval[0] = opval[0] / 0.976
                        opvals[i].append(opval[0])
                prev_site_list = site_list
                prev_cn = this_cn
                prev_d_fac = d_fac
                if prev_cn >= 12:
                    break

        opvals_out = []
        ps = PropertyStats()
        for j in range(len(opvals[0])):
            # Compute histogram, determine peak, and location
            # of peak value.
            op_tmp = [opvals[i][j] for i in range(-self.ndr, self.ndr+1)]
            minval = float(int(min(op_tmp) * idop - 1.5)) * self.dop
            #print(minval)
            if minval < 0.0:
                minval = 0.0
            if minval > 1.0:
                minval = 1.0
            #print(minval)
            maxval = float(int(max(op_tmp) * idop + 1.5)) * self.dop
            #print(maxval)
            if maxval < 0.0:
                maxval = 0.0
            if maxval > 1.0:
                maxval = 1.0
            #print(maxval)
            if minval == maxval:
                minval = minval - self.dop
                maxval = maxval + self.dop
            #print(minval)
            #print(maxval)
            nbins = int((maxval - minval) * idop)
            #print('{} {} {}'.format(minval, maxval, nbins))
            hist, bin_edges = np.histogram(
                op_tmp, bins=nbins, range=(minval, maxval),
                normed=False, weights=None, density=False)
            max_hist = max(hist)
            op_peaks = []
            for i, h in enumerate(hist):
                if h == max_hist:
                    op_peaks.append([i, 0.5 * (bin_edges[i] + bin_edges[i+1])])
            # Address problem that 2 OP values can be close to a bin edge.
            hist2 = []
            op_peaks2 = []
            i = 0
            while i < len(op_peaks):
                if i < len(op_peaks)-1:
                    if op_peaks[i+1][0] - op_peaks[i][0] == 1:
                        op_peaks2.append(0.5 * (op_peaks[i][1] + op_peaks[i+1][1]))
                        hist2.append(hist[op_peaks[i][0]]+hist[op_peaks[i+1][0]])
                        i += 1
                    else:
                        op_peaks2.append(op_peaks[i][1])
                        hist2.append(hist[op_peaks[i][0]])
                else:
                    op_peaks2.append(op_peaks[i][1])
                    hist2.append(hist[op_peaks[i][0]])
                i += 1
            opvals_out.append(op_peaks2[list(hist2).index(max(hist2))])
        return np.array(opvals_out)

    def feature_labels(self):
        labels = []
        for cn, li in self.optypes.items():
            for e in li:
                labels.append('{} CN_{}'.format(e, cn))
        return labels

    def citations(self):
        return ('@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}')

    def implementors(self):
        return (['Nils E. R. Zimmermann'])



class OPSiteFingerprint_alt(BaseFeaturizer):
    """
    An alternate site fingerprint currently undergoing testing. This code
    will either be improved or deleted depending on how the tests go. For now,
    docs are minimal.
    """

    @staticmethod
    def _semicircle_integral(x, r):
        if r == x:  # avoid tan-1(infinity) braekdown
            return 0.25 * math.pi * r**2

        return 0.5 * ((x * math.sqrt(r**2 - x**2)) + (r**2 * math.atan(x/math.sqrt(r**2 - x**2))))

    def __init__(self, optypes=None, r_max=0.75, tol=1E-3, method="triangle"):
        self.optypes = {
            # 1: ["sgl_bd"],  # TODO: add this back when I figure out how to use this. e.g. mp-4324, mp-3951
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

        self.r_max = r_max
        self.tol = tol
        self.method = method

    def featurize(self, struct, idx):
        c_site = struct[idx]
        neigh_dist = None
        r = 4
        while not neigh_dist:
            neigh_dist = struct.get_neighbors(c_site, r)
            r += 1

        d_min = min([d for n, d in neigh_dist])
        neigh_dist = struct.get_neighbors(c_site, d_min * (1 + self.r_max))

        neigh_dist = [[n, d / d_min] for n, d in neigh_dist]  # normalize
        dist_sort = sorted([d for n, d in neigh_dist])

        tol = d_min * self.tol
        dist_bins = []  # bin numerical tolerances (~error bar of measurement)
        for d in dist_sort:
            if not dist_bins or d > dist_bins[-1] * (1+tol):
                dist_bins.append(d)

        cn_fingerprint_array = defaultdict(list)  # dict where key = CN, val is 2D array that contains a fingerprint vector for each OP in that CN

        if self.method == "triangle":
            total_weight = self.r_max * 0.5
        elif self.method == "circle":
            total_weight = self.r_max**2 * math.pi / 4

        for dist_idx, dist in enumerate(dist_bins):
            neigh_sites = [n for n, d in neigh_dist if d <= dist*(1+tol)]
            cn = len(neigh_sites)
            if cn in self.ops:
                for idx, op in enumerate(self.ops[cn]):
                    # if self.optypes[cn][idx] == "sgl_bd":  # TODO; figure out how to use this
                    #     neigh_sites = [n for n, d in struct.get_neighbors(c_site, 6)]

                    opval = op.get_order_parameters([c_site] + neigh_sites, 0,
                        indices_neighs=[i for i in
                                        range(1, len(neigh_sites)+1)])[0]

                    opval = opval or 0  # handle None

                    if self.optypes[cn][idx] == 'bcc':  # TODO: ask Nils what this is
                        opval = opval / 0.976

                    # compute weight of this CN based on area in weighting triangle
                    if self.method == "triangle":
                        x1 = dist - 1
                        x2 = self.r_max if dist_idx == len(dist_bins) - 1 else \
                            dist_bins[dist_idx + 1] - 1
                        y1 = 1 - (x1 / self.r_max)
                        y2 = 1 - (x2 / self.r_max)
                        weight = 0.5 * (y1 + y2) * (x2 - x1)
                    elif self.method == "circle":
                        x1 = dist - 1
                        x2 = self.r_max if dist_idx == len(dist_bins) - 1 else \
                            dist_bins[dist_idx + 1] - 1
                        weight = self._semicircle_integral(x2,
                                                           self.r_max) - self._semicircle_integral(
                            x1, self.r_max)

                    opval = opval * weight / total_weight

                    cn_fingerprint_array[cn].append(opval)

        # convert dict to list
        cn_fingerprint = []
        for cn in sorted(self.optypes):
            for op_idx, _ in enumerate(self.optypes[cn]):
                try:
                    cn_fingerprint.append(cn_fingerprint_array[cn][op_idx])
                except IndexError:  # no OP value computed
                    cn_fingerprint.append(0)

        return cn_fingerprint

    def feature_labels(self):
        labels = []
        for cn in sorted(self.optypes):
            for op in self.optypes[cn]:
                labels.append("{} CN_{}".format(op, cn))

        return labels

    def citations(self):
        return ['@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}']

    def implementors(self):
        return ['Anubhav Jain', 'Nils E. R. Zimmermann']


# TODO: @nisse3000 this should be made into a Featurizer and more general than 2 classes. Also add unit test afterward, especially since it depends on certain default for OPSiteFingerprint - AJ
def get_tet_bcc_motif(structure, idx):
    """
    Convenience class-method from Nils Zimmermann.
    Used to distinguish coordination environment in half-Heuslers.
    Args:
        structure (pymatgen Structure): the target structure to evaluate
        idx (index): the site index in the structure
    Returns:
        (str) that describes site coordination enviornment
            'bcc'
            'tet'
            'unrecognized'
    """

    op_site_fp = OPSiteFingerprint()
    fp = op_site_fp.featurize(structure, idx)
    labels = op_site_fp.feature_labels()
    i_tet = labels.index('tet CN_4')
    i_bcc = labels.index('bcc CN_8')
    if fp[i_bcc] > 0.5:
        return 'bcc'
    elif fp[i_tet] > 0.5:
        return 'tet'
    else:
        return 'unrecognized'


class CNFingerprint(BaseFeaturizer):
    """
    An alternate site fingerprint currently undergoing testing. This code
    will either be improved or deleted depending on how the tests go. For now,
    docs are minimal.
    """

    def _semicircle_integral(self, x, r):
        if r == x:
            return 0.25 * math.pi * r**2

        return 0.5 * ((x * math.sqrt(r**2 - x**2)) + (r**2 * math.atan(x/math.sqrt(r**2 - x**2))))

    def __init__(self, r_max=0.75, max_cn=12, tol=1E-2, method="circle"):
        self.r_max = r_max
        self.tol = tol
        self.max_cn = max_cn
        self.method = method

    def featurize(self, struct, idx):
        c_site = struct[idx]
        neigh_dist = None

        r = 4
        while not neigh_dist:
            neigh_dist = struct.get_neighbors(c_site, r)
            r += 1

        d_min = min([d for n, d in neigh_dist])
        neigh_dist = struct.get_neighbors(c_site, d_min * (1 + self.r_max))

        neigh_dist = [[n, d / d_min] for n, d in neigh_dist]  # normalize
        dist_sort = sorted([d for n, d in neigh_dist])

        tol = d_min * self.tol
        dist_bins = []  # bin numerical tolerances (~error bar of measurement)
        for d in dist_sort:
            if not dist_bins or d > dist_bins[-1] * (1+tol):
                dist_bins.append(d)

        cn_fingerprint_array = [0 for i in range(self.max_cn)]

        if self.method == "triangle":
            total_weight = self.r_max * 0.5
        elif self.method == "circle":
            total_weight = self.r_max**2 * math.pi / 4

        for dist_idx, dist in enumerate(dist_bins):
            neigh_sites = [n for n, d in neigh_dist if d <= dist*(1+tol)]
            cn = len(neigh_sites)
            if cn <= self.max_cn:
                if self.method == "triangle":
                    x1 = dist - 1
                    x2 = self.r_max if dist_idx == len(dist_bins) - 1 else \
                        dist_bins[dist_idx + 1] - 1
                    y1 = 1 - (x1 / self.r_max)
                    y2 = 1 - (x2 / self.r_max)
                    weight = 0.5 * (y1 + y2) * (x2 - x1)
                elif self.method == "circle":
                    x1 = dist - 1
                    x2 = self.r_max if dist_idx == len(dist_bins) - 1 else \
                        dist_bins[dist_idx + 1] - 1
                    weight = self._semicircle_integral(x2, self.r_max) - self._semicircle_integral(x1, self.r_max)

                cn_fingerprint_array[cn-1] = weight

        cn_fingerprint_array = [x/total_weight for x in cn_fingerprint_array]

        return cn_fingerprint_array

    def feature_labels(self):
        return ["CN_{}".format(i+1) for i in range(self.max_cn)]

    def citations(self):
        return ['']

    def implementors(self):
        return ['Anubhav Jain']
