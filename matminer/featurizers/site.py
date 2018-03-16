from __future__ import division

"""
Features that describe the local environment of a single atom. Note that
structural features can be constructed from a combination of site features from
every site in the structure.

The `featurize` function takes two arguments:
    struct (Structure): Object representing the structure containing the site
        of interest
    idx (int): Index of the site to be featurized
We have to use two parameters because the Site object does not hold a pointer
back to its structure and often information on neighbors is required. To run
:code:`featurize_dataframe`, you must pass the column names for both the site
index and the structure. For example:
.. code:: python
    f = AGNIFingerprints()
    f.featurize_dataframe(data, ['structure', 'site_idx'])
"""

import os
import warnings
import ruamel.yaml as yaml
import itertools
import numpy as np
import math
import scipy.integrate as integrate

from collections import defaultdict

from matminer.featurizers.base import BaseFeaturizer
from math import pi
from scipy.spatial import Voronoi, Delaunay
from pymatgen import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import LocalStructOrderParas, \
    VoronoiNN
import pymatgen.analysis
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder \
    import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies \
   import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from matminer.featurizers.stats import PropertyStats
from sklearn.utils.validation import check_is_fitted

cn_motif_op_params = {}
with open(os.path.join(os.path.dirname(
        pymatgen.analysis.__file__), 'cn_opt_params.yaml'), 'r') as f:
    cn_motif_op_params = yaml.safe_load(f)
cn_target_motif_op = {}
with open(os.path.join(os.path.dirname(
        __file__), 'cn_target_motif_op.yaml'), 'r') as f:
    cn_target_motif_op = yaml.safe_load(f)



class AGNIFingerprints(BaseFeaturizer):
    """Integral of the product of the radial distribution function and a
    Gaussian window function. Originally used by [Botu *et al*]
    (http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908) to fit empiricial
    potentials. These features come in two forms: atomic fingerprints and
    direction-resolved fingerprints.
    Atomic fingerprints describe the local environment of an atom and are
    computed using the function:
    :math:`A_i(\eta) = \sum\limits_{i \ne j} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`i` is the index of the atom, :math:`j` is the index of a neighboring atom, :math:`\eta` is a scaling function,
    :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`, and :math:`f(r)` is a cutoff function where
    :math:`f(r) = 0.5[cos(\frac{\pi r_{ij}}{R_c}) + 1]` if :math:`r < R_c:math:` and 0 otherwise.
    The direction-resolved fingerprints are computed using
    :math:`V_i^k(\eta) = \sum\limits_{i \ne j} \frac{r_{ij}^k}{r_{ij}} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`r_{ij}^k` is the :math:`k^{th}` component of :math:`\bold{r}_i - \bold{r}_j`.
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
        sites, dists = zip(*struct.get_neighbors(my_site, self.cutoff))

        # Convert dists to a ndarray
        dists = np.array(dists)

        # If one of the features is direction-dependent, compute the :math:`(r_i - r_j) / r_{ij}`
        if any([x in self.directions for x in ['x', 'y', 'z']]):
            disps = np.array(
                [my_site.coords - s.coords for s in sites]) / dists[:,
                                                              np.newaxis]

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
        target_motifs (dict): target op or motif type where keys
                              are corresponding coordination numbers
                              (e.g., {4: "tetrahedral"}).
        dr (float): width for binning neighbors in unit of relative
                    distances (= distance/nearest neighbor
                    distance).  The binning is necessary to make the
                    neighbor-finding step robust against small numerical
                    variations in neighbor distances (default: 0.1).
        ddr (float): variation of width for finding stable OP values.
        ndr (int): number of width variations for each variation direction
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

    def __init__(self, target_motifs=None, dr=0.1, ddr=0.01, ndr=1, dop=0.001,
                 dist_exp=2, zero_ops=True):
        self.cn_target_motif_op = cn_target_motif_op.copy() \
            if target_motifs is None else target_motifs.copy()
        self.dr = dr
        self.ddr = ddr
        self.ndr = ndr
        self.dop = dop
        self.dist_exp = dist_exp
        self.zero_ops = zero_ops
        self.ops = {}
        for cn, t_list in self.cn_target_motif_op.items():
            self.ops[cn] = []
            for t in t_list:
                ot = t
                p = None
                if cn in cn_motif_op_params.keys():
                    if t in cn_motif_op_params[cn].keys():
                        ot = cn_motif_op_params[cn][t][0]
                        if len(cn_motif_op_params[cn][t]) > 1:
                            p = cn_motif_op_params[cn][t][1]
                self.ops[cn].append(LocalStructOrderParas([ot], parameters=[p]))

    def featurize(self, struct, idx):
        """
        Get OP fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
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

        for i in range(-self.ndr, self.ndr + 1):
            opvals[i] = []
            this_dr = self.dr + float(i) * self.ddr
            this_idr = 1.0 / this_dr
            neigh_dist_alldrs[i] = []
            for j in range(len(neigh_dist)):
                neigh_dist_alldrs[i].append([neigh_dist[j][0],
                                             (float(
                                                 int(neigh_dist[j][1] * this_idr \
                                                     + 0.5)) + 0.5) * this_dr])
            d_sorted_alldrs[i] = []
            for n, d in neigh_dist_alldrs[i]:
                if d not in d_sorted_alldrs[i]:
                    d_sorted_alldrs[i].append(d)
            d_sorted_alldrs[i] = sorted(d_sorted_alldrs[i])

        # Do q_sgl_bd separately.
        #if self.optypes[1][0] == "sgl_bd":
        if self.cn_target_motif_op[1][0] == "sgl_bd":
            for i in range(-self.ndr, self.ndr + 1):
                site_list = [s]
                for n, dn in neigh_dist_alldrs[i]:
                    site_list.append(n)
                opval = self.ops[1][0].get_order_parameters(
                    site_list, 0,
                    indices_neighs=[j for j in range(1, len(site_list))])
                opvals[i].append(opval[0])

        for i in range(-self.ndr, self.ndr + 1):
            prev_cn = 0
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
                for cn in range(max(2, prev_cn + 1), min(this_cn + 1, 13)):
                    # Set all OPs of non-CN-complying neighbor environments
                    # to zero if applicable.
                    if self.zero_ops and cn != this_cn:
                        for it in range(len(self.cn_target_motif_op[cn])):
                            opvals[i].append(0)
                        continue

                    # Set all (remaining) OPs.
                    for it in range(len(self.cn_target_motif_op[cn])):
                        opval = self.ops[cn][it].get_order_parameters(
                            site_list, 0,
                            indices_neighs=[j for j in
                                            range(1, len(site_list))])
                        if opval[0] is None:
                            opval[0] = 0
                        else:
                            opval[0] = d_fac * opval[0]
                        opvals[i].append(opval[0])
                prev_cn = this_cn
                if prev_cn >= 12:
                    break

        opvals_out = []

        for j in range(len(opvals[0])):
            # Compute histogram, determine peak, and location
            # of peak value.
            op_tmp = [opvals[i][j] for i in range(-self.ndr, self.ndr + 1)]
            minval = float(int(min(op_tmp) * idop - 1.5)) * self.dop
            # print(minval)
            if minval < 0.0:
                minval = 0.0
            if minval > 1.0:
                minval = 1.0
            # print(minval)
            maxval = float(int(max(op_tmp) * idop + 1.5)) * self.dop
            # print(maxval)
            if maxval < 0.0:
                maxval = 0.0
            if maxval > 1.0:
                maxval = 1.0
            # print(maxval)
            if minval == maxval:
                minval = minval - self.dop
                maxval = maxval + self.dop
            # print(minval)
            # print(maxval)
            nbins = int((maxval - minval) * idop)
            # print('{} {} {}'.format(minval, maxval, nbins))
            hist, bin_edges = np.histogram(
                op_tmp, bins=nbins, range=(minval, maxval),
                normed=False, weights=None, density=False)
            max_hist = max(hist)
            op_peaks = []
            for i, h in enumerate(hist):
                if h == max_hist:
                    op_peaks.append(
                        [i, 0.5 * (bin_edges[i] + bin_edges[i + 1])])
            # Address problem that 2 OP values can be close to a bin edge.
            hist2 = []
            op_peaks2 = []
            i = 0
            while i < len(op_peaks):
                if i < len(op_peaks) - 1:
                    if op_peaks[i + 1][0] - op_peaks[i][0] == 1:
                        op_peaks2.append(
                            0.5 * (op_peaks[i][1] + op_peaks[i + 1][1]))
                        hist2.append(
                            hist[op_peaks[i][0]] + hist[op_peaks[i + 1][0]])
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
        for cn, li in self.cn_target_motif_op.items():
            for e in li:
                labels.append('{} CN_{}'.format(e, cn))
        return labels

    def citations(self):
        return ['@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}']

    def implementors(self):
        return ['Nils E. R. Zimmermann']


# TODO: unit tests!!
class CrystalSiteFingerprint(BaseFeaturizer):
    """
    A site fingerprint intended for periodic crystals. The fingerprint represents
    the value of various order parameters for the site; each value is the product
    two quantities: (i) the value of the order parameter itself and (ii) a factor
    that describes how consistent the number of neighbors is with that order
    parameter. Note that we can include only factor (ii) using the "wt" order
    parameter which is always set to 1.
    """

    @staticmethod
    def from_preset(preset, cation_anion=False):
        """
        Use preset parameters to get the fingerprint
        Args:
            preset (str): name of preset ("cn" or "ops")
            cation_anion (bool): whether to only consider cation<->anion bonds
                (bonds with zero charge are also allowed)
        """
        if preset == "cn":
            optypes = dict([(k + 1, ["wt"]) for k in range(16)])
            return CrystalSiteFingerprint(optypes, cation_anion=cation_anion)

        elif preset == "ops":
            optypes = {}
            for cn, li in cn_target_motif_op.items():
                optypes[cn] = li[:]
            optypes[1] = []
            for cn in optypes.keys():
                optypes[cn].insert(0, "wt")
            return CrystalSiteFingerprint(optypes, cation_anion=cation_anion)

        else:
            raise RuntimeError('preset "{}" is not supported in '
                               'CrystalSiteFingerprint'.format(preset))

    def __init__(self, optypes, override_cn1=True, cutoff_radius=8, tol=1E-2,
                 cation_anion=False):
        """
        Initialize the CrystalSiteFingerprint. Use the from_preset() function to
        use default params.
        Args:
            optypes (dict): a dict of coordination number (int) to a list of str
                representing the order parameter types
            override_cn1 (bool): whether to use a special function for the single
                neighbor case. Suggest to keep True.
            cutoff_radius (int): radius in Angstroms for neighbor finding
            tol (float): numerical tolerance (in case your site distances are
                not perfect or to correct for float tolerances)
            cation_anion (bool): whether to only consider cation<->anion bonds
                (bonds with zero charge are also allowed)
        """

        self.optypes = optypes.copy()
        self.override_cn1 = override_cn1
        self.cutoff_radius = cutoff_radius
        self.tol = tol
        self.cation_anion = cation_anion

        if self.override_cn1 and self.optypes.get(1) != ["wt"]:
            raise ValueError(
                "If override_cn1 is True, optypes[1] must be ['wt']!")

        self.ops = {}
        for cn, t_list in self.optypes.items():
            self.ops[cn] = []
            for t in t_list:
                if t == "wt":
                    self.ops[cn].append(t)

                else:
                    ot = t
                    p = None
                    if cn in cn_motif_op_params.keys():
                        if t in cn_motif_op_params[cn].keys():
                            ot = cn_motif_op_params[cn][t][0]
                            if len(cn_motif_op_params[cn][t]) > 1:
                                p = cn_motif_op_params[cn][t][1]
                    self.ops[cn].append(LocalStructOrderParas([ot], parameters=[p]))

    def featurize(self, struct, idx):
        """
        Get crystal fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            list of weighted order parameters of target site.
        """

        cn_fingerprint_array = defaultdict(
            list)  # dict where key = CN, val is array that contains each OP for that CN
        total_weight = math.pi / 4  # 1/4 unit circle area

        target = None
        if self.cation_anion:
            target = []
            m_oxi = struct[idx].specie.oxi_state
            for site in struct:
                if site.specie.oxi_state * m_oxi <= 0:  # opposite charge
                    target.append(site.specie)
            if not target:
                raise ValueError(
                    "No valid targets for site within cation_anion constraint!")

        vnn = VoronoiNN(cutoff=self.cutoff_radius,
                        targets=target)
        n_w = vnn.get_voronoi_polyhedra(struct, idx)

        dist_sorted = (sorted(n_w.values(), reverse=True))

        if self.override_cn1:
            cn1 = 1
            for d in dist_sorted[1:]:
                cn1 = cn1 * (dist_sorted[0] ** 2 - d ** 2) / dist_sorted[0] ** 2
            cn_fingerprint_array[1] = [round(cn1, 6)]
            dist_sorted[0] = dist_sorted[1]

        dist_norm = [d / dist_sorted[0] for d in dist_sorted if d > 0]

        dist_bins = []  # bin numerical tolerances (~error bar of measurement)
        for d in dist_norm:
            if not dist_bins or (
                    d > self.tol and dist_bins[-1] / (1 + self.tol) > d):
                dist_bins.append(d)

        for dist_idx, dist in enumerate(dist_bins):
            neigh_sites = [n for n, w in n_w.items() if
                           w > 0 and w / dist_sorted[0] >= dist / (
                                   1 + self.tol)]
            cn = len(neigh_sites)
            if cn in self.ops:
                for opidx, op in enumerate(self.ops[cn]):
                    if self.optypes[cn][opidx] == "wt":
                        opval = 1
                    else:
                        opval = \
                        op.get_order_parameters([struct[idx]] + neigh_sites, 0,
                                                indices_neighs=[i for i in
                                                                range(1, len(
                                                                    neigh_sites) + 1)])[
                            0]

                    opval = opval or 0  # handles None

                    # figure out the weight for this opval based on semicircle integration method
                    x1 = 1 - dist
                    x2 = 1 if dist_idx == len(dist_bins) - 1 else \
                        1 - dist_bins[dist_idx + 1]
                    weight = self._semicircle_integral(x2) - \
                             self._semicircle_integral(x1)

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
        return []

    def implementors(self):
        return ['Anubhav Jain', 'Nils E.R. Zimmermann']

    @staticmethod
    def _semicircle_integral(x, r=1):
        if r == x:
            return 0.25 * math.pi * r ** 2

        return 0.5 * ((x * math.sqrt(r ** 2 - x ** 2)) + (
                r ** 2 * math.atan(x / math.sqrt(r ** 2 - x ** 2))))


class VoronoiFingerprint(BaseFeaturizer):
    """
    Calculate the following sets of features based on Voronoi tessellation
    analysis around the target site:
    Voronoi indices
        n_i denotes the number of i-edged facets, and i is in the range of 3-10.
        e.g.
        for bcc lattice, the Voronoi indices are [0,6,0,8,...];
        for fcc/hcp lattice, the Voronoi indices are [0,12,0,0,...];
        for icosahedra, the Voronoi indices are [0,0,12,0,...];
    i-fold symmetry indices
        computed as n_i/sum(n_i), and i is in the range of 3-10.
        reflect the strength of i-fold symmetry in local sites.
        e.g.
        for bcc lattice, the i-fold symmetry indices are [0,6/14,0,8/14,...]
            indicating both 4-fold and a stronger 6-fold symmetries are present;
        for fcc/hcp lattice, the i-fold symmetry factors are [0,1,0,0,...],
            indicating only 4-fold symmetry is present;
        for icosahedra, the Voronoi indices are [0,0,1,0,...],
            indicating only 5-fold symmetry is present;
    Weighted i-fold symmetry indices
        if use_weights = True
    Voronoi volume
        total volume of the Voronoi polyhedron around the target site
    Voronoi volume statistics of sub_polyhedra formed by each facet + center
        e.g. stats_vol = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi area
        total area of the Voronoi polyhedron around the target site
    Voronoi area statistics of the facets
        e.g. stats_area = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi nearest-neighboring distance statistics
        e.g. stats_dist = ['mean', 'std_dev', 'minimum', 'maximum']

    Args:
        cutoff (float): cutoff distance in determining the potential
                        neighbors for Voronoi tessellation analysis.
                        (default: 6.5)
        use_weights(bool): whether to use weights to derive weighted
                           i-fold symmetry indices.
        stats_vol (list of str): volume statistics types.
        stats_area (list of str): area statistics types.
        stats_dist (list of str): neighboring distance statistics types.
    """

    def __init__(self, cutoff=6.5, use_weights=False, stats_vol=None,
                 stats_area=None, stats_dist=None):
        self.cutoff = cutoff
        self.use_weights = use_weights
        self.stats_vol = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_vol is None else stats_vol.copy()
        self.stats_area = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_area is None else stats_area.copy()
        self.stats_dist = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_dist is None else stats_dist.copy()

    @staticmethod
    def vol_tetra(vt1, vt2, vt3, vt4):
        """
        Calculate the volume of a tetrahedron, given the four vertices of vt1,
        vt2, vt3 and vt4.
        Args:
            vt1 (array-like): coordinates of vertex 1.
            vt2 (array-like): coordinates of vertex 2.
            vt3 (array-like): coordinates of vertex 3.
            vt4 (array-like): coordinates of vertex 4.
        Returns:
            (float): volume of the tetrahedron.
        """
        vol_tetra = np.abs(np.dot((vt1 - vt4),
                                  np.cross((vt2 - vt4), (vt3 - vt4))))/6
        return vol_tetra

    def featurize(self, struct, idx):
        """
        Get Voronoi fingerprints of site with given index in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Voronoi fingerprints.
                -Voronoi indices
                -i-fold symmetry indices
                -weighted i-fold symmetry indices (if use_weights = True)
                -Voronoi volume
                -Voronoi volume statistics
                -Voronoi area
                -Voronoi area statistics
                -Voronoi area statistics
        """
        n_w = VoronoiNN(cutoff=self.cutoff).get_voronoi_polyhedra(struct, idx)
        voro_idx_list = np.zeros(8, int)
        voro_idx_weights = np.zeros(8)

        vertices = [struct[idx].coords] + [nn.coords for nn in n_w.keys()]
        voro = Voronoi(vertices)

        vol_list = []
        area_list = []
        dist_list = [np.linalg.norm(vertices[0] - vertices[i])
                     for i in range(1, len(vertices))]

        for nn, vind in voro.ridge_dict.items():
            if 0 in nn:
                if -1 in vind:
                    continue
                try:
                    voro_idx_list[len(vind) - 3] += 1
                    if self.use_weights:
                        for neigh, weight in n_w.items():
                            if np.array_equal(neigh.coords,
                                              vertices[sorted(nn)[1]]):
                                voro_idx_weights[len(vind) - 3] += weight
                except IndexError:
                    # If a facet has more than 10 edges, it's skipped here.
                    pass

                polysub = [vertices[0]]
                vol = 0
                for v in vind:
                    polysub.append(list(voro.vertices[v]))
                tetra = Delaunay(np.array(polysub))
                for simplex in tetra.simplices:
                    vol += self.vol_tetra(np.array(polysub[simplex[0]]),
                                          np.array(polysub[simplex[1]]),
                                          np.array(polysub[simplex[2]]),
                                          np.array(polysub[simplex[3]]))
                vol_list.append(vol)
                area_list.append(vol * 6 / dist_list[sorted(nn)[1] - 1])

        symm_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_weights:
            symm_wt_list = voro_idx_weights / sum(voro_idx_weights)
            voro_fps = list(np.concatenate((voro_idx_list, symm_idx_list,
                                           symm_wt_list), axis=0))
        else:
            voro_fps = list(np.concatenate((voro_idx_list,
                                           symm_idx_list), axis=0))

        voro_fps.append(sum(vol_list))
        voro_fps.append(sum(area_list))
        voro_fps += [PropertyStats().calc_stat(vol_list, stat_vol)
                     for stat_vol in self.stats_vol]
        voro_fps += [PropertyStats().calc_stat(area_list, stat_area)
                     for stat_area in self.stats_area]
        voro_fps += [PropertyStats().calc_stat(dist_list, stat_dist)
                     for stat_dist in self.stats_dist]
        return voro_fps

    def feature_labels(self):
        labels = ['Voro_index_%d' % i for i in range(3, 11)]
        labels += ['Symmetry_index_%d' % i for i in range(3, 11)]
        if self.use_weights:
            labels += ['Symmetry_weighted_index_%d' % i for i in range(3, 11)]
        labels.append('Voro_vol_sum')
        labels.append('Voro_area_sum')
        labels += ['Voro_vol_%s' % stat_vol for stat_vol in self.stats_vol]
        labels += ['Voro_area_%s' % stat_area for stat_area in self.stats_area]
        labels += ['Voro_dist_%s' % stat_dist for stat_dist in self.stats_dist]
        return labels

    def citations(self):
        citation = ['@book{okabe1992spatial,  '
                    'title  = {Spatial tessellations}, '
                    'author = {Okabe, Atsuyuki}, '
                    'year   = {1992}, '
                    'publisher = {Wiley Online Library}}']
        return citation

    def implementors(self):
        return ['Qi Wang']


class ChemicalSRO(BaseFeaturizer):
    """
    Chemical short-range ordering (SRO) features to evaluate the deviation
    of local chemistry with the nominal composition of the structure.
    f_el = N_el/(sum of N_el) - c_el,
    where N_el is the number of each element type in the neighbors around
    the target site, sum of N_el is the sum of all possible element types
    (coordination number), and c_el is the composition of the specific
    element in the entire structure.
    A positive f_el indicates the "bonding" with the specific element
    is favored, at least in the target site;
    A negative f_el indicates the "bonding" is not favored, at least
    in the target site.

    Note that ChemicalSRO is only featurized for elements identified by
    "fit" (see following), thus "fit" must be called before "featurize",
    or else an error will be raised.
    Args:
        nn (NearestNeighbor): instance of one of pymatgen's Nearest Neighbor
                              classes.
        includes (array-like or str): elements included to calculate CSRO.
        excludes (array-like or str): elements excluded to calculate CSRO.
        sort (bool): whether to sort elements by mendeleev number.
    """

    def __init__(self, nn, includes=None, excludes=None, sort=True):
        self.nn = nn
        self.includes = includes
        if self.includes:
            self.includes = [Element(el).symbol
                             for el in np.atleast_1d(self.includes)]
        self.excludes = excludes
        if self.excludes:
            self.excludes = [Element(el).symbol
                             for el in np.atleast_1d(self.excludes)]
        self.sort = sort
        self.el_list_ = None
        self.el_amt_dict_ = None

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use one of the standard instances of a given NearNeighbor class.
        Args:
            preset (str): preset type ("VoronoiNN", "JMolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
            **kwargs: allow to pass args to the NearNeighbor class.
        Returns:
            ChemicalSRO from a preset.
        """
        nn_ = getattr(pymatgen.analysis.local_env, preset)
        return ChemicalSRO(nn_(**kwargs))

    def fit(self, X, y=None):
        """
        Identify elements to be included in the following featurization,
        by intersecting the elements present in the passed structures with
        those explicitly included (or excluded) in __init__. Only elements
        in the self.el_list_ will be featurized.
        Besides, compositions of the passed structures will also be "stored"
        in a dict of self.el_amt_dict_, avoiding repeated calculation of
        composition when featurizing multiple sites in the same structure.
        Args:
            X (array-like): containing Pymatgen structures and sites, supports
                            multiple choices:
                            -2D array-like object:
                             e.g. [[struct, site], [struct, site], …]
                                  np.array([[struct, site], [struct, site], …])
                            -Pandas dataframe:
                             e.g. df[['struct', 'site']]
            y : unused (added for consistency with overridden method signature)
        Returns:
            self
        """
        structs = np.atleast_2d(X)[:, 0]
        if not all([isinstance(struct, Structure) for struct in structs]):
            raise TypeError("This fit requires an array-like input of Pymatgen "
                            "Structures and sites!")

        self.el_amt_dict_ = {}
        el_set_ = set()
        for s in structs:
            if str(s) not in self.el_amt_dict_.keys():
                el_amt_ = s.composition.fractional_composition.get_el_amt_dict()
                els_ = set(el_amt_.keys()) if self.includes is None \
                    else set([el for el in el_amt_.keys()
                              if el in self.includes])
                els_ = els_ if self.excludes is None \
                    else els_ - set(self.excludes)
                if els_:
                    self.el_amt_dict_[str(s)] = el_amt_
                el_set_ = el_set_ | els_
        self.el_list_ = sorted(list(el_set_), key=lambda el:
                Element(el).mendeleev_no) if self.sort else list(el_set_)
        return self

    def featurize(self, struct, idx):
        """
        Get CSRO features of site with given index in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Chemical SRO features for each element.
        """

        check_is_fitted(self, ['el_amt_dict_', 'el_list_'])

        csro = [0.]*len(self.el_list_)
        if str(struct) in self.el_amt_dict_.keys():
            el_amt = self.el_amt_dict_[str(struct)]
            nn_el_amt = dict.fromkeys(el_amt, 0)
            nn_list = self.nn.get_nn(struct, idx)
            for nn in nn_list:
                if str(nn.specie.symbol) in self.el_list_:
                    nn_el_amt[str(nn.specie.symbol)] += 1/len(nn_list)
            for el in el_amt.keys():
                if el in self.el_list_:
                    csro[self.el_list_.index(el)] = nn_el_amt[el] - el_amt[el]
        return csro

    def feature_labels(self):
        check_is_fitted(self, ['el_amt_dict_', 'el_list_'])

        return ['CSRO_{}_{}'.format(el, self.nn.__class__.__name__)
                for el in self.el_list_]

    def citations(self):
        citations = []
        if self.nn.__class__.__name__ == 'VoronoiNN':
            citations.append('@article{voronoi_jreineangewmath_1908, title={'
                'Nouvelles applications des param\\`{e}tres continus \\`{a} la '
                'th\'{e}orie des formes quadratiques. Sur quelques '
                'propri\'{e}t\'{e}s des formes quadratiques positives'
                ' parfaites}, journal={Journal f\"ur die reine und angewandte '
                'Mathematik}, number={133}, pages={97-178}, year={1908}}')
            citations.append('@article{dirichlet_jreineangewmath_1850, title={'
                '\"{U}ber die Reduction der positiven quadratischen Formen '
                'mit drei unbestimmten ganzen Zahlen}, journal={Journal '
                'f\"ur die reine und angewandte Mathematik}, number={40}, '
                'pages={209-227}, doi={10.1515/crll.1850.40.209}, year={1850}}')
        if self.nn.__class__.__name__ == 'JMolNN':
            citations.append('@misc{jmol, title = {Jmol: an open-source Java '
                'viewer for chemical structures in 3D}, howpublished = {'
                '\\url{http://www.jmol.org/}}}')
        if self.nn.__class__.__name__ == 'MinimumOKeeffeNN':
            citations.append('@article{okeeffe_jamchemsoc_1991, title={Atom '
                'sizes and bond lengths in molecules and crystals}, journal='
                '{Journal of the American Chemical Society}, author={'
                'O\'Keeffe, M. and Brese, N. E.}, number={113}, pages={'
                '3226-3229}, doi={doi:10.1021/ja00009a002}, year={1991}}')
        if self.nn.__class__.__name__ == 'MinimumVIRENN':
            citations.append('@article{shannon_actacryst_1976, title={'
                'Revised effective ionic radii and systematic studies of '
                'interatomic distances in halides and chalcogenides}, '
                'journal={Acta Crystallographica}, author={Shannon, R. D.}, '
                'number={A32}, pages={751-767}, doi={'
                '10.1107/S0567739476001551}, year={1976}')
        if self.nn.__class__.__name__ in [
                'MinimumDistanceNN', 'MinimumOKeeffeNN', 'MinimumVIRENN']:
            citations.append('@article{zimmermann_frontmater_2017, '
                'title={Assessing local structure motifs using order '
                'parameters for motif recognition, interstitial '
                'identification, and diffusion path characterization}, '
                'journal={Frontiers in Materials}, author={Zimmermann, '
                'N. E. R. and Horton, M. K. and Jain, A. and Haranczyk, M.}, '
                'number={4:34}, doi={10.3389/fmats.2017.00034}, year={2017}}')
        return citations

    def implementors(self):
        return ['Qi Wang']


class GaussianSymmFunc(BaseFeaturizer):
    """
    Gaussian symmetry function features suggested by Behler et al.,
    based on pair distances and angles, to approximate the functional
    dependence of local energies, originally used in the fitting of
    machine-learning potentials.
    The symmetry functions can be divided to a set of radial functions
    (g2 function), and a set of angular functions (g4 function).
    The number of symmetry functions returned are based on parameters
    of etas_g2, etas_g4, zetas_g4 and gammas_g4.
    See the original papers for more details:
    “Atom-centered symmetry functions for constructing high-dimensional
    neural network potentials”, J Behler, J Chem Phys 134, 074106 (2011).
    The cutoff function is taken as the polynomial form (cosine_cutoff)
    to give a smoothed truncation.
    A Fortran and a different Python version can be found in the code
    Amp: Atomistic Machine-learning Package
    (https://bitbucket.org/andrewpeterson/amp).
    Args:
        etas_g2 (list of floats): etas used in radial functions.
                                  (default: [0.05, 4., 20., 80.])
        etas_g4 (list of floats): etas used in angular functions.
                                  (default: [0.005])
        zetas_g4 (list of floats): zetas used in angular functions.
                                   (default: [1., 4.])
        gammas_g4 (list of floats): gammas used in angular functions.
                                    (default: [+1., -1.])
        cutoff (float): cutoff distance. (default: 6.5)
    """

    def __init__(self, etas_g2=None, etas_g4=None, zetas_g4=None,
                 gammas_g4=None, cutoff=6.5):
        self.etas_g2 = etas_g2 if etas_g2 else [0.05, 4., 20., 80.]
        self.etas_g4 = etas_g4 if etas_g4 else [0.005]
        self.zetas_g4 = zetas_g4 if zetas_g4 else [1., 4.]
        self.gammas_g4 = gammas_g4 if gammas_g4 else [+1., -1.]
        self.cutoff = cutoff

    @staticmethod
    def cosine_cutoff(r, cutoff):
        """
        Polynomial cutoff function to give a smoothed truncation of the Gaussian
        symmetry functions.
        Args:
            r (float): distance.
            cutoff (float): cutoff distance.
        Returns:
            (float) cutoff function.
        """
        return 0 if r > cutoff else 0.5 * (np.cos(np.pi * r / cutoff) + 1.)

    @staticmethod
    def g2(eta, center_coord, neigh_coords, cutoff):
        """
        Gaussian radial symmetry function of the center atom,
        given an eta parameter.
        Args:
            eta: radial function parameter.
            center_coord (list of floats): coordinates of center atom.
            neigh_coords (list of [floats]): coordinates of neighboring atoms.
            cutoff (float): cutoff distance.
        Returns:
            (float) Gaussian radial symmetry function.
        """
        ridge = 0.
        for neigh_coord in neigh_coords:
            if np.array_equal(neigh_coord, center_coord):
                continue
            r = np.linalg.norm(neigh_coord - center_coord)
            ridge += (np.exp(-eta * (r ** 2.) / (cutoff ** 2.)) *
                      GaussianSymmFunc.cosine_cutoff(r, cutoff))
        return ridge

    @staticmethod
    def g4(eta, zeta, gamma, center_coord, neigh_coords, cutoff):
        """
        Gaussian angular symmetry function of the center atom,
        given a set of eta, zeta and gamma parameters.
        Args:
            eta (float): angular function parameter.
            zeta (float): angular function parameter.
            gamma (float): angular function parameter.
            center_coord (list of floats): coordinates of center atom.
            neigh_coords (list of [floats]): coordinates of neighboring atoms.
            cutoff (float): cutoff parameter.
        Returns:
            (float) Gaussian angular symmetry function.
        """
        ridge = 0.
        for j, neigh_j in enumerate(neigh_coords):
            for neigh_k in neigh_coords[j+1:]:
                if np.array_equal(neigh_j, center_coord) or \
                   np.array_equal(neigh_k, center_coord):
                    continue
                r_ij = np.linalg.norm(neigh_j - center_coord)
                r_ik = np.linalg.norm(neigh_k - center_coord)
                r_jk = np.linalg.norm(neigh_k - neigh_j)
                cos_theta = np.dot((neigh_j - center_coord),
                                   (neigh_k - center_coord)) / r_ij / r_ik
                term = (1. + gamma * cos_theta) ** zeta * \
                       np.exp(-eta * (r_ij ** 2. + r_ik ** 2. + r_jk ** 2.) /
                              (cutoff ** 2.)) * \
                       GaussianSymmFunc.cosine_cutoff(r_ij, cutoff) * \
                       GaussianSymmFunc.cosine_cutoff(r_ik, cutoff) * \
                       GaussianSymmFunc.cosine_cutoff(r_jk, cutoff)
                ridge += term
        ridge *= 2. ** (1. - zeta)
        return ridge

    def featurize(self, struct, idx):
        """
        Get Gaussian symmetry function features of site with given index
        in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Gaussian symmetry function features.
        """
        gaussian_funcs = []
        neighbors = struct.get_sites_in_sphere(
            struct[idx].coords, self.cutoff)
        neigh_coords = [neigh[0].coords for neigh in neighbors]
        for eta_g2 in self.etas_g2:
            gaussian_funcs.append(self.g2(eta_g2,
                                          struct[idx].coords,
                                          neigh_coords,
                                          self.cutoff))

        for eta_g4 in self.etas_g4:
            for zeta_g4 in self.zetas_g4:
                for gamma_g4 in self.gammas_g4:
                    gaussian_funcs.append(self.g4(eta_g4, zeta_g4, gamma_g4,
                                                  struct[idx].coords,
                                                  neigh_coords,
                                                  self.cutoff))
        return gaussian_funcs

    def feature_labels(self):
        return ['G2_{}'.format(eta_g2) for eta_g2 in self.etas_g2] + \
               ['G4_{}_{}_{}'.format(eta_g4, zeta_g4, gamma_g4)
                for eta_g4 in self.etas_g4
                for zeta_g4 in self.zetas_g4
                for gamma_g4 in self.gammas_g4]

    def citations(self):
        gsf_citation = (
            '@Article{Behler2011, author = {Jörg Behler}, '
            'title = {Atom-centered symmetry functions for constructing '
            'high-dimensional neural network potentials}, '
            'journal = {The Journal of Chemical Physics}, year = {2011}, '
            'volume = {134}, number = {7}, pages = {074106}, '
            'doi = {10.1063/1.3553717}}')
        amp_citation = (
            '@Article{Khorshidi2016, '
            'author = {Alireza Khorshidi and Andrew A. Peterson}, '
            'title = {Amp : A modular approach to machine learning in '
            'atomistic simulations}, '
            'journal = {Computer Physics Communications}, year = {2016}, '
            'volume = {207}, pages = {310--324}, '
            'doi = {10.1016/j.cpc.2016.05.010}}')
        return [gsf_citation, amp_citation]

    def implementors(self):
        return ['Qi Wang']


class EwaldSiteEnergy(BaseFeaturizer):
    """Compute site energy from Coulombic interactions
    User notes:
        - This class uses that `charges that are already-defined for the structure`.
        - Ewald summations can be expensive. If you evaluating every site in many
          large structures, run all of the sites for each structure at the same time.
          We cache the Ewald result for the structure that was run last, so looping
          over sites and then structures is faster than structures than sites.
    Features:
        ewald_site_energy - Energy for the site computed from Coulombic interactions"""

    def __init__(self, accuracy=None):
        """
        Args:
            accuracy (int): Accuracy of Ewald summation, number of decimal places
        """
        self.accuracy = accuracy

        # Variables used then caching the Ewald result
        self.__last_structure = None
        self.__last_ewald = None

    def featurize(self, strc, idx):
        """
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            ([float]) - Electrostatic energy of the site
        """

        # Check if the new input is the last
        #  Note: We use 'is' rather than structure comparisons for speed
        if strc is self.__last_structure:
            ewald = self.__last_ewald
        else:
            self.__last_structure = strc
            ewald = EwaldSummation(strc, acc_factor=self.accuracy)
            self.__last_ewald = ewald
        return [ewald.get_site_energy(idx)]

    def feature_labels(self):
        return ["ewald_site_energy"]

    def implementors(self):
        return ["Logan Ward"]

    def citations(self):
        return ["@Article{Ewald1921,"
                "author = {Ewald, P. P.},"
                "doi = {10.1002/andp.19213690304},"
                "issn = {00033804},"
                "journal = {Annalen der Physik},"
                "number = {3},"
                "pages = {253--287},"
                "title = {{Die Berechnung optischer und elektrostatischer Gitterpotentiale}},"
                "url = {http://doi.wiley.com/10.1002/andp.19213690304},"
                "volume = {369},"
                "year = {1921}"
                "}"]


class ChemEnvSiteFingerprint(BaseFeaturizer):
    """
    Site fingerprint computed from pymatgen's ChemEnv package
    that provides resemblance percentages of a given site
    to ideal environments.
    Args:
        cetypes ([str]): chemical environments (CEs) to be
            considered.
        strategy (ChemenvStrategy): ChemEnv neighbor-finding strategy.
        geom_finder (LocalGeometryFinder): ChemEnv local geometry finder.
        max_csm (float): maximum continuous symmetry measure (CSM;
            default of 8 taken from chemenv). Note that any CSM
            larger than max_csm will be set to max_csm in order
            to avoid negative values (i.e., all features are
            constrained to be between 0 and 1).
        max_dist_fac (float): maximum distance factor (default: 1.41).
    """

    @staticmethod
    def from_preset(preset):
        """
        Use a standard collection of CE types and
        choose your ChemEnv neighbor-finding strategy.
        Args:
            preset (str): preset types ("simple" or
                          "multi_weights").
        Returns:
            ChemEnvSiteFingerprint object from a preset.
        """
        cetypes = [
            'S:1', 'L:2', 'A:2', 'TL:3', 'TY:3', 'TS:3', 'T:4',
            'S:4', 'SY:4', 'SS:4', 'PP:5', 'S:5', 'T:5', 'O:6',
            'T:6', 'PP:6', 'PB:7', 'ST:7', 'ET:7', 'FO:7', 'C:8',
            'SA:8', 'SBT:8', 'TBT:8', 'DD:8', 'DDPN:8', 'HB:8',
            'BO_1:8', 'BO_2:8', 'BO_3:8', 'TC:9', 'TT_1:9',
            'TT_2:9', 'TT_3:9', 'HD:9', 'TI:9', 'SMA:9', 'SS:9',
            'TO_1:9', 'TO_2:9', 'TO_3:9', 'PP:10', 'PA:10',
            'SBSA:10', 'MI:10', 'S:10', 'H:10', 'BS_1:10',
            'BS_2:10', 'TBSA:10', 'PCPA:11', 'H:11', 'SH:11',
            'CO:11', 'DI:11', 'I:12', 'PBP:12', 'TT:12', 'C:12',
            'AC:12', 'SC:12', 'S:12', 'HP:12', 'HA:12', 'SH:13',
            'DD:20']
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(
            centering_type='centroid',
            include_central_site_in_centroid=True,
            structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)
        if preset == "simple":
            return ChemEnvSiteFingerprint(
                cetypes,
                SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3),
                lgf)
        elif preset == "multi_weights":
            return ChemEnvSiteFingerprint(
                cetypes,
                MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
                lgf)
        else:
            raise RuntimeError('unknown neighbor-finding strategy preset.')

    def __init__(self, cetypes, strategy, geom_finder, max_csm=8, \
            max_dist_fac=1.41):
        self.cetypes = tuple(cetypes)
        self.strat = strategy
        self.lgf = geom_finder
        self.max_csm = max_csm
        self.max_dist_fac = max_dist_fac

    def featurize(self, struct, idx):
        """
        Get ChemEnv fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            (numpy array): resemblance fraction of target site to ideal
                           local environments.
        """
        cevals = []
        self.lgf.setup_structure(structure=struct)
        se = self.lgf.compute_structure_environments(
                maximum_distance_factor=self.max_dist_fac)
        for ce in self.cetypes:
            try:
                tmp = se.get_csms(idx, ce)
                tmp = tmp[0]['symmetry_measure'] if len(tmp) != 0 \
                    else self.max_csm
                tmp = tmp if tmp < self.max_csm else self.max_csm
                cevals.append(1 - tmp / self.max_csm)
            except IndexError:
                cevals.append(0)
        return np.array(cevals)

    def feature_labels(self):
        return list(self.cetypes)

    def citations(self):
        return ['@article{waroquiers_chemmater_2017, '
                'title={Statistical analysis of coordination environments '
                'in oxides}, journal={Chemistry of Materials},'
                'author={Waroquiers, D. and Gonze, X. and Rignanese, G.-M.'
                'and Welker-Nieuwoudt, C. and Rosowski, F. and Goebel, M. '
                'and Schenk, S. and Degelmann, P. and Andre, R. '
                'and Glaum, R. and Hautier, G.}, year={2017}}']

    def implementors(self):
        return ['Nils E. R. Zimmermann']

class CoordinationNumber(BaseFeaturizer):
    """
    Coordination number (CN) computed using one of pymatgen's
    NearNeighbor classes for determination of near neighbors
    contributing to the CN.
    Args:
        nn (NearNeighbor): instance of one of pymatgen's NearNeighbor
                           classes.
    """

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use one of the standard instances of a given NearNeighbor
        class.
        Args:
            preset (str): preset type ("VoronoiNN", "JMolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
            **kwargs: allow to pass args to the NearNeighbor class.
        Returns:
            CoordinationNumber from a preset.
        """
        nn_ = getattr(pymatgen.analysis.local_env, preset)
        return CoordinationNumber(nn_(**kwargs))

    def __init__(self, nn, use_weights=False):
        self.nn = nn
        self.use_weights = use_weights

    def featurize(self, struct, idx):
        """
        Get coordintion number of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            (float): coordination number.
        """
        return [self.nn.get_cn(struct, idx, use_weights=self.use_weights)]

    def feature_labels(self):
        return ['CN_{}'.format(self.nn.__class__.__name__)]

    def citations(self):
        citations = []
        if self.nn.__class__.__name__ == 'VoronoiNN':
            citations.append('@article{voronoi_jreineangewmath_1908, title={'
                'Nouvelles applications des param\\`{e}tres continus \\`{a} la '
                'th\'{e}orie des formes quadratiques. Sur quelques '
                'propri\'{e}t\'{e}s des formes quadratiques positives'
                ' parfaites}, journal={Journal f\"ur die reine und angewandte '
                'Mathematik}, number={133}, pages={97-178}, year={1908}}')
            citations.append('@article{dirichlet_jreineangewmath_1850, title={'
                '\"{U}ber die Reduction der positiven quadratischen Formen '
                'mit drei unbestimmten ganzen Zahlen}, journal={Journal '
                'f\"ur die reine und angewandte Mathematik}, number={40}, '
                'pages={209-227}, doi={10.1515/crll.1850.40.209}, year={1850}}')
        if self.nn.__class__.__name__ == 'JMolNN':
            citations.append('@misc{jmol, title = {Jmol: an open-source Java '
                'viewer for chemical structures in 3D}, howpublished = {'
                '\\url{http://www.jmol.org/}}}')
        if self.nn.__class__.__name__ == 'MinimumOKeeffeNN':
            citations.append('@article{okeeffe_jamchemsoc_1991, title={Atom '
                'sizes and bond lengths in molecules and crystals}, journal='
                '{Journal of the American Chemical Society}, author={'
                'O\'Keeffe, M. and Brese, N. E.}, number={113}, pages={'
                '3226-3229}, doi={doi:10.1021/ja00009a002}, year={1991}}')
        if self.nn.__class__.__name__ == 'MinimumVIRENN':
            citations.append('@article{shannon_actacryst_1976, title={'
                'Revised effective ionic radii and systematic studies of '
                'interatomic distances in halides and chalcogenides}, '
                'journal={Acta Crystallographica}, author={Shannon, R. D.}, '
                'number={A32}, pages={751-767}, doi={'
                '10.1107/S0567739476001551}, year={1976}')
        if self.nn.__class__.__name__ in [
                'MinimumDistanceNN', 'MinimumOKeeffeNN', 'MinimumVIRENN']:
            citations.append('@article{zimmermann_frontmater_2017, '
                'title={Assessing local structure motifs using order '
                'parameters for motif recognition, interstitial '
                'identification, and diffusion path characterization}, '
                'journal={Frontiers in Materials}, author={Zimmermann, '
                'N. E. R. and Horton, M. K. and Jain, A. and Haranczyk, M.}, '
                'number={4:34}, doi={10.3389/fmats.2017.00034}, year={2017}}')
        return citations

    def implementors(self):
        return ['Nils E. R. Zimmermann']


class GeneralizedRadialDistributionFunction(BaseFeaturizer):
    """
    Compute the general radial distribution function (GRDF) for a site. The
    GRDF is a radial measure of crystal order around a site. There are two
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

    There are two preset conditions:
        gaussian: bin functionals are gaussians
        histogram: bin functionals are rectangular functions

    Args:
        bins:   (list of tuples) a list of (str, functions). The str is a text
                    label for each bin functional. The functions should accept
                    scalar numpy arrays (each scalar value corresponds to a
                    distance) and return arrays of floats.
                    (e.g. lambda d: exp( a_0 * (d - b_0)**2 ))
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

        if self.n_jobs != 1:
            warnings.warn("This featurizer does not support n_jobs > 1.")
            self.set_n_jobs(1)

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
        self.fit_labels = ['site2 {} {}'.format(i, bin[0]) for bin in self.bins
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
            corresponds to one of the bin functionals
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
            distance_collection = [
                [neighbor[1] for neighbor in neighbors_lst]]
        else:
            # Make pairwise distance collections for pairwise GRDF
            distance_collection = [
                [neighbor[1] for neighbor in neighbors_lst
                    if neighbor[2] == site_idx] for site_idx in sites]

        # compute bin counts for each list of pairwise distances
        bin_counts = []
        for values in distance_collection:
            values = np.array(values)  # use scalar array function calling
            bin_counts.append([sum(bin[1](values)) for bin in self.bins])

        # Compute "volume" of each bin to normalize GRDFs
        integrations = [integrate.quad(lambda x: 4. * pi * bin[1](x) * x**2.,
                                       0, self.cutoff) for bin in self.bins]

        volumes = [item[0] for item in integrations]
        errors = [item[1] for item in integrations]
        if max(errors) > 1e-5:
            raise ValueError('Numerical integration does not play well with '
                             'the chosen bin functionals, choose new ones.')

        # normalize the bin counts by the bin volume to compute features
        features = []
        for values in bin_counts:
            features.extend(np.array(values) / np.array(volumes))

        return features

    def feature_labels(self):
        if self.mode == 'GRDF':
            return [bin[0] for bin in self.bins]
        else:
            if self.fit_labels:
                return self.fit_labels
            else:
                raise AttributeError('the fit method must be called first, to '
                                     'determine the correct feature labels.')

    @staticmethod
    def from_preset(preset, width=0.5, spacing=0.5, cutoff=10, mode='GRDF'):
        '''
        Preset bin functionals for this featurizer. Example use:
            >>> GRDF = GeneralizedRadialDistributionFunction.from_preset('gaussian')
            >>> GRDF.featurize(struct, idx)

        Args:
            preset (str): shape of bin (either 'gaussian' or 'histogram')
            width (float): bin width. std dev for gaussian, width for histogram
            spacing (float): the spacing between bin centers
            cutoff (float): maximum distance to look for neighbors
            mode (str): featurizing mode. either 'GRDF' or 'pairwise_GRDF'
        '''

        if preset == "gaussian":
            bins = []
            for center in np.arange(0., cutoff, spacing):
                bins.append(('Gauss {}'.format(center),
                             lambda d: np.exp(-width * (d - center)**2.)))
            return GeneralizedRadialDistributionFunction(bins, cutoff=cutoff,
                                                         mode=mode)

        elif preset == "histogram":
            bins = []
            for center in np.arange(0. + (width / 2.), cutoff, spacing):
                bins.append(('Hist {}'.format(center),
                             lambda d: np.where(center - (width / 2.) <= d,
                                                1., 0.) *
                             np.where(d < center + (width / 2.), 1., 0.)))
            return GeneralizedRadialDistributionFunction(bins, cutoff=cutoff,
                                                         mode=mode)

        else:
            raise ValueError('Not a valid preset condition.')

    def citations(self):
        return ['@article{PhysRevB.95.144110, title = {Representation of compo'
                'unds for machine-learning prediction of physical properties},'
                ' author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, '
                'Keita and Takahashi, Akira and Tanaka, Isao},'
                'journal = {Phys. Rev. B}, volume = {95}, issue = {14}, '
                'pages = {144110}, year = {2017}, publisher = {American Physic'
                'al Society}, doi = {10.1103/PhysRevB.95.144110}}']

    def implementors(self):
        return ["Maxwell Dylla", "Saurabh Bajaj"]


class AngularFourierSeries(BaseFeaturizer):
    """
    Compute the angular Fourier series (AFS) for a site. The AFS includes
    both radial and angular information about site neighbors. The AFS is the
    product of distance functionals (g_n, g_n') between two pairs of atoms
    (sharing the common central site) and the cosine of the angle between the
    two pairs. The AFS is a 2-dimensional feature (the axes are g_n, g_n').

    Examples of distance functionals are square functions, Gaussian, trig
    functions, and Bessel functions. An example for Gaussian:
        lambda d: exp( -(d - d_n)**2 ), where d_n is the coefficient for g_n

    There are two preset conditions:
        gaussian: bin functionals are gaussians
        histogram: bin functionals are rectangular functions

    Args:
        bins:   (list of tuples) a list of (str, functions). The str is a text
                 label for each bin functional. The functions should accept
                 scalar numpy arrays (each scalar value corresponds to a
                 distance) and return arrays of floats.
                  (e.g. lambda d: exp( - a_0 * (d - b_0)**2 ))
        cutoff: (float) maximum distance to look for neighbors. The
                 featurizer will run slowly for large distance cutoffs
                 because of the number of neighbor pairs scales as
                 the square of the number of neighbors
    """

    def __init__(self, bins, cutoff=10.0):
        self.bins = bins
        self.cutoff = cutoff

        if self.n_jobs != 1:
            warnings.warn("This featurizer does not support n_jobs > 1.")
            self.set_n_jobs(1)

    def featurize(self, struct, idx):
        """
        Get AFS of the input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.

        Returns:
            Flattened list of AFS values. the list order is:
                g_n g_n'
        """

        if not struct.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Generate list of neighbor position vectors (relative to central
        # atom) and distances from each central site as tuples
        sites = struct._sites
        central_site = sites[idx]
        neighbors_lst = struct.get_neighbors(central_site, self.cutoff)
        neighbor_collection = [
            (neighbor[0].coords - central_site.coords, neighbor[1])
            for neighbor in neighbors_lst]

        # Generate exhaustive permutations of neighbor pairs around each
        # central site (order matters). Does not allow repeat elements (i.e.
        # there are two distinct sites in every permutation)
        neighbor_tuples = itertools.permutations(neighbor_collection, 2)

        # Generate cos(theta) between neighbor pairs for each central site.
        # Also, retain data on neighbor distances for each pair
        # process with matrix algebra, we really need the speed here
        data = np.array(list(neighbor_tuples))
        v1, v2 = np.vstack(data[:, 0, 0]), np.vstack(data[:, 1, 0])
        distances = data[:, :, 1]
        neighbor_pairs = np.concatenate([
            np.clip(np.einsum('ij,ij->i', v1, v2) /
                    np.linalg.norm(v1, axis=1) /
                    np.linalg.norm(v2, axis=1), -1.0, 1.0).reshape(-1, 1),
            distances], axis=1)

        # Generate distance functional matrix (g_n, g_n')
        bin_combos = list(itertools.product(self.bins, repeat=2))

        # Compute AFS values for each element of the bin matrix
        # need to cast arrays as floats to use np.exp
        cos_angles, dist1, dist2 = neighbor_pairs[:, 0].astype(float),\
            neighbor_pairs[:, 1].astype(float),\
            neighbor_pairs[:, 2].astype(float)
        features = [sum(combo[0][1](dist1) * combo[1][1](dist2) *
                        cos_angles) for combo in bin_combos]

        return features

    def feature_labels(self):
        bin_combos = list(itertools.product(self.bins, repeat=2))
        return ['bin {} bin {}'.format(combo[0][0], combo[1][0])
                for combo in bin_combos]

    @staticmethod
    def from_preset(preset, width=0.5, spacing=0.5, cutoff=10):
        '''
        Preset bin functionals for this featurizer. Example use:
            >>> AFS = AngularFourierSeries.from_preset('gaussian')
            >>> AFS.featurize(struct, idx)

        Args:
            preset (str): shape of bin (either 'gaussian' or 'histogram')
            width (float): bin width. std dev for gaussian, width for histogram
            spacing (float): the spacing between bin centers
            cutoff (float): maximum distance to look for neighbors
        '''

        if preset == "gaussian":
            bins = []
            for center in np.arange(0., cutoff, spacing):
                bins.append(('Gauss {}'.format(center),
                             lambda d: np.exp(-width * (d - center)**2.)))
            return AngularFourierSeries(bins, cutoff=cutoff)

        elif preset == "histogram":
            bins = []
            for center in np.arange(0. + (width / 2.), cutoff, spacing):
                bins.append(('Hist {}'.format(center),
                             lambda d: np.where(center - (width / 2.) <= d,
                                                1., 0.) *
                             np.where(d < center + (width / 2.), 1., 0.)))
            return AngularFourierSeries(bins, cutoff=cutoff)

        else:
            raise ValueError('Not a valid preset condition.')

    def citations(self):
        return ['@article{PhysRevB.95.144110, title = {Representation of compo'
                'unds for machine-learning prediction of physical properties},'
                ' author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, '
                'Keita and Takahashi, Akira and Tanaka, Isao},'
                'journal = {Phys. Rev. B}, volume = {95}, issue = {14}, '
                'pages = {144110}, year = {2017}, publisher = {American Physic'
                'al Society}, doi = {10.1103/PhysRevB.95.144110}}']

    def implementors(self):
        return ["Maxwell Dylla"]
