from __future__ import division

"""Features that describe the local environment of a single atom
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

import numpy as np
import math

from collections import defaultdict

from matminer.featurizers.base import BaseFeaturizer
from scipy.spatial import Voronoi, Delaunay
from pymatgen.analysis.local_env import LocalStructOrderParas, \
    VoronoiNN, JMolNN, MinimumDistanceNN, MinimumOKeeffeNN, \
    MinimumVIRENN
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder \
    import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies \
   import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from matminer.featurizers.stats import PropertyStats


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

    def __init__(self, optypes=None, dr=0.1, ddr=0.01, ndr=1, dop=0.001,
                 dist_exp=2, zero_ops=True):
        self.optypes = {
            1: ["sgl_bd"],
            2: ["bent180", "bent45", "bent90", "bent135"],
            3: ["tri_plan", "tet", "T"],
            4: ["sq_plan", "sq", "tet", "see_saw_rect", "tri_pyr"],
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
                    self.ops[cn].append(LocalStructOrderParas(
                        [t[:4]], parameters=[{'TA': float(t[4:]) / 180.0, \
                                              'IGW_TA': 1.0 / 0.0667}]))
                else:
                    self.ops[cn].append(LocalStructOrderParas([t]))

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
        if self.optypes[1][0] == "sgl_bd":
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
                        for it in range(len(self.optypes[cn])):
                            opvals[i].append(0)
                        continue

                    # Set all (remaining) OPs.
                    for it in range(len(self.optypes[cn])):
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
        for cn, li in self.optypes.items():
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
            optypes = {
                1: ["wt"],
                2: ["wt", "bent180", "bent45", "bent90", "bent135"],
                3: ["wt", "tri_plan", "tet", "T"],
                4: ["wt", "sq_plan", "sq", "tet", "see_saw_rect", "tri_pyr"],
                5: ["wt", "pent_plan", "sq_pyr", "tri_bipyr"],
                6: ["wt", "oct", "pent_pyr"],
                7: ["wt", "hex_pyr", "pent_bipyr"],
                8: ["wt", "bcc", "hex_bipyr"],
                9: ["wt", "q2", "q4", "q6"],
                10: ["wt", "q2", "q4", "q6"],
                11: ["wt", "q2", "q4", "q6"],
                12: ["wt", "cuboct", "q2", "q4", "q6"],
                13: ["wt"],
                14: ["wt"],
                15: ["wt"],
                16: ["wt"]}

            return CrystalSiteFingerprint(optypes, cation_anion=cation_anion)

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

                elif t[:4] == 'bent':
                    self.ops[cn].append(LocalStructOrderParas(
                        [t[:4]], parameters=[{'TA': float(t[4:]) / 180.0, \
                                              'IGW_TA': 1.0 / 0.0667}]))
                else:
                    self.ops[cn].append(LocalStructOrderParas([t]))

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
                        target=target)
        n_w = vnn.get_voronoi_polyhedra(idx, struct, use_weights=True)

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
    -Voronoi indices {n_i}
     n_i denotes the number of i-edged facets, and i is in the range of 3-10.
     e.g. for bcc lattice, the Voronoi indices are [0,6,0,8,...];
          for fcc/hcp lattice, the Voronoi indices are [0,12,0,0,...];
          for icosahedra, the Voronoi indices are [0,0,12,0,...];

    -i-fold symmetry indices
     computed as n_i/sum(n_i), and i is in the range of 3-10.
     reflect the strength of i-fold symmetry in local sites.
     e.g. for bcc lattice, the i-fold symmetry indices are [0,6/14,0,8/14,...]
             indicating both 4-fold and a stronger 6-fold symmetries are present;
          for fcc/hcp lattice, the i-fold symmetry factors are [0,1,0,0,...],
             indicating only 4-fold symmetry is present;
          for icosahedra, the Voronoi indices are [0,0,1,0,...],
             indicating only 5-fold symmetry is present;

    -weighted i-fold symmetry indices
     if use_weights = True

    -Voronoi volume
     total volume of the Voronoi polyhedron around the target site

    -Voronoi volume statistics of the sub_polyhedra formed by each facet
     and the center site
     e.g. stats_vol = ['mean', 'std_dev', 'minimum', 'maximum']

    -Voronoi area
     total area of the Voronoi polyhedron around the target site

    -Voronoi area statistics of the facets
     e.g. stats_area = ['mean', 'std_dev', 'minimum', 'maximum']

    -Voronoi nearest-neighboring distance statistics
     e.g. stats_dist = ['mean', 'std_dev', 'minimum', 'maximum']

    Args:
        cutoff (float): cutoff distance in determining the potential
                        neighbors for Voronoi tessellation analysis.
        use_weights(bool): whether to use weights to derive weighted
                           i-fold symmetry indices.
        stats_vol (list of str): volume statistics types.
        stats_area (list of str): area statistics types.
        stats_dist (list of str): neighboring distance statistics types.
    """

    def __init__(self, cutoff=6.0, use_weights=False, stats_vol=None,
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
        vt2, vt3 and vt4
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
            (list of floats): Voronoi fingerprints
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
        voro_idx_list = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        voro_idx_weights = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

        vertices = [struct[idx].coords] + [key.coords for key in n_w.keys()]
        voro = Voronoi(vertices)

        vol_list = []
        area_list = []
        dist_list = [np.linalg.norm(vertices[0] - vertices[i]) for i in
                     range(1, len(vertices))]

        for nn, vind in voro.ridge_dict.items():
            if 0 in nn:
                if -1 in vind:
                    continue
                try:
                    voro_idx_list[len(vind) - 3] += 1
                    if self.use_weights:
                        for key, value in n_w.items():
                            if str(key.coords) == str(vertices[sorted(nn)[1]]):
                                voro_idx_weights[len(vind) - 3] += value

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

        symmetry_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_weights:
            symmetry_wt_list = voro_idx_weights / sum(voro_idx_weights)
            voro_fingerprint = list(np.concatenate((voro_idx_list,
                                                    symmetry_idx_list,
                                                    symmetry_wt_list), axis=0))
        else:
            voro_fingerprint = list(np.concatenate((voro_idx_list,
                                                    symmetry_idx_list), axis=0))

        voro_fingerprint.append(sum(vol_list))
        voro_fingerprint.append(sum(area_list))
        voro_fingerprint += [PropertyStats().calc_stat(vol_list, stat_vol)
                             for stat_vol in self.stats_vol]
        voro_fingerprint += [PropertyStats().calc_stat(area_list, stat_area)
                             for stat_area in self.stats_area]
        voro_fingerprint += [PropertyStats().calc_stat(dist_list, stat_dist)
                             for stat_dist in self.stats_dist]
        return voro_fingerprint

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
                    'title={Spatial tessellations}, '
                    'author={Okabe, Atsuyuki}, '
                    'year={1992}, '
                    'publisher={Wiley Online Library}}']
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

    Here the calculation is run for each element present in the structure.

    A positive f_el indicates the "bonding" with the specific element
    is favored, at least in the target site;
    A negative f_el indicates the "bonding" is not favored, at least
    in the target site.

    Args:
        nn (NearestNeighbor): instance of one of pymatgen's Nearest Neighbor
                              classes.
    """

    @staticmethod
    def from_preset(preset):
        """
        Use one of the standard instances of a given NearNeighbor class.
        Args:
            preset (str): preset type ("VoronoiNN", "JMolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
        Returns:
            ChemicalSRO from a preset.
        """
        if preset == "VoronoiNN":
            return ChemicalSRO(VoronoiNN())
        elif preset == "JMolNN":
            return ChemicalSRO(JMolNN())
        elif preset == "MinimumDistanceNN":
            return ChemicalSRO(MinimumDistanceNN())
        elif preset == "MinimumOKeeffeNN":
            return ChemicalSRO(MinimumOKeeffeNN())
        elif preset == "MinimumVIRENN":
            return ChemicalSRO(MinimumVIRENN())
        else:
            raise RuntimeError('Unknown preset.')

    def __init__(self, nn):
        self.nn = nn
        self.elements = None

    def featurize(self, struct, idx):
        """
        Get CSRO features of site with given index in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Chemical SRO features for each element.
        """
        el_amt = struct.composition.fractional_composition.get_el_amt_dict()
        self.elements = el_amt.keys()
        nn_list = self.nn.get_nn(struct, idx)
        nn_el_amt = dict.fromkeys(el_amt, 0)
        for nn in nn_list:
            nn_el_amt[str(nn.specie)] += 1/len(nn_list)
        csro_el = [el_amt[el] - nn_el_amt[el] for el in self.elements]
        return csro_el

    def feature_labels(self):
        return ['CSRO_{}_{}'.format(el, self.nn.__class__.__name__)
                for el in self.elements]

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


class EwaldSiteEnergy:
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
        #
        #  TODO: Figure out if this implementation is thread-safe! I was debating adding
        #        Locks, but think we are OK
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

    def __init__(self, cetypes, strategy, geom_finder, max_csm=8):
        self.cetypes = tuple(cetypes)
        self.strat = strategy
        self.lgf = geom_finder
        self.max_csm = max_csm

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
        se = self.lgf.compute_structure_environments()
        #        maximum_distance_factor=1.41)
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
    def from_preset(preset):
        """
        Use one of the standard instances of a given NearNeighbor
        class.
        Args:
            preset (str): preset type ("VoronoiNN", "JMolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
        Returns:
            CoordinationNumber from a preset.
        """
        if preset == "VoronoiNN":
            return CoordinationNumber(VoronoiNN())
        elif preset == "JMolNN":
            return CoordinationNumber(JMolNN())
        elif preset == "MinimumDistanceNN":
            return CoordinationNumber(MinimumDistanceNN())
        elif preset == "MinimumOKeeffeNN":
            return CoordinationNumber(MinimumOKeeffeNN())
        elif preset == "MinimumVIRENN":
            return CoordinationNumber(MinimumVIRENN())
        else:
            raise RuntimeError('Unknown preset.')

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
