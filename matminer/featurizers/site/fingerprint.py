"""
Site featurizers that fingerprint a site using local geometry.
"""
import copy
import os

import numpy as np
import pymatgen.analysis.local_env
import ruamel.yaml as yaml
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    MultiWeightsChemenvStrategy,
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams, VoronoiNN

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_nearest_neighbors


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
    :math:`A_i(\\eta) = \\sum\\limits_{i \\ne j} e^{-(\\frac{r_{ij}}{\\eta})^2} f(r_{ij})`
    where :math:`i` is the index of the atom, :math:`j` is the index of a neighboring atom, :math:`\\eta` is a scaling function,
    :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`, and :math:`f(r)` is a cutoff function where
    :math:`f(r) = 0.5[\\cos(\\frac{\\pi r_{ij}}{R_c}) + 1]` if :math:`r < R_c` and :math:`0` otherwise.
    The direction-resolved fingerprints are computed using
    :math:`V_i^k(\\eta) = \\sum\\limits_{i \\ne j} \\frac{r_{ij}^k}{r_{ij}} e^{-(\\frac{r_{ij}}{\\eta})^2} f(r_{ij})`
    where :math:`r_{ij}^k` is the :math:`k^{th}` component of :math:`\\bold{r}_i - \\bold{r}_j`.
    Parameters:
    TODO: Differentiate between different atom types (maybe as another class)
    """

    def __init__(self, directions=(None, "x", "y", "z"), etas=None, cutoff=8):
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
        if any([x in self.directions for x in ["x", "y", "z"]]):
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
                if d == "x":
                    proj = [1.0, 0.0, 0.0]
                elif d == "y":
                    proj = [0.0, 1.0, 0.0]
                elif d == "z":
                    proj = [0.0, 0.0, 1.0]
                else:
                    raise Exception("Unrecognized direction")
                output.append(np.sum(windowed * np.dot(disps, proj)[:, np.newaxis], axis=0))

        # Return the results
        return np.hstack(output)

    def feature_labels(self):
        labels = []
        for d in self.directions:
            for e in self.etas:
                if d is None:
                    labels.append("AGNI eta=%.2e" % e)
                else:
                    labels.append(f"AGNI dir={d} eta={e:.2e}")
        return labels

    def citations(self):
        return [
            "@article{Botu2015, author = {Botu, Venkatesh and Ramprasad, Rampi},doi = {10.1002/qua.24836},"
            "journal = {International Journal of Quantum Chemistry},number = {16},pages = {1074--1083},"
            "title = {{Adaptive machine learning framework to accelerate ab initio molecular dynamics}},"
            "volume = {115},year = {2015}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class OPSiteFingerprint(BaseFeaturizer):
    """
    Local structure order parameters computed from a site's neighbor env.

    For each order parameter, we determine
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

    def __init__(
        self,
        target_motifs=None,
        dr=0.1,
        ddr=0.01,
        ndr=1,
        dop=0.001,
        dist_exp=2,
        zero_ops=True,
    ):

        cn_target_motif_op = load_cn_target_motif_op()
        cn_motif_op_params = load_cn_motif_op_params()

        self.cn_target_motif_op = (
            copy.deepcopy(cn_target_motif_op) if target_motifs is None else copy.deepcopy(target_motifs)
        )
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
                self.ops[cn].append(LocalStructOrderParams([ot], parameters=[p]))

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
        neighbors = []
        r = 6

        while len(neighbors) < 12:
            r += 1.0
            neighbors = struct.get_neighbors(s, r)

        # Smoothen distance, but use relative distances.
        dmin = min(n[1] for n in neighbors)
        neigh_dist = [[n[0], n[1] / dmin] for n in neighbors]

        neigh_dist_alldrs = {}
        d_sorted_alldrs = {}

        for i in range(-self.ndr, self.ndr + 1):
            opvals[i] = []
            this_dr = self.dr + float(i) * self.ddr
            this_idr = 1.0 / this_dr
            neigh_dist_alldrs[i] = []
            for j in range(len(neigh_dist)):
                neigh_dist_alldrs[i].append(
                    [
                        neigh_dist[j][0],
                        (float(int(neigh_dist[j][1] * this_idr + 0.5)) + 0.5) * this_dr,
                    ]
                )
            d_sorted_alldrs[i] = []
            for n, d in neigh_dist_alldrs[i]:
                if d not in d_sorted_alldrs[i]:
                    d_sorted_alldrs[i].append(d)
            d_sorted_alldrs[i] = sorted(d_sorted_alldrs[i])

        # Do q_sgl_bd separately.
        # if self.optypes[1][0] == "sgl_bd":
        if self.cn_target_motif_op[1][0] == "sgl_bd":
            for i in range(-self.ndr, self.ndr + 1):
                site_list = [s]
                for n, dn in neigh_dist_alldrs[i]:
                    site_list.append(n)
                opval = self.ops[1][0].get_order_parameters(
                    site_list, 0, indices_neighs=[j for j in range(1, len(site_list))]
                )
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
                        this_av_inv_drel += 1.0 / (neigh_dist[j][1])
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
                            site_list,
                            0,
                            indices_neighs=[j for j in range(1, len(site_list))],
                        )
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
            hist, bin_edges = np.histogram(op_tmp, bins=nbins, range=(minval, maxval), weights=None, density=False)
            max_hist = max(hist)
            op_peaks = []
            for i, h in enumerate(hist):
                if h == max_hist:
                    op_peaks.append([i, 0.5 * (bin_edges[i] + bin_edges[i + 1])])
            # Address problem that 2 OP values can be close to a bin edge.
            hist2 = []
            op_peaks2 = []
            i = 0
            while i < len(op_peaks):
                if i < len(op_peaks) - 1:
                    if op_peaks[i + 1][0] - op_peaks[i][0] == 1:
                        op_peaks2.append(0.5 * (op_peaks[i][1] + op_peaks[i + 1][1]))
                        hist2.append(hist[op_peaks[i][0]] + hist[op_peaks[i + 1][0]])
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
                labels.append(f"{e} CN_{cn}")
        return labels

    def citations(self):
        return [
            "@article{zimmermann_jain_2017, title={Applications of order"
            " parameter feature vectors}, journal={in progress}, author={"
            "Zimmermann, N. E. R. and Jain, A.}, year={2017}}"
        ]

    def implementors(self):
        return ["Nils E. R. Zimmermann"]


class CrystalNNFingerprint(BaseFeaturizer):
    """
    A local order parameter fingerprint for periodic crystals.

    The fingerprint represents the value of various order parameters for the
    site. The "wt" order parameter describes how consistent a site is with a
    certain coordination number. The remaining order parameters are computed
    by multiplying the "wt" for that coordination number with the OP value.

    The chem_info parameter can be used to also get chemical descriptors that
    describe differences in some chemical parameter (e.g., electronegativity)
    between the central site and the site neighbors.
    """

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use preset parameters to get the fingerprint
        Args:
            preset (str): name of preset ("cn" or "ops")
            **kwargs: other settings to be passed into CrystalNN class
        """
        if preset == "cn":
            op_types = {k + 1: ["wt"] for k in range(24)}
            return CrystalNNFingerprint(op_types, **kwargs)

        elif preset == "ops":
            cn_target_motif_op = load_cn_target_motif_op()
            op_types = copy.deepcopy(cn_target_motif_op)
            for k in range(24):
                if k + 1 in op_types:
                    op_types[k + 1].insert(0, "wt")
                else:
                    op_types[k + 1] = ["wt"]

            return CrystalNNFingerprint(op_types, chem_info=None, **kwargs)

        else:
            raise RuntimeError('preset "{}" is not supported in ' "CrystalNNFingerprint".format(preset))

    def __init__(self, op_types, chem_info=None, **kwargs):
        """
        Initialize the CrystalNNFingerprint. Use the from_preset() function to
        use default params.
        Args:
            op_types (dict): a dict of coordination number (int) to a list of str
                representing the order parameter types
            chem_info (dict): a dict of chemical properties (e.g., atomic mass)
                to dictionaries that map an element to a value
                (e.g., chem_info["Pauling scale"]["O"] = 3.44)
            **kwargs: other settings to be passed into CrystalNN class
        """

        self.op_types = copy.deepcopy(op_types)
        self.cnn = CrystalNN(**kwargs)
        if chem_info is not None:
            self.chem_info = copy.deepcopy(chem_info)
            self.chem_props = list(chem_info.keys())
        else:
            self.chem_info = None
        cn_motif_op_params = load_cn_motif_op_params()

        self.ops = {}  # load order parameter objects & paramaters
        for cn, t_list in self.op_types.items():
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
                    self.ops[cn].append(LocalStructOrderParams([ot], parameters=[p]))

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

        nndata = self.cnn.get_nn_data(struct, idx)
        max_cn = sorted(self.op_types)[-1]

        cn_fingerprint = []

        if self.chem_info is not None:
            prop_delta = {}  # dictionary of chemical property to final value
            for prop in self.chem_props:
                prop_delta[prop] = 0
            sum_wt = 0
            elem_central = struct.sites[idx].specie.symbol
            specie_central = str(struct.sites[idx].specie)

        for k in range(max_cn):
            cn = k + 1
            wt = nndata.cn_weights.get(cn, 0)
            if cn in self.ops:
                for op in self.ops[cn]:
                    if op == "wt":
                        cn_fingerprint.append(wt)

                        if self.chem_info is not None and wt != 0:
                            # Compute additional chemistry-related features
                            sum_wt += wt
                            neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]

                            for prop in self.chem_props:
                                # get the value for specie, if not fall back to
                                # value defined for element
                                prop_central = self.chem_info[prop].get(
                                    specie_central,
                                    self.chem_info[prop].get(elem_central),
                                )

                                for neigh in neigh_sites:
                                    elem_neigh = neigh.specie.symbol
                                    specie_neigh = str(neigh.specie)
                                    prop_neigh = self.chem_info[prop].get(
                                        specie_neigh,
                                        self.chem_info[prop].get(elem_neigh),
                                    )

                                    prop_delta[prop] += wt * (prop_neigh - prop_central) / cn

                    elif wt == 0:
                        cn_fingerprint.append(wt)
                    else:
                        neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]
                        opval = op.get_order_parameters(
                            [struct[idx]] + neigh_sites,
                            0,
                            indices_neighs=[i for i in range(1, len(neigh_sites) + 1)],
                        )[0]
                        opval = opval or 0  # handles None
                        cn_fingerprint.append(wt * opval)
        chem_fingerprint = []

        if self.chem_info is not None:
            for val in prop_delta.values():
                chem_fingerprint.append(val / sum_wt)

        return cn_fingerprint + chem_fingerprint

    def feature_labels(self):
        labels = []
        max_cn = sorted(self.op_types)[-1]
        for k in range(max_cn):
            cn = k + 1
            if cn in list(self.ops.keys()):
                for op in self.op_types[cn]:
                    labels.append(f"{op} CN_{cn}")
        if self.chem_info is not None:
            for prop in self.chem_props:
                labels.append(f"{prop} local diff")
        return labels

    def citations(self):
        return []

    def implementors(self):
        return ["Anubhav Jain", "Nils E.R. Zimmermann"]


class VoronoiFingerprint(BaseFeaturizer):
    """
    Voronoi tessellation-based features around target site.

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
        stats_vol = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi area
        total area of the Voronoi polyhedron around the target site
    Voronoi area statistics of the facets
        stats_area = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi nearest-neighboring distance statistics
        stats_dist = ['mean', 'std_dev', 'minimum', 'maximum']

    Args:
        cutoff (float): cutoff distance in determining the potential
                        neighbors for Voronoi tessellation analysis.
                        (default: 6.5)
        use_symm_weights(bool): whether to use weights to derive weighted
                                i-fold symmetry indices.
        symm_weights(str): weights to be used in weighted i-fold symmetry
                           indices.
                           Supported options: 'solid_angle', 'area', 'volume',
                           'face_dist'. (default: 'solid_angle')
        stats_vol (list of str): volume statistics types.
        stats_area (list of str): area statistics types.
        stats_dist (list of str): neighboring distance statistics types.
    """

    def __init__(
        self,
        cutoff=6.5,
        use_symm_weights=False,
        symm_weights="solid_angle",
        stats_vol=None,
        stats_area=None,
        stats_dist=None,
    ):
        self.cutoff = cutoff
        self.use_symm_weights = use_symm_weights
        self.symm_weights = symm_weights
        self.stats_vol = ["mean", "std_dev", "minimum", "maximum"] if stats_vol is None else copy.deepcopy(stats_vol)
        self.stats_area = ["mean", "std_dev", "minimum", "maximum"] if stats_area is None else copy.deepcopy(stats_area)
        self.stats_dist = ["mean", "std_dev", "minimum", "maximum"] if stats_dist is None else copy.deepcopy(stats_dist)

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
                -weighted i-fold symmetry indices (if use_symm_weights = True)
                -Voronoi volume
                -Voronoi volume statistics
                -Voronoi area
                -Voronoi area statistics
                -Voronoi dist statistics
        """
        # Get the nearest neighbors using a Voronoi tessellation
        n_w = get_nearest_neighbors(VoronoiNN(cutoff=self.cutoff), struct, idx)

        # Prepare storage for the Voronoi indices
        voro_idx_list = np.zeros(8, int)
        voro_idx_weights = np.zeros(8)
        vol_list = []
        area_list = []
        dist_list = []

        # Get statistics
        for nn in n_w:
            if nn["poly_info"]["n_verts"] <= 10:
                # If a facet has more than 10 edges, it's skipped here.
                voro_idx_list[nn["poly_info"]["n_verts"] - 3] += 1
                vol_list.append(nn["poly_info"]["volume"])
                area_list.append(nn["poly_info"]["area"])
                dist_list.append(nn["poly_info"]["face_dist"] * 2)
                if self.use_symm_weights:
                    voro_idx_weights[nn["poly_info"]["n_verts"] - 3] += nn["poly_info"][self.symm_weights]

        symm_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_symm_weights:
            symm_wt_list = voro_idx_weights / sum(voro_idx_weights)
            voro_fps = list(np.concatenate((voro_idx_list, symm_idx_list, symm_wt_list), axis=0))
        else:
            voro_fps = list(np.concatenate((voro_idx_list, symm_idx_list), axis=0))

        voro_fps.append(sum(vol_list))
        voro_fps.append(sum(area_list))
        voro_fps += [PropertyStats().calc_stat(vol_list, stat_vol) for stat_vol in self.stats_vol]
        voro_fps += [PropertyStats().calc_stat(area_list, stat_area) for stat_area in self.stats_area]
        voro_fps += [PropertyStats().calc_stat(dist_list, stat_dist) for stat_dist in self.stats_dist]
        return voro_fps

    def feature_labels(self):
        labels = ["Voro_index_%d" % i for i in range(3, 11)]
        labels += ["Symmetry_index_%d" % i for i in range(3, 11)]
        if self.use_symm_weights:
            labels += ["Symmetry_weighted_index_%d" % i for i in range(3, 11)]
        labels.append("Voro_vol_sum")
        labels.append("Voro_area_sum")
        labels += ["Voro_vol_%s" % stat_vol for stat_vol in self.stats_vol]
        labels += ["Voro_area_%s" % stat_area for stat_area in self.stats_area]
        labels += ["Voro_dist_%s" % stat_dist for stat_dist in self.stats_dist]
        return labels

    def citations(self):
        voronoi_citation = (
            "@book{okabe1992spatial,  "
            "title  = {Spatial tessellations}, "
            "author = {Okabe, Atsuyuki}, "
            "year   = {1992}, "
            "publisher = {Wiley Online Library}}"
        )
        symm_idx_citation = (
            "@article{peng2011, "
            "title={Structural signature of plastic deformation in metallic "
            "glasses}, "
            "author={H L Peng, M Z Li and W H Wang}, "
            "journal={Physical Review Letters}, year={2011}}, "
            "pages = {135503}, volume = {106}, issue = {13}, "
            "doi = {10.1103/PhysRevLett.106.135503}}"
        )
        nn_stats_citation = (
            "@article{Wang2019, "
            "title = {A transferable machine-learning framework linking "
            "interstice distribution and plastic heterogeneity in metallic "
            "glasses}, "
            "author = {Qi Wang and Anubhav Jain}, "
            "journal = {Nature Communications}, year = {2019}, "
            "pages = {5537}, volume = {10}, "
            "doi = {10.1038/s41467-019-13511-9}, "
            "url = {https://www.nature.com/articles/s41467-019-13511-9}}"
        )
        return [voronoi_citation, symm_idx_citation, nn_stats_citation]

    def implementors(self):
        return ["Qi Wang"]


class ChemEnvSiteFingerprint(BaseFeaturizer):
    """
    Resemblance of given sites to ideal environments

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
            "S:1",
            "L:2",
            "A:2",
            "TL:3",
            "TY:3",
            "TS:3",
            "T:4",
            "S:4",
            "SY:4",
            "SS:4",
            "PP:5",
            "S:5",
            "T:5",
            "O:6",
            "T:6",
            "PP:6",
            "PB:7",
            "ST:7",
            "ET:7",
            "FO:7",
            "C:8",
            "SA:8",
            "SBT:8",
            "TBT:8",
            "DD:8",
            "DDPN:8",
            "HB:8",
            "BO_1:8",
            "BO_2:8",
            "BO_3:8",
            "TC:9",
            "TT_1:9",
            "TT_2:9",
            "TT_3:9",
            "HD:9",
            "TI:9",
            "SMA:9",
            "SS:9",
            "TO_1:9",
            "TO_2:9",
            "TO_3:9",
            "PP:10",
            "PA:10",
            "SBSA:10",
            "MI:10",
            "S:10",
            "H:10",
            "BS_1:10",
            "BS_2:10",
            "TBSA:10",
            "PCPA:11",
            "H:11",
            "SH:11",
            "CO:11",
            "DI:11",
            "I:12",
            "PBP:12",
            "TT:12",
            "C:12",
            "AC:12",
            "SC:12",
            "S:12",
            "HP:12",
            "HA:12",
            "SH:13",
            "DD:20",
        ]
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(
            centering_type="centroid",
            include_central_site_in_centroid=True,
            structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE,
        )
        if preset == "simple":
            return ChemEnvSiteFingerprint(
                cetypes,
                SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3),
                lgf,
            )
        elif preset == "multi_weights":
            return ChemEnvSiteFingerprint(
                cetypes,
                MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
                lgf,
            )
        else:
            raise RuntimeError("unknown neighbor-finding strategy preset.")

    def __init__(self, cetypes, strategy, geom_finder, max_csm=8, max_dist_fac=1.41):
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
        se = self.lgf.compute_structure_environments(only_indices=[idx], maximum_distance_factor=self.max_dist_fac)
        for ce in self.cetypes:
            try:
                tmp = se.get_csms(idx, ce)
                tmp = tmp[0]["symmetry_measure"] if len(tmp) != 0 else self.max_csm
                tmp = tmp if tmp < self.max_csm else self.max_csm
                cevals.append(1 - tmp / self.max_csm)
            except IndexError:
                cevals.append(0)
        return np.array(cevals)

    def feature_labels(self):
        return list(self.cetypes)

    def citations(self):
        return [
            "@article{waroquiers_chemmater_2017, "
            "title={Statistical analysis of coordination environments "
            "in oxides}, journal={Chemistry of Materials},"
            "author={Waroquiers, D. and Gonze, X. and Rignanese, G.-M."
            "and Welker-Nieuwoudt, C. and Rosowski, F. and Goebel, M. "
            "and Schenk, S. and Degelmann, P. and Andre, R. "
            "and Glaum, R. and Hautier, G.}, year={2017}}"
        ]

    def implementors(self):
        return ["Nils E. R. Zimmermann"]


def load_cn_motif_op_params():
    """
    Load the file for the local env motif parameters into a dictionary.

    Returns:
        (dict)
    """
    with open(
        os.path.join(os.path.dirname(pymatgen.analysis.local_env.__file__), "cn_opt_params.yaml"),
    ) as f:
        return yaml.safe_load(f)


def load_cn_target_motif_op():
    """
    Load the file fpor the

    Returns:
        (dict)
    """
    with open(os.path.join(os.path.dirname(__file__), "cn_target_motif_op.yaml")) as f:
        return yaml.safe_load(f)
