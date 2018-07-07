import copy
import math

from collections.__init__ import defaultdict
from monty.dev import deprecated

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import cn_target_motif_op, cn_motif_op_params
from matminer.utils.caching import get_nearest_neighbors
from pymatgen.analysis.local_env import LocalStructOrderParams, VoronoiNN


class CrystalSiteFingerprint(BaseFeaturizer):
    """
    A local order parameter fingerprint for periodic crystals.

    A site fingerprint intended for periodic crystals. The fingerprint represents
    the value of various order parameters for the site; each value is the product
    two quantities: (i) the value of the order parameter itself and (ii) a factor
    that describes how consistent the number of neighbors is with that order
    parameter. Note that we can include only factor (ii) using the "wt" order
    parameter which is always set to 1. Also note that the cation-anion flag
    works only if the structures are oxidation-state decorated (e.g., use
    pymatgen's BVAnalyzer or matminer's structure_to_oxidstructure()).
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

    @deprecated(message="CrystalSiteFingerprint is deprecated in favor of "
                        "CrystalNNFingerprint")
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

        self.optypes = copy.deepcopy(optypes)
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

        # Use a Voronoi tessellation to identify neighbors of this site
        vnn = VoronoiNN(cutoff=self.cutoff_radius,
                        targets=target)
        n_w = get_nearest_neighbors(vnn, struct, idx)

        # Convert nn info to just a dict of neighbor -> weight
        n_w = dict((x['site'], x['weight']) for x in n_w)

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