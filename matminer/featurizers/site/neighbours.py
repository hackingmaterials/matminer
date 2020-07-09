from __future__ import division

import os
import copy
import numpy as np
import ruamel.yaml as yaml

import pymatgen.analysis
from pymatgen.analysis.local_env import (
    LocalStructOrderParams,
    VoronoiNN,
    CrystalNN,
    )

from matminer.utils.caching import get_nearest_neighbors
from matminer.featurizers.utils.stats import PropertyStats
from matminer.featurizers.base import BaseFeaturizer

cn_motif_op_params = {}
with open(os.path.join(os.path.dirname(
        pymatgen.analysis.__file__), 'cn_opt_params.yaml'), 'r') as f:
    cn_motif_op_params = yaml.safe_load(f)

cn_target_motif_op = {}
with open(os.path.join(os.path.dirname(
        __file__), 'cn_target_motif_op.yaml'), 'r') as f:
    cn_target_motif_op = yaml.safe_load(f)


class AverageBondLength(BaseFeaturizer):
    '''
    Determines the average bond length between one specific site
    and all its nearest neighbors using one of pymatgen's NearNeighbor
    classes. These nearest neighbor calculators return weights related
    to the proximity of each neighbor to this site. 'Average bond
    length' of a site is the weighted average of the distance between
    site and all its nearest neighbors.
    '''

    def __init__(self, method):
        '''
        Initialize featurizer

        Args:
            method (NearNeighbor) - subclass under NearNeighbor used to compute nearest neighbors
        '''
        self.method = method

    def featurize(self, strc, idx):
        '''
        Get weighted average bond length of a site and all its nearest
        neighbors.

        Args:
            strc (Structure): Pymatgen Structure object
            idx (int): index of target site in structure object

        Returns:
            average bond length (list)
        '''
        # Compute nearest neighbors of the indexed site
        nns = self.method.get_nn_info(strc, idx)
        if len(nns) == 0:
            raise IndexError("Input structure has no bonds.")

        weights = [info['weight'] for info in nns]
        center_coord = strc[idx].coords

        dists = np.linalg.norm(np.subtract([site['site'].coords for site in nns], center_coord), axis=1)

        return [PropertyStats.mean(dists, weights)]

    def feature_labels(self):
        return ['Average bond length']

    def citations(self):
        return ['@article{jong_chen_notestine_persson_ceder_jain_asta_gamst_2016,'
                'title={A Statistical Learning Framework for Materials Science: '
                'Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds}, '
                'volume={6}, DOI={10.1038/srep34256}, number={1}, journal={Scientific Reports}, '
                'author={Jong, Maarten De and Chen, Wei and Notestine, Randy and Persson, '
                'Kristin and Ceder, Gerbrand and Jain, Anubhav and Asta, Mark and Gamst, Anthony}, '
                'year={2016}, month={Mar}}'
                ]

    def implementors(self):
        return ['Logan Ward', 'Aik Rui Tan']


class AverageBondAngle(BaseFeaturizer):
    '''
    Determines the average bond angles of a specific site with
    its nearest neighbors using one of pymatgen's NearNeighbor
    classes. Neighbors that are adjacent to each other are stored
    and angle between them are computed. 'Average bond angle' of
    a site is the mean bond angle between all its nearest neighbors.
    '''

    def __init__(self, method):
        '''
        Initialize featurizer

        Args:
            method (NearNeighbor) - subclass under NearNeighbor used to compute nearest
                                    neighbors
        '''
        self.method = method

    def featurize(self, strc, idx):
        '''
        Get average bond length of a site and all its nearest
        neighbors.

        Args:
            strc (Structure): Pymatgen Structure object
            idx (int): index of target site in structure object

        Returns:
            average bond length (list)
        '''
        # Compute nearest neighbors of the indexed site
        nns = self.method.get_nn_info(strc, idx)
        if len(nns) == 0:
            raise IndexError("Input structure has no bonds.")
        center = strc[idx].coords

        sites = [i['site'].coords for i in nns]

        # Calculate bond angles for each neighbor
        bond_angles = np.empty((len(sites), len(sites)))
        bond_angles.fill(np.nan)
        for a, a_site in enumerate(sites):
            for b, b_site in enumerate(sites):
                if (b == a):
                    continue
                dot = np.dot(a_site - center, b_site - center) / (
                            np.linalg.norm(a_site - center) * np.linalg.norm(b_site - center))
                if np.isnan(np.arccos(dot)):
                    bond_angles[a, b] = bond_angles[b, a] = np.arccos(round(dot, 5))
                else:
                    bond_angles[a, b] = bond_angles[b, a] = np.arccos(dot)
        # Take the minimum bond angle of each neighbor
        minimum_bond_angles = np.nanmin(bond_angles, axis=1)

        return [PropertyStats.mean(minimum_bond_angles)]

    def feature_labels(self):
        return ['Average bond angle']

    def citations(self):
        return ['@article{jong_chen_notestine_persson_ceder_jain_asta_gamst_2016,'
                'title={A Statistical Learning Framework for Materials Science: '
                'Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds}, '
                'volume={6}, DOI={10.1038/srep34256}, number={1}, journal={Scientific Reports}, '
                'author={Jong, Maarten De and Chen, Wei and Notestine, Randy and Persson, '
                'Kristin and Ceder, Gerbrand and Jain, Anubhav and Asta, Mark and Gamst, Anthony}, '
                'year={2016}, month={Mar}}'
                ]

    def implementors(self):
        return ['Logan Ward', 'Aik Rui Tan']


class CoordinationNumber(BaseFeaturizer):
    """
    Number of first nearest neighbors of a site.

    Determines the number of nearest neighbors of a site using one of
    pymatgen's NearNeighbor classes. These nearest neighbor calculators
    can return weights related to the proximity of each neighbor to this
    site. It is possible to take these weights into account to prevent
    the coordination number from changing discontinuously with small
    perturbations of a structure, either by summing the total weights
    or using the normalization method presented by
    [Ward et al.](http://link.aps.org/doi/10.1103/PhysRevB.96.014107)

    Features:
        CN_[method] - Coordination number computed using a certain method
            for calculating nearest neighbors.
    """

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use one of the standard instances of a given NearNeighbor class.
        Args:
            preset (str): preset type ("VoronoiNN", "JmolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
            **kwargs: allow to pass args to the NearNeighbor class.
        Returns:
            CoordinationNumber from a preset.
        """
        nn_ = getattr(pymatgen.analysis.local_env, preset)
        return CoordinationNumber(nn_(**kwargs))

    def __init__(self, nn=None, use_weights='none'):
        """Initialize the featurizer

        Args:
            nn (NearestNeighbor) - Method used to determine coordination number
            use_weights (string) - Method used to account for weights of neighbors:
                'none' - Do not use weights when computing coordination number
                'sum' - Use sum of weights as the coordination number
                'effective' - Compute the 'effective coordination number', which
                    is computed as :math:`\\frac{(\sum_n w_n)^2)}{\sum_n w_n^2}`
            """
        self.nn = nn or VoronoiNN()
        self.use_weights = use_weights

    def featurize(self, struct, idx):
        """
        Get coordintion number of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            [float] - Coordination number
        """
        if self.use_weights is None or self.use_weights == 'none':
            return [self.nn.get_cn(struct, idx, use_weights=False)]
        elif self.use_weights == 'sum':
            return [self.nn.get_cn(struct, idx, use_weights=True)]
        elif self.use_weights == 'effective':
            # TODO: Should this weighting code go in pymatgen? I'm not sure if it even necessary to distinguish it from the 'sum' method -lw
            nns = get_nearest_neighbors(self.nn, struct, idx)
            weights = [n['weight'] for n in nns]
            return [np.sum(weights) ** 2 / np.sum(np.power(weights, 2))]
        else:
            raise ValueError('Weighting method not recognized: ' + str(self.use_weights))

    def feature_labels(self):
        # TODO: Should names contain weighting scheme? -lw
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
        if self.nn.__class__.__name__ == 'JmolNN':
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
        return ['Nils E. R. Zimmermann', 'Logan Ward']


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

    def __init__(self, cutoff=6.5,
                 use_symm_weights=False, symm_weights='solid_angle',
                 stats_vol=None, stats_area=None, stats_dist=None):
        self.cutoff = cutoff
        self.use_symm_weights = use_symm_weights
        self.symm_weights = symm_weights
        self.stats_vol = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_vol is None else copy.deepcopy(stats_vol)
        self.stats_area = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_area is None else copy.deepcopy(stats_area)
        self.stats_dist = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_dist is None else copy.deepcopy(stats_dist)

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
            if nn['poly_info']['n_verts'] <= 10:
                # If a facet has more than 10 edges, it's skipped here.
                voro_idx_list[nn['poly_info']['n_verts'] - 3] += 1
                vol_list.append(nn['poly_info']['volume'])
                area_list.append(nn['poly_info']['area'])
                dist_list.append(nn['poly_info']['face_dist'] * 2)
                if self.use_symm_weights:
                    voro_idx_weights[nn['poly_info']['n_verts'] - 3] += \
                        nn['poly_info'][self.symm_weights]

        symm_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_symm_weights:
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
        if self.use_symm_weights:
            labels += ['Symmetry_weighted_index_%d' % i for i in range(3, 11)]
        labels.append('Voro_vol_sum')
        labels.append('Voro_area_sum')
        labels += ['Voro_vol_%s' % stat_vol for stat_vol in self.stats_vol]
        labels += ['Voro_area_%s' % stat_area for stat_area in self.stats_area]
        labels += ['Voro_dist_%s' % stat_dist for stat_dist in self.stats_dist]
        return labels

    def citations(self):
        voronoi_citation = (
            '@book{okabe1992spatial,  '
            'title  = {Spatial tessellations}, '
            'author = {Okabe, Atsuyuki}, '
            'year   = {1992}, '
            'publisher = {Wiley Online Library}}')
        symm_idx_citation = (
            '@article{peng2011, '
            'title={Structural signature of plastic deformation in metallic '
            'glasses}, '
            'author={H L Peng, M Z Li and W H Wang}, '
            'journal={Physical Review Letters}, year={2011}}, '
            'pages = {135503}, volume = {106}, issue = {13}, '
            'doi = {10.1103/PhysRevLett.106.135503}}')
        nn_stats_citation = (
            '@article{Wang2019, '
            'title = {A transferable machine-learning framework linking '
            'interstice distribution and plastic heterogeneity in metallic '
            'glasses}, '
            'author = {Qi Wang and Anubhav Jain}, '
            'journal = {Nature Communications}, year = {2019}, '
            'pages = {5537}, volume = {10}, '
            'doi = {10.1038/s41467-019-13511-9}, '
            'url = {https://www.nature.com/articles/s41467-019-13511-9}}')
        return [voronoi_citation, symm_idx_citation, nn_stats_citation]

    def implementors(self):
        return ['Qi Wang']


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

    def __init__(self, target_motifs=None, dr=0.1, ddr=0.01, ndr=1, dop=0.001,
                 dist_exp=2, zero_ops=True):
        self.cn_target_motif_op = copy.deepcopy(cn_target_motif_op) \
            if target_motifs is None else copy.deepcopy(target_motifs)
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
        dmin = min([n[1] for n in neighbors])
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
                    [neigh_dist[j][0], (float(int(neigh_dist[j][1] * this_idr \
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
                weights=None, density=False)
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
            op_types = dict([(k + 1, ["wt"]) for k in range(24)])
            return CrystalNNFingerprint(op_types, **kwargs)

        elif preset == "ops":
            op_types = copy.deepcopy(cn_target_motif_op)
            for k in range(24):
                if k + 1 in op_types:
                    op_types[k + 1].insert(0, "wt")
                else:
                    op_types[k + 1] = ["wt"]

            return CrystalNNFingerprint(op_types, chem_info=None, **kwargs)

        else:
            raise RuntimeError('preset "{}" is not supported in '
                               'CrystalNNFingerprint'.format(preset))

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
                            neigh_sites = [d["site"] for d in
                                           nndata.cn_nninfo[cn]]

                            for prop in self.chem_props:
                                # get the value for specie, if not fall back to
                                # value defined for element
                                prop_central = self.chem_info[prop].get(
                                    specie_central, self.chem_info[prop].get(
                                        elem_central))

                                for neigh in neigh_sites:
                                    elem_neigh = neigh.specie.symbol
                                    specie_neigh = str(neigh.specie)
                                    prop_neigh = self.chem_info[prop].get(
                                        specie_neigh,
                                        self.chem_info[prop].get(
                                            elem_neigh))

                                    prop_delta[prop] += wt * \
                                                           (prop_neigh -
                                                            prop_central) / cn

                    elif wt == 0:
                        cn_fingerprint.append(wt)
                    else:
                        neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]
                        opval = op.get_order_parameters(
                            [struct[idx]] + neigh_sites, 0,
                            indices_neighs=[i for i in
                                            range(1, len(neigh_sites) + 1)])[0]
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
                    labels.append("{} CN_{}".format(op, cn))
        if self.chem_info is not None:
            for prop in self.chem_props:
                labels.append("{} local diff".format(prop))
        return labels

    def citations(self):
        return []

    def implementors(self):
        return ['Anubhav Jain', 'Nils E.R. Zimmermann']

