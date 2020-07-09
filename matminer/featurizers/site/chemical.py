from __future__ import division

import numpy as np

from sklearn.utils.validation import check_is_fitted
from pymatgen import Structure
from pymatgen.core.periodic_table import Element
import pymatgen.analysis
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder \
    import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies \
   import SimplestChemenvStrategy, MultiWeightsChemenvStrategy

from matminer.featurizers.base import BaseFeaturizer


class ChemicalSRO(BaseFeaturizer):
    """
    Chemical short range ordering, deviation of local site and nominal structure compositions

    Chemical SRO features to evaluate the deviation
    of local chemistry with the nominal composition of the structure.

    A local bonding preference is computed using
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

    Features:
        CSRO__[nn method]_[element] - The Chemical SRO of a site computed based
            on neighbors determined with a certain  NN-detection method for
            a certain element.
    """

    def __init__(self, nn, includes=None, excludes=None, sort=True):
        """Initialize the featurizer

        Args:
            nn (NearestNeighbor): instance of one of pymatgen's NearestNeighbor
                                  classes.
            includes (array-like or str): elements included to calculate CSRO.
            excludes (array-like or str): elements excluded to calculate CSRO.
            sort (bool): whether to sort elements by mendeleev number."""
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
            preset (str): preset type ("VoronoiNN", "JmolNN",
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
        return ['Qi Wang']


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
                only_indices=[idx],
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

