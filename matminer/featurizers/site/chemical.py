"""
Site featurizers based on local chemical information, rather than geometry alone.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import (
    LocalStructOrderParams,
    VoronoiNN,
)
import pymatgen.analysis.local_env
from pymatgen.analysis.ewald import EwaldSummation


from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.caching import get_nearest_neighbors
from matminer.utils.data import MagpieData


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
            self.includes = [Element(el).symbol for el in np.atleast_1d(self.includes)]
        self.excludes = excludes
        if self.excludes:
            self.excludes = [Element(el).symbol for el in np.atleast_1d(self.excludes)]
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
            raise TypeError("This fit requires an array-like input of Pymatgen " "Structures and sites!")

        self.el_amt_dict_ = {}
        el_set_ = set()
        for s in structs:
            if str(s) not in self.el_amt_dict_.keys():
                el_amt_ = s.composition.fractional_composition.get_el_amt_dict()
                els_ = (
                    set(el_amt_.keys())
                    if self.includes is None
                    else set([el for el in el_amt_.keys() if el in self.includes])
                )
                els_ = els_ if self.excludes is None else els_ - set(self.excludes)
                if els_:
                    self.el_amt_dict_[str(s)] = el_amt_
                el_set_ = el_set_ | els_
        self.el_list_ = sorted(list(el_set_), key=lambda el: Element(el).mendeleev_no) if self.sort else list(el_set_)
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

        check_is_fitted(self, ["el_amt_dict_", "el_list_"])

        csro = [0.0] * len(self.el_list_)
        if str(struct) in self.el_amt_dict_.keys():
            el_amt = self.el_amt_dict_[str(struct)]
            nn_el_amt = dict.fromkeys(el_amt, 0)
            nn_list = self.nn.get_nn(struct, idx)
            for nn in nn_list:
                if str(nn.specie.symbol) in self.el_list_:
                    nn_el_amt[str(nn.specie.symbol)] += 1 / len(nn_list)
            for el in el_amt.keys():
                if el in self.el_list_:
                    csro[self.el_list_.index(el)] = nn_el_amt[el] - el_amt[el]
        return csro

    def feature_labels(self):
        check_is_fitted(self, ["el_amt_dict_", "el_list_"])

        return ["CSRO_{}_{}".format(el, self.nn.__class__.__name__) for el in self.el_list_]

    def citations(self):
        citations = []
        if self.nn.__class__.__name__ == "VoronoiNN":
            citations.append(
                "@article{voronoi_jreineangewmath_1908, title={"
                "Nouvelles applications des param\\`{e}tres continus \\`{a} la "
                "th'{e}orie des formes quadratiques. Sur quelques "
                "propri'{e}t'{e}s des formes quadratiques positives"
                ' parfaites}, journal={Journal f"ur die reine und angewandte '
                "Mathematik}, number={133}, pages={97-178}, year={1908}}"
            )
            citations.append(
                "@article{dirichlet_jreineangewmath_1850, title={"
                '"{U}ber die Reduction der positiven quadratischen Formen '
                "mit drei unbestimmten ganzen Zahlen}, journal={Journal "
                'f"ur die reine und angewandte Mathematik}, number={40}, '
                "pages={209-227}, doi={10.1515/crll.1850.40.209}, year={1850}}"
            )
        if self.nn.__class__.__name__ == "JmolNN":
            citations.append(
                "@misc{jmol, title = {Jmol: an open-source Java "
                "viewer for chemical structures in 3D}, howpublished = {"
                "\\url{http://www.jmol.org/}}}"
            )
        if self.nn.__class__.__name__ == "MinimumOKeeffeNN":
            citations.append(
                "@article{okeeffe_jamchemsoc_1991, title={Atom "
                "sizes and bond lengths in molecules and crystals}, journal="
                "{Journal of the American Chemical Society}, author={"
                "O'Keeffe, M. and Brese, N. E.}, number={113}, pages={"
                "3226-3229}, doi={doi:10.1021/ja00009a002}, year={1991}}"
            )
        if self.nn.__class__.__name__ == "MinimumVIRENN":
            citations.append(
                "@article{shannon_actacryst_1976, title={"
                "Revised effective ionic radii and systematic studies of "
                "interatomic distances in halides and chalcogenides}, "
                "journal={Acta Crystallographica}, author={Shannon, R. D.}, "
                "number={A32}, pages={751-767}, doi={"
                "10.1107/S0567739476001551}, year={1976}"
            )
        if self.nn.__class__.__name__ in [
            "MinimumDistanceNN",
            "MinimumOKeeffeNN",
            "MinimumVIRENN",
        ]:
            citations.append(
                "@article{zimmermann_frontmater_2017, "
                "title={Assessing local structure motifs using order "
                "parameters for motif recognition, interstitial "
                "identification, and diffusion path characterization}, "
                "journal={Frontiers in Materials}, author={Zimmermann, "
                "N. E. R. and Horton, M. K. and Jain, A. and Haranczyk, M.}, "
                "number={4:34}, doi={10.3389/fmats.2017.00034}, year={2017}}"
            )
        return citations

    def implementors(self):
        return ["Qi Wang"]


class EwaldSiteEnergy(BaseFeaturizer):
    """
    Compute site energy from Coulombic interactions

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
        return [
            "@Article{Ewald1921,"
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
            "}"
        ]


# TODO: Figure out whether to take NN-counting method as an option (see VoronoiFingerprint)
class LocalPropertyDifference(BaseFeaturizer):
    """
    Differences in elemental properties between site and its neighboring sites.

    Uses the Voronoi tessellation of the structure to determine the
    neighbors of the site, and assigns each neighbor (:math:`n`) a
    weight (:math:`A_n`) that corresponds to the area of the facet
    on the tessellation corresponding to that neighbor.
    The local property difference is then computed by
    :math:`\\frac{\sum_n {A_n |p_n - p_0|}}{\sum_n {A_n}}`
    where :math:`p_n` is the property (e.g., atomic number) of a neighbor
    and :math:`p_0` is the property of a site. If signed parameter is assigned
    True, signed difference of the properties is returned instead of absolute
    difference.

    Features:
        - "local property difference in [property]" - Weighted average
            of differences between an elemental property of a site and
            that of each of its neighbors, weighted by size of face on
            Voronoi tessellation

    References:
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.024104>`_
    """

    def __init__(
        self,
        data_source=MagpieData(),
        weight="area",
        properties=("Electronegativity",),
        signed=False,
    ):
        """Initialize the featurizer

        Args:
            data_source (AbstractData) - Class from which to retrieve
                elemental properties
            weight (str) - What aspect of each voronoi facet to use to
                weigh each neighbor (see VoronoiNN)
            properties ([str]) - List of properties to use (default=['Electronegativity'])
            signed (bool) - whether to return absolute difference or signed difference of
                            properties(default=False (absolute difference))
        """
        self.data_source = data_source
        self.properties = properties
        self.weight = weight
        self.signed = signed

    @staticmethod
    def from_preset(preset):
        """
        Create a new LocalPropertyDifference class according to a preset

        Args:
            preset (str) - Name of preset
        """

        if preset == "ward-prb-2017":
            return LocalPropertyDifference(
                data_source=MagpieData(),
                properties=[
                    "Number",
                    "MendeleevNumber",
                    "AtomicWeight",
                    "MeltingT",
                    "Column",
                    "Row",
                    "CovalentRadius",
                    "Electronegativity",
                    "NsValence",
                    "NpValence",
                    "NdValence",
                    "NfValence",
                    "NValence",
                    "NsUnfilled",
                    "NpUnfilled",
                    "NdUnfilled",
                    "NfUnfilled",
                    "NUnfilled",
                    "GSvolume_pa",
                    "GSbandgap",
                    "GSmagmom",
                    "SpaceGroupNumber",
                ],
            )
        else:
            raise ValueError("Unrecognized preset: " + preset)

    def featurize(self, strc, idx):
        # Get the targeted site
        my_site = strc[idx]

        # Get the tessellation of a site
        nn = get_nearest_neighbors(VoronoiNN(weight=self.weight), strc, idx)

        # Get the element and weight of each site
        elems = [n["site"].specie for n in nn]
        weights = [n["weight"] for n in nn]

        # Compute the difference for each property
        output = np.zeros((len(self.properties),))
        total_weight = np.sum(weights)
        for i, p in enumerate(self.properties):
            my_prop = self.data_source.get_elemental_property(my_site.specie, p)
            n_props = self.data_source.get_elemental_properties(elems, p)
            if self.signed == False:
                output[i] = np.dot(weights, np.abs(np.subtract(n_props, my_prop))) / total_weight
            else:
                output[i] = np.dot(weights, np.subtract(n_props, my_prop)) / total_weight

        return output

    def feature_labels(self):
        if self.signed == False:
            return ["local difference in " + p for p in self.properties]
        else:
            return ["local signed difference in " + p for p in self.properties]

    def citations(self):
        return [
            "@article{Ward2017,"
            "author = {Ward, Logan and Liu, Ruoqian "
            "and Krishna, Amar and Hegde, Vinay I. "
            "and Agrawal, Ankit and Choudhary, Alok "
            "and Wolverton, Chris},"
            "doi = {10.1103/PhysRevB.96.024104},"
            "journal = {Physical Review B},"
            "pages = {024104},"
            "title = {{Including crystal structure attributes "
            "in machine learning models of formation energies "
            "via Voronoi tessellations}},"
            "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.014107},"
            "volume = {96},year = {2017}}",
            "@article{jong_chen_notestine_persson_ceder_jain_asta_gamst_2016,"
            "title={A Statistical Learning Framework for Materials Science: "
            "Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds}, "
            "volume={6}, DOI={10.1038/srep34256}, number={1}, journal={Scientific Reports}, "
            "author={Jong, Maarten De and Chen, Wei and Notestine, Randy and Persson, "
            "Kristin and Ceder, Gerbrand and Jain, Anubhav and Asta, Mark and Gamst, Anthony}, "
            "year={2016}, month={Mar}}",
        ]

    def implementors(self):
        return ["Logan Ward", "Aik Rui Tan"]


class SiteElementalProperty(BaseFeaturizer):
    """
    Elemental properties of atom on a certain site

    Features:
        site [property] - Elemental property for this site

    References:
        `Seko et al., _PRB_ (2017) <http://link.aps.org/doi/10.1103/PhysRevB.95.144110>`_
        `Schmidt et al., _Chem Mater_. (2017) <http://dx.doi.org/10.1021/acs.chemmater.7b00156>`_
    """

    def __init__(self, data_source=None, properties=("Number",)):
        """Initialize the featurizer

        Args:
            data_source (AbstractData): Tool used to look up elemental properties
            properties ([string]): List of properties to use for features
        """
        self.data_source = data_source or MagpieData()
        self.properties = properties
        self._preset_citations = []

    def featurize(self, strc, idx):
        # Get the site
        site = strc[idx]

        # Get the properties
        elem = site.specie if isinstance(site.specie, Element) else site.specie.element
        props = [self.data_source.get_elemental_property(elem, p) for p in self.properties]

        return props

    def feature_labels(self):
        return ["site {}".format(p) for p in self.properties]

    def citations(self):
        return self._preset_citations

    def implementors(self):
        return ["Logan Ward"]

    @staticmethod
    def from_preset(preset):
        """Create the class with pre-defined settings

        Args:
            preset (string): Desired preset
        Returns:
            SiteElementalProperty initialized with desired settings
        """

        if preset == "seko-prb-2017":
            output = SiteElementalProperty(
                data_source=MagpieData(),
                properties=[
                    "Number",
                    "AtomicWeight",
                    "Row",
                    "Column",
                    "FirstIonizationEnergy",
                    "SecondIonizationEnergy",
                    "ElectronAffinity",
                    "Electronegativity",
                    "AllenElectronegativity",
                    "VdWRadius",
                    "CovalentRadius",
                    "AtomicRadius",
                    "ZungerPP-r_s",
                    "ZungerPP-r_p",
                    "MeltingT",
                    "BoilingT",
                    "Density",
                    "MolarVolume",
                    "HeatFusion",
                    "HeatVaporization",
                    "LogThermalConductivity",
                    "HeatCapacityMass",
                ],
            )
            output._preset_citations.append(
                "@article{Seko2017,"
                "author = {Seko, Atsuto and Hayashi, Hiroyuki and "
                "Nakayama, Keita and Takahashi, Akira and Tanaka, Isao},"
                "doi = {10.1103/PhysRevB.95.144110},"
                "journal = {Physical Review B}, number = {14},"
                "pages = {144110},"
                "title = {{Representation of compounds for machine-learning prediction of physical properties}},"
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.95.144110},"
                "volume = {95}, year = {2017}}"
            )
            return output
        else:
            raise ValueError("Unrecognized preset: {}".format(preset))
