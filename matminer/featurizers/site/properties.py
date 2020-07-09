from __future__ import division

import numpy as np

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.caching import get_nearest_neighbors
from matminer.utils.data import MagpieData


class SiteElementalProperty(BaseFeaturizer):
    """
    Elemental properties of atom on a certain site

    Features:
        site [property] - Elemental property for this site

    References:
        `Seko et al., _PRB_ (2017) <http://link.aps.org/doi/10.1103/PhysRevB.95.144110>`_
        `Schmidt et al., _Chem Mater_. (2017) <http://dx.doi.org/10.1021/acs.chemmater.7b00156>`_
    """

    def __init__(self, data_source=None, properties=('Number',)):
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
        return ['site {}'.format(p) for p in self.properties]

    def citations(self):
        return self._preset_citations

    def implementors(self):
        return ['Logan Ward']

    @staticmethod
    def from_preset(preset):
        """Create the class with pre-defined settings

        Args:
            preset (string): Desired preset
        Returns:
            SiteElementalProperty initialized with desired settings
        """

        if preset == "seko-prb-2017":
            output = SiteElementalProperty(data_source=MagpieData(),
                                           properties=["Number", "AtomicWeight", "Row", "Column",
                                                       "FirstIonizationEnergy",
                                                       "SecondIonizationEnergy",
                                                       "ElectronAffinity",
                                                       "Electronegativity",
                                                       "AllenElectronegativity",
                                                       "VdWRadius", "CovalentRadius",
                                                       "AtomicRadius",
                                                       "ZungerPP-r_s", "ZungerPP-r_p",
                                                       "MeltingT", "BoilingT", "Density",
                                                       "MolarVolume", "HeatFusion",
                                                       "HeatVaporization",
                                                       "LogThermalConductivity", "HeatCapacityMass"
                                                       ])
            output._preset_citations.append("@article{Seko2017,"
                                            "author = {Seko, Atsuto and Hayashi, Hiroyuki and "
                                            "Nakayama, Keita and Takahashi, Akira and Tanaka, Isao},"
                                            "doi = {10.1103/PhysRevB.95.144110},"
                                            "journal = {Physical Review B}, number = {14},"
                                            "pages = {144110},"
                                            "title = {{Representation of compounds for machine-learning prediction of physical properties}},"
                                            "url = {http://link.aps.org/doi/10.1103/PhysRevB.95.144110},"
                                            "volume = {95}, year = {2017}}")
            return output
        else:
            raise ValueError('Unrecognized preset: {}'.format(preset))


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

    def __init__(self, data_source=MagpieData(), weight='area',
                 properties=('Electronegativity',), signed=False):
        """ Initialize the featurizer

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
                properties=["Number", "MendeleevNumber", "AtomicWeight",
                            "MeltingT", "Column", "Row", "CovalentRadius",
                            "Electronegativity", "NsValence", "NpValence",
                            "NdValence", "NfValence", "NValence", "NsUnfilled",
                            "NpUnfilled", "NdUnfilled", "NfUnfilled",
                            "NUnfilled", "GSvolume_pa", "GSbandgap",
                            "GSmagmom", "SpaceGroupNumber"]
            )
        else:
            raise ValueError('Unrecognized preset: ' + preset)

    def featurize(self, strc, idx):
        # Get the targeted site
        my_site = strc[idx]

        # Get the tessellation of a site
        nn = get_nearest_neighbors(VoronoiNN(weight=self.weight), strc, idx)

        # Get the element and weight of each site
        elems = [n['site'].specie for n in nn]
        weights = [n['weight'] for n in nn]

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
            return ['local difference in ' + p for p in self.properties]
        else:
            return ['local signed difference in ' + p for p in self.properties]

    def citations(self):
        return ["@article{Ward2017,"
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

                '@article{jong_chen_notestine_persson_ceder_jain_asta_gamst_2016,'
                'title={A Statistical Learning Framework for Materials Science: '
                'Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds}, '
                'volume={6}, DOI={10.1038/srep34256}, number={1}, journal={Scientific Reports}, '
                'author={Jong, Maarten De and Chen, Wei and Notestine, Randy and Persson, '
                'Kristin and Ceder, Gerbrand and Jain, Anubhav and Asta, Mark and Gamst, Anthony}, '
                'year={2016}, month={Mar}}'
                ]

    def implementors(self):
        return ['Logan Ward', 'Aik Rui Tan']
