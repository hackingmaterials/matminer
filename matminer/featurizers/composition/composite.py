"""
Composition featurizers for composite features containing more than 1 category of general-purpose data.
"""

from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.orbital import ValenceOrbital
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import (
    DemlData,
    MagpieData,
    PymatgenData,
    MatscholarElementData,
    MEGNetElementData,
)


class ElementProperty(BaseFeaturizer):
    """
    Class to calculate elemental property attributes.

    To initialize quickly, use the from_preset() method.

    Features: Based on the statistics of the data_source chosen, computed
    by element stoichiometry. The format generally is:

    "{data source} {statistic} {property}"

    For example:

    "PymetgenData range X"  # Range of electronegativity from Pymatgen data

    For a list of all statistics, see the PropertyStats documentation; for a
    list of all attributes available for a given data_source, see the
    documentation for the data sources (e.g., PymatgenData, MagpieData,
    MatscholarElementData, etc.).

    Args:
        data_source (AbstractData or str): source from which to retrieve
            element property data (or use str for preset: "pymatgen",
            "magpie", or "deml")
        features (list of strings): List of elemental properties to use
            (these must be supported by data_source)
        stats (list of strings): a list of weighted statistics to compute to for each
            property (see PropertyStats for available stats)
    """

    def __init__(self, data_source, features, stats):
        if data_source == "pymatgen":
            self.data_source = PymatgenData()
        elif data_source == "magpie":
            self.data_source = MagpieData()
        elif data_source == "deml":
            self.data_source = DemlData()
        elif data_source == "matscholar_el":
            self.data_source = MatscholarElementData()
        elif data_source == "megnet_el":
            self.data_source = MEGNetElementData()
        else:
            self.data_source = data_source

        self.features = features
        self.stats = stats
        # Initialize stats computer
        self.pstats = PropertyStats()

    @classmethod
    def from_preset(cls, preset_name):
        """
        Return ElementProperty from a preset string
        Args:
            preset_name: (str) can be one of "magpie", "deml", "matminer",
                "matscholar_el", or "megnet_el".

        Returns:
            ElementProperty based on the preset name.
        """
        if preset_name == "magpie":
            data_source = "magpie"
            features = [
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
            ]
            stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]

        elif preset_name == "deml":
            data_source = "deml"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = [
                "atom_num",
                "atom_mass",
                "row_num",
                "col_num",
                "atom_radius",
                "molar_vol",
                "heat_fusion",
                "melting_point",
                "boiling_point",
                "heat_cap",
                "first_ioniz",
                "electronegativity",
                "electric_pol",
                "GGAU_Etot",
                "mus_fere",
                "FERE correction",
            ]

        elif preset_name == "matminer":
            data_source = "pymatgen"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = [
                "X",
                "row",
                "group",
                "block",
                "atomic_mass",
                "atomic_radius",
                "mendeleev_no",
                "electrical_resistivity",
                "velocity_of_sound",
                "thermal_conductivity",
                "melting_point",
                "bulk_modulus",
                "coefficient_of_linear_thermal_expansion",
            ]

        elif preset_name == "matscholar_el":
            data_source = "matscholar_el"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = MatscholarElementData().prop_names

        elif preset_name == "megnet_el":
            data_source = "megnet_el"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = MEGNetElementData().prop_names

        else:
            raise ValueError("Invalid preset_name specified!")

        return cls(data_source, features, stats)

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
        """

        all_attributes = []

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.items())

        for attr in self.features:
            elem_data = [self.data_source.get_elemental_property(e, attr) for e in elements]

            for stat in self.stats:
                all_attributes.append(self.pstats.calc_stat(elem_data, stat, fractions))

        return all_attributes

    def feature_labels(self):
        labels = []
        for attr in self.features:
            src = self.data_source.__class__.__name__
            for stat in self.stats:
                labels.append("{} {} {}".format(src, stat, attr))
        return labels

    def citations(self):
        if self.data_source.__class__.__name__ == "MagpieData":
            citation = [
                "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
                "machine learning framework for predicting properties of inorganic materials}, "
                "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
                "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
                "Alok and Wolverton, Christopher}, year={2016}}"
            ]
        elif self.data_source.__class__.__name__ == "DemlData":
            citation = [
                "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
                "functional theory total energies and enthalpies of formation of metal-nonmetal "
                "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
                "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
                "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
            ]
        elif self.data_source.__class__.__name__ == "PymatgenData":
            citation = [
                "@article{Ong2013, author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, "
                "Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, "
                "Kristin A. and Ceder, Gerbrand}, doi = {10.1016/j.commatsci.2012.10.028}, issn = {09270256}, "
                "journal = {Computational Materials Science}, month = {feb}, pages = {314--319}, "
                "publisher = {Elsevier B.V.}, title = {{Python Materials Genomics (pymatgen): A robust, open-source python "
                "library for materials analysis}}, url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295}, "
                "volume = {68}, year = {2013} } "
            ]
        elif self.data_source.__class__.__name__ == "MEGNetElementData":
            # TODO: Cite MEGNet publication (not preprint) once released!
            citation = [
                "@ARTICLE{2018arXiv181205055C,"
                "author = {{Chen}, Chi and {Ye}, Weike and {Zuo}, Yunxing and {Zheng}, Chen and {Ong}, Shyue Ping},"
                "title = '{Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals}',"
                "journal = {arXiv e-prints},"
                "keywords = {Condensed Matter - Materials Science, Physics - Computational Physics},"
                "year = '2018',"
                "month = 'Dec',"
                "eid = {arXiv:1812.05055},"
                "pages = {arXiv:1812.05055},"
                "archivePrefix = {arXiv},"
                "eprint = {1812.05055},"
                "primaryClass = {cond-mat.mtrl-sci},"
                "adsurl = {https://ui.adsabs.harvard.edu/\#abs/2018arXiv181205055C},"
                "adsnote = {Provided by the SAO/NASA Astrophysics Data System}}"
            ]
        else:
            citation = []
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward", "Anubhav Jain", "Alex Dunn"]


class Meredig(BaseFeaturizer):
    """
    Class to calculate features as defined in Meredig et. al.

    Features:
        Atomic fraction of each of the first 103 elements, in order of atomic number.
        17 statistics of elemental properties;
            Mean atomic weight of constituent elements
            Mean periodic table row and column number
            Mean and range of atomic number
            Mean and range of atomic radius
            Mean and range of electronegativity
            Mean number of valence electrons in each orbital
            Fraction of total valence electrons in each orbital

    """

    def __init__(self):
        self.data_source = MagpieData()

        # The labels for statistics on element properties
        self._element_property_feature_labels = [
            "mean AtomicWeight",
            "mean Column",
            "mean Row",
            "range Number",
            "mean Number",
            "range AtomicRadius",
            "mean AtomicRadius",
            "range Electronegativity",
            "mean Electronegativity",
        ]
        # Initialize stats computer
        self.pstats = PropertyStats()

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
        """

        # First 103 features are element fractions, we can get these from the ElementFraction featurizer
        element_fraction_features = ElementFraction().featurize(comp)

        # Next 9 features are statistics on elemental properties
        elements, fractions = zip(*comp.element_composition.items())
        element_property_features = [0] * len(self._element_property_feature_labels)

        for i, feat in enumerate(self._element_property_feature_labels):
            stat = feat.split(" ")[0]
            attr = " ".join(feat.split(" ")[1:])

            elem_data = [self.data_source.get_elemental_property(e, attr) for e in elements]
            element_property_features[i] = self.pstats.calc_stat(elem_data, stat, fractions)

        # Final 8 features are statistics on valence orbitals, available from the ValenceOrbital featurizer
        valence_orbital_features = ValenceOrbital(orbitals=("s", "p", "d", "f"), props=("avg", "frac")).featurize(comp)

        return element_fraction_features + element_property_features + valence_orbital_features

    def feature_labels(self):
        # Since we have more features than just element fractions, append 'fraction' to element symbols for clarity
        element_fraction_features = [e + " fraction" for e in ElementFraction().feature_labels()]
        valence_orbital_features = ValenceOrbital().feature_labels()
        return element_fraction_features + self._element_property_feature_labels + valence_orbital_features

    def citations(self):
        citation = [
            "@article{meredig_agrawal_kirklin_saal_doak_thompson_zhang_choudhary_wolverton_2014, title={Combinatorial "
            "screening for new materials in unconstrained composition space with machine learning}, "
            "volume={89}, DOI={10.1103/PhysRevB.89.094104}, number={1}, journal={Physical "
            "Review B}, author={B. Meredig, A. Agrawal, S. Kirklin, J. E. Saal, J. W. Doak, A. Thompson, "
            "K. Zhang, A. Choudhary, and C. Wolverton}, year={2014}}"
        ]
        return citation

    def implementors(self):
        return ["Amalie Trewartha"]
