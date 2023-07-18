"""
Composition featurizers for orbital data.
"""

import collections
from warnings import warn

from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import MagpieData


class AtomicOrbitals(BaseFeaturizer):
    """
    Determine HOMO/LUMO features based on a composition.

    The highest occupied molecular orbital (HOMO) and lowest unoccupied
    molecular orbital (LUMO) are estiated from the atomic orbital energies
    of the composition. The atomic orbital energies are from NIST:
    https://www.nist.gov/pml/data/atomic-reference-data-electronic-structure-calculations

    Warning:
    For compositions with inter-species fractions greater than 10,000 (e.g.
    dilute alloys such as FeC0.00001) the composition will be truncated (to Fe
    in this example). In such extreme cases, the truncation likely reflects the
    true physics of the situation (i.e. that the dilute element does not
    significantly contribute orbital character to the band structure), but the
    user should be aware of this behavior.
    """

    def featurize(self, comp):
        """
        Args:
            comp: (Composition)
                pymatgen Composition object

        Returns:
            HOMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
            HOMO_element: (str) symbol of element for HOMO
            HOMO_energy: (float in eV) absolute energy of HOMO
            LUMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
            LUMO_element: (str) symbol of element for LUMO
            LUMO_energy: (float in eV) absolute energy of LUMO
            gap_AO: (float in eV)
                the estimated bandgap from HOMO and LUMO energeis
        """

        integer_comp, factor = comp.get_integer_formula_and_factor()

        # warning message if composition is dilute and truncated
        if not len(Composition(comp).elements) == len(Composition(integer_comp).elements):
            warn(f"AtomicOrbitals: {comp} truncated to {integer_comp}")

        homo_lumo = MolecularOrbitals(integer_comp).band_edges

        feat = collections.OrderedDict()
        for edge in ["HOMO", "LUMO"]:
            feat[f"{edge}_character"] = homo_lumo[edge][1][-1]
            feat[f"{edge}_element"] = homo_lumo[edge][0]
            feat[f"{edge}_energy"] = homo_lumo[edge][2]
        feat["gap_AO"] = feat["LUMO_energy"] - feat["HOMO_energy"]

        return list(feat.values())

    def feature_labels(self):
        feat = []
        for edge in ["HOMO", "LUMO"]:
            feat.extend(
                [
                    f"{edge}_character",
                    f"{edge}_element",
                    f"{edge}_energy",
                ]
            )
        feat.append("gap_AO")
        return feat

    def citations(self):
        return [
            "@article{PhysRevA.55.191,"
            "title = {Local-density-functional calculations of the energy of atoms},"
            "author = {Kotochigova, Svetlana and Levine, Zachary H. and Shirley, "
            "Eric L. and Stiles, M. D. and Clark, Charles W.},"
            "journal = {Phys. Rev. A}, volume = {55}, issue = {1}, pages = {191--199},"
            "year = {1997}, month = {Jan}, publisher = {American Physical Society},"
            "doi = {10.1103/PhysRevA.55.191}, "
            "url = {https://link.aps.org/doi/10.1103/PhysRevA.55.191}}"
        ]

    def implementors(self):
        return ["Maxwell Dylla", "Anubhav Jain"]


class ValenceOrbital(BaseFeaturizer):
    """
    Attributes of valence orbital shells

    Args:
        data_source (data object): source from which to retrieve element data
        orbitals (list): orbitals to calculate
        props (list): specifies whether to return average number of electrons in each orbital,
            fraction of electrons in each orbital, or both
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, orbitals=("s", "p", "d", "f"), props=("avg", "frac"), impute_nan=True):
        self.impute_nan = impute_nan
        self.data_source = MagpieData(impute_nan=self.impute_nan)
        self.orbitals = orbitals
        self.props = props

    def featurize(self, comp):
        """Weighted fraction of valence electrons in each orbital

        Args:
             comp: Pymatgen composition object

        Returns:
             valence_attributes (list of floats): Average number and/or
                 fraction of valence electrons in specified orbitals
        """

        elements, fractions = zip(*comp.element_composition.items())

        # Get the mean number of electrons in each shell
        avg = [
            PropertyStats.mean(
                self.data_source.get_elemental_properties(elements, f"N{orb}Valence"),
                weights=fractions,
            )
            for orb in self.orbitals
        ]

        # If needed, get fraction of electrons in each shell
        if "frac" in self.props:
            avg_total_valence = PropertyStats.mean(
                self.data_source.get_elemental_properties(elements, "NValence"),
                weights=fractions,
            )
            frac = [a / avg_total_valence for a in avg]

        # Get the desired attributes
        valence_attributes = []
        for prop in self.props:
            valence_attributes += locals()[prop]

        return valence_attributes

    def feature_labels(self):
        labels = []
        for prop in self.props:
            for orb in self.orbitals:
                labels.append(f"{prop} {orb} valence electrons")

        return labels

    def citations(self):
        ward_citation = (
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}"
        )
        deml_citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        )
        citations = [ward_citation, deml_citation]
        return citations

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]
