"""
Composition featurizers for elemental data and stoichiometry.
"""

from pymatgen.core import Element

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.data import (
    DemlData,
    MagpieData,
)


class ElementFraction(BaseFeaturizer):
    """
    Class to calculate the atomic fraction of each element in a composition.

    Generates a vector where each index represents an element in atomic number order.
    """

    def __init__(self):
        pass

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            vector (list of floats): fraction of each element in a composition
        """

        vector = [0] * 103
        el_list = list(comp.element_composition.fractional_composition.items())
        for el in el_list:
            obj = el
            atomic_number_i = obj[0].number - 1
            vector[atomic_number_i] = obj[1]
        return vector

    def feature_labels(self):
        labels = []
        for i in range(1, 104):
            labels.append(Element.from_Z(i).symbol)
        return labels

    def implementors(self):
        return ["Ashwin Aggarwal", "Logan Ward"]

    def citations(self):
        return []


class TMetalFraction(BaseFeaturizer):
    """
    Class to calculate fraction of magnetic transition metals in a composition.

    Parameters:
        data_source (data class): source from which to retrieve element data

    Generates: Fraction of magnetic transition metal atoms in a compound
    """

    def __init__(self):
        self.magn_elem = [
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
        ]

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            frac_magn_atoms (single-element list): fraction of magnetic transitional metal atoms in a compound
        """

        el_amt = comp.get_el_amt_dict()

        frac_magn_atoms = 0
        for el in el_amt:
            if el in self.magn_elem:
                frac_magn_atoms += el_amt[el]

        frac_magn_atoms /= sum(el_amt.values())

        return [frac_magn_atoms]

    def feature_labels(self):
        labels = ["transition metal fraction"]
        return labels

    def citations(self):
        citation = [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        ]
        return citation

    def implementors(self):
        return ["Jiming Chen, Logan Ward"]


class Stoichiometry(BaseFeaturizer):
    """
    Calculate norms of stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
        num_atoms (bool): whether to return number of atoms per formula unit
    """

    def __init__(self, p_list=(0, 2, 3, 5, 7, 10), num_atoms=False):
        self.p_list = p_list
        self.num_atoms = num_atoms

    def featurize(self, comp):
        """
        Get stoichiometric attributes
        Args:
            comp: Pymatgen composition object
            p_list (list of ints)

        Returns:
            p_norm (list of floats): Lp norm-based stoichiometric attributes.
                Returns number of atoms if no p-values specified.
        """

        el_amt = comp.get_el_amt_dict()

        # Compute the number of atoms per formula unit
        n_atoms_per_unit = comp.num_atoms / comp.get_integer_formula_and_factor()[1]

        if self.p_list is None:
            stoich_attr = [n_atoms_per_unit]  # return num atoms if no norms specified
        else:
            p_norms = [0] * len(self.p_list)
            n_atoms = sum(el_amt.values())

            for i in range(len(self.p_list)):
                if self.p_list[i] < 0:
                    raise ValueError("p-norm not defined for p < 0")
                if self.p_list[i] == 0:
                    p_norms[i] = len(el_amt.values())
                else:
                    for j in el_amt:
                        p_norms[i] += (el_amt[j] / n_atoms) ** self.p_list[i]
                    p_norms[i] = p_norms[i] ** (1.0 / self.p_list[i])

            if self.num_atoms:
                stoich_attr = [n_atoms_per_unit] + p_norms
            else:
                stoich_attr = p_norms

        return stoich_attr

    def feature_labels(self):
        labels = []
        if self.num_atoms:
            labels.append("num atoms")

        if self.p_list != None:
            for p in self.p_list:
                labels.append("%d-norm" % p)

        return labels

    def citations(self):
        citation = [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}"
        ]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class BandCenter(BaseFeaturizer):
    """
    Estimation of absolute position of band center using electronegativity.

    Features
        - Band center
    """

    def featurize(self, comp):
        """
        (Rough) estimation of absolution position of band center using
        geometric mean of electronegativity.

        Args:
            comp (Composition).

        Returns:
            (float) band center.

        """
        gmean = 1.0
        sumamt = sum(comp.get_el_amt_dict().values())
        for el, amt in comp.get_el_amt_dict().items():
            first_ioniz = DemlData().get_elemental_property(Element(el), "first_ioniz") / 1000
            elec_aff = MagpieData().get_elemental_property(Element(el), "ElectronAffinity")
            gmean *= (0.5 * (first_ioniz + elec_aff) / 96.48) ** (amt / sumamt)
        return [gmean]

    def feature_labels(self):
        return ["band center"]

    def citations(self):
        return [
            "@article{Butler1978, author = {Butler, M A and Ginley, D S}, "
            "doi = {10.1149/1.2131419}, isbn = {0013-4651}, issn = {00134651}, "
            "journal = {Journal of The Electrochemical Society}, month = {feb},"
            " number = {2}, pages = {228--232}, title = {{Prediction of "
            "Flatband Potentials at Semiconductor-Electrolyte Interfaces from "
            "Atomic Electronegativities}}, url = "
            "{http://jes.ecsdl.org/content/125/2/228}, volume = {125}, "
            "year = {1978} } "
        ]

    def implementors(self):
        return ["Anubhav Jain"]
