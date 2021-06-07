"""
Composition featurizers for thermodynamic properties.
"""

from pymatgen.ext.matproj import MPRester

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.data import (
    CohesiveEnergyData,
)


class CohesiveEnergy(BaseFeaturizer):
    """
    Cohesive energy per atom using elemental cohesive energies and
    formation energy.

    Get cohesive energy per atom of a compound by adding known
    elemental cohesive energies from the formation energy of the
    compound.

    Parameters:
        mapi_key (str): Materials API key for looking up formation energy
            by composition alone (if you don't set the formation energy
            yourself).
    """

    def __init__(self, mapi_key=None):
        self.mapi_key = mapi_key

    def featurize(self, comp, formation_energy_per_atom=None):
        """
        Args:
            comp: (pymatgen.Composition): A composition
            formation_energy_per_atom: (float) the formation energy per atom of
                your compound. If not set, will look up the most stable
                formation energy from the Materials Project database.
        """
        comp = comp.reduced_composition
        el_amt_dict = comp.get_el_amt_dict()

        formation_energy_per_atom = formation_energy_per_atom or None

        if not formation_energy_per_atom:
            # Get formation energy of most stable structure from MP
            struct_lst = MPRester(self.mapi_key).get_data(comp.reduced_formula)
            if len(struct_lst) > 0:
                most_stable_entry = sorted(struct_lst, key=lambda e: e["energy_per_atom"])[0]
                formation_energy_per_atom = most_stable_entry["formation_energy_per_atom"]
            else:
                raise ValueError("No structure found in MP for {}".format(comp))

        # Subtract elemental cohesive energies from formation energy
        cohesive_energy = -formation_energy_per_atom * comp.num_atoms
        for el in el_amt_dict:
            cohesive_energy += el_amt_dict[el] * CohesiveEnergyData().get_elemental_property(el)

        cohesive_energy_per_atom = cohesive_energy / comp.num_atoms

        return [cohesive_energy_per_atom]

    def feature_labels(self):
        return ["cohesive energy"]

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]

    def citations(self):
        # Cohesive energy values for the elements are taken from the
        # Knowledgedoor web site, which obtained those values from Kittel.
        # We include both citations.
        return [
            "@misc{, title = {{Knowledgedoor Cohesive energy handbook}}, "
            "url = {http://www.knowledgedoor.com/2/elements{\_}handbook/cohesive{\_}energy.html}}",
            "@book{Kittel, author = {Kittel, C}, isbn = {978-0-471-41526-8}, "
            "publisher = {Wiley}, title = {{Introduction to Solid State "
            "Physics, 8th Edition}}, year = {2005}}",
        ]


class CohesiveEnergyMP(BaseFeaturizer):
    """
    Cohesive energy per atom lookup using Materials Project

    Parameters:
        mapi_key (str): Materials API key for looking up cohesive energy
            by composition alone.
    """

    def __init__(self, mapi_key=None):
        self.mapi_key = mapi_key

    def featurize(self, comp):
        """

        Args:
            comp: (str) compound composition, eg: "NaCl"
        """

        # Get formation energy of most stable structure from MP
        with MPRester(self.mapi_key) as mpr:
            struct_lst = mpr.get_data(comp.reduced_formula)
            if len(struct_lst) > 0:
                most_stable_entry = sorted(struct_lst, key=lambda e: e["energy_per_atom"])[0]
                try:
                    return [mpr.get_cohesive_energy(most_stable_entry["material_id"], per_atom=True)]
                except:
                    raise ValueError(
                        "No cohesive energy can be determined for material_id: {}".format(
                            most_stable_entry["material_id"]
                        )
                    )
            else:
                raise ValueError("No structure found in MP for {}".format(comp))

    def feature_labels(self):
        return ["cohesive energy (MP)"]

    def implementors(self):
        return ["Anubhav Jain"]

    def citations(self):
        return [
            "@article{doi:10.1063/1.4812323, author = {Jain,Anubhav and Ong,"
            "Shyue Ping  and Hautier,Geoffroy and Chen,Wei and Richards, "
            "William Davidson  and Dacek,Stephen and Cholia,Shreyas "
            "and Gunter,Dan  and Skinner,David and Ceder,Gerbrand "
            "and Persson,Kristin A. }, title = {Commentary: The Materials "
            "Project: A materials genome approach to accelerating materials "
            "innovation}, journal = {APL Materials}, volume = {1}, number = "
            "{1}, pages = {011002}, year = {2013}, doi = {10.1063/1.4812323}, "
            "URL = {https://doi.org/10.1063/1.4812323}, "
            "eprint = {https://doi.org/10.1063/1.4812323}}",
            "@article{Ong2015, author = {Ong, Shyue Ping and Cholia, "
            "Shreyas and Jain, Anubhav and Brafman, Miriam and Gunter, Dan "
            "and Ceder, Gerbrand and Persson, Kristin a.}, doi = "
            "{10.1016/j.commatsci.2014.10.037}, issn = {09270256}, "
            "journal = {Computational Materials Science}, month = {feb}, "
            "pages = {209--215}, publisher = {Elsevier B.V.}, title = "
            "{{The Materials Application Programming Interface (API): A simple, "
            "flexible and efficient API for materials data based on "
            "REpresentational State Transfer (REST) principles}}, "
            "url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025614007113}, "
            "volume = {97}, year = {2015} } ",
        ]
