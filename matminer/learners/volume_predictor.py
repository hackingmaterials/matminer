import warnings
from matminer.descriptors.composition_features import get_element_data, get_std
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.bond_valence import BVAnalyzer


class VolumePredictor:
    """
    Predicts volume; given a structure, finds the minimum volume such that no
    two sites are closer than a weighted sum of their atomic and ionic radii.
    When run over all stable elemental and binary structures from MP,
    it is found that:
    (i) RMSE % error = 23.6 %
    (ii) Average percentage difference in volume from initial volume = 2.88 %
    (iii) Average absolute difference in volume from initial volume = 17.0 %
    (iv) Performs worst for materials that are gaseous eg: H2, N2,
        and f-electron systems, eg: Np, Pu, etc. as well as noble gas compounds
    This is really intended for bonded systems!
    """
    def __init__(self, cutoff=4, ionic_factor=0.30):
        """
        Args:
            cutoff (float): cutoff radius added to site radius for finding
                site pairs. Necessary to increase only if your initial
                structure guess is extremely bad (atoms way too far apart). In
                all other instances, increasing cutoff gives same answer
                but takes more time.
            ionic_factor: (float) This factor, multiplied by the compound's
                spread in electronegativity, determines the weighting between
                ionic and atomic radii for expected bond distances.
        """

        self.cutoff = cutoff
        self.ionic_factor = ionic_factor

    def predict(self, structure):
        """
        Given a structure, returns back the predicted volume.

        Args:
            structure (Structure): structure w/unknown volume

        Returns:
            a float value of the predicted volume
        """

        if not structure.is_ordered:
            raise ValueError("VolumePredictor requires ordered structures!")

        smallest_ratio = None  # ratio of observed vs expected bond distance
        ionic_mix = min(get_std(get_element_data(structure.composition,
                                                 'X')) * self.ionic_factor, 1)

        for site in structure:
            el1 = site.specie
            if el1.atomic_radius:
                r1 = el1.average_ionic_radius * ionic_mix + \
                     el1.atomic_radius * (1-ionic_mix) if \
                    el1.average_ionic_radius else el1.atomic_radius

                neighbors = structure.get_neighbors(site,
                                            el1.atomic_radius + self.cutoff)

                for site2, dist in neighbors:
                    el2 = site2.specie
                    if el2.atomic_radius:
                        r2 = el2.average_ionic_radius * ionic_mix + \
                             el2.atomic_radius * (1-ionic_mix) if \
                            el2.average_ionic_radius else el2.atomic_radius

                        expected_dist = float(r1 + r2)

                        if not smallest_ratio or dist/expected_dist \
                                < smallest_ratio:
                            smallest_ratio = dist/expected_dist
                    else:
                        warnings.warn("VolumePredictor: no atomic radius data "
                                      "for {}".format(el2))
            else:
                warnings.warn("VolumePredictor: no atomic radius data for "
                              "{}".format(el1))

        if not smallest_ratio:
            raise ValueError("Could not find any bonds in this material!")

        volume_factor = (1/smallest_ratio)**3

        return structure.volume * volume_factor

    def get_predicted_structure(self, structure):
        """
        Given a structure, returns back the structure scaled to predicted
        volume.

        Args:
            structure (Structure): structure w/unknown volume

        Returns:
            a Structure object with predicted volume
        """
        new_structure = structure.copy()
        new_structure.scale_lattice(self.predict(structure))

        return new_structure


def is_ox(structure):
    comp = structure.composition
    for k in comp.keys():
        try:
            k.oxi_state
        except AttributeError:
            return False
    return True


class ConditionalVolumePredictor:
    """
    Unlike the above, the idea here is to predict the volume of a structure
    based on an existing known structure. May integrate with above at a later
    stage to improve overall predictions.
    """
    def __init__(self):
        pass

    def predict(self, structure, ref_structure):
        """
        Given a structure, returns back the predicted volume.

        Args:
            structure (Structure): structure w/unknown volume
            ref_structure (Structure): A reference structure with a similar
                structure but different species.

        Returns:
            a float value of the predicted volume
        """
        if not is_ox(structure):
            a = BVAnalyzer()
            structure = a.get_oxi_state_decorated_structure(structure)
        if not is_ox(ref_structure):
            a = BVAnalyzer()
            ref_structure = a.get_oxi_state_decorated_structure(ref_structure)

        m = StructureMatcher()
        mapping = m.get_best_electronegativity_anonymous_mapping(structure,
                                                                 ref_structure)
        if mapping is None:
            raise ValueError("Input structures do not match!")

        comp = structure.composition
        ref_comp = ref_structure.composition

        numerator = 0
        denominator = 0

        for k, v in mapping.items():
            numerator += k.ionic_radius ** 3 * comp[k]
            denominator += v.ionic_radius ** 3 * ref_comp[v]

        return ref_structure.volume * numerator / denominator

    def get_predicted_structure(self, structure, ref_structure):
        """
        Given a structure, returns back the structure scaled to predicted
        volume.

        Args:
            structure (Structure): structure w/unknown volume
            ref_structure (Structure): A reference structure with a similar
                structure but different species.

        Returns:
            a Structure object with predicted volume
        """
        new_structure = structure.copy()
        new_structure.scale_lattice(self.predict(structure, ref_structure))

        return new_structure
