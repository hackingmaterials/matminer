import warnings
from matminer.descriptors.composition_features import get_pymatgen_eldata_lst, get_std


class VolumePredictor:
    """
    Predicts volume; given a structure, finds the minimum volume such that
    no two sites are closer than a linear combination of their atomic and ionic radii.
    When run over all stable elemental and binary structures from MP, it is found that:
    (i) RMSE % error = 23.6 %
    (ii) Average percentage difference in volume from initial volume = 2.88 %
    (iii) Performs worst for materials that are gaseous at standard state conditions, eg: H2, N2,
        and f-electron systems, eg: Np, Pu, etc.
    """
    def __init__(self, cutoff=4, ionic_factor=0.30):
        """
        :param cutoff: (float) cutoff for site pairs (added to site radius)
                in Angstrom. Increase if your initial structure guess
                is extremely bad (atoms way too far apart). In all other cases,
                increasing cutoff gives same answer but at lower performance.
        :param ionic_factor: (float) Factor that accounts for ionicity in a bond.
                It determines the contribution of ionic and atomic radii of each element
                making up a bond to the sum of their radii.
        """
        self.cutoff = cutoff
        if ionic_factor > 0.40:
            raise ValueError("specified ionic factor is out of range!")
        self.ionic_factor = ionic_factor

    def predict(self, structure):
        """
        Given a structure, returns back the predicted volume.
        Volume is predicted based on minimum bond distance, which is determined using
        an ionic mix factor based on electronegativity spread in a structure.

        :param structure: pymatgen structure object
        :return: scaled pymatgen structure object
        """
        if not structure.is_ordered:
            raise ValueError("VolumePredictorSimple requires "
                             "ordered structures!")

        smallest_distance = None
        ionic_mix = get_std(get_pymatgen_eldata_lst(structure.composition, 'X')) * self.ionic_factor

        for site in structure:
            el1 = site.specie
            if el1.atomic_radius:
                x = structure.get_neighbors(site,
                                            el1.atomic_radius + self.cutoff)
                r1 = el1.average_ionic_radius * ionic_mix + el1.atomic_radius * (1-ionic_mix) if \
                    el1.average_ionic_radius else el1.atomic_radius

                for site2, dist in x:
                    el2 = site2.specie
                    if el2.atomic_radius:
                        r2 = el2.average_ionic_radius * ionic_mix + el2.atomic_radius * (1-ionic_mix) if \
                            el2.average_ionic_radius else el2.atomic_radius

                        expected_dist = float(r1 + r2)

                        if not smallest_distance or dist/expected_dist \
                                < smallest_distance:
                            smallest_distance = dist/expected_dist
                    else:
                        warnings.warn("VolumePredictor: no atomic radius data for {}".format(el2))
            else:
                warnings.warn("VolumePredictor: no atomic radius data for {}".format(el1))

        if not smallest_distance:
            raise ValueError("Could not find any bonds in this material!")

        volume_factor = (1/smallest_distance)**3

        return structure.volume * volume_factor

    def get_predicted_structure(self, structure):
        """
        Given a structure, returns back the structure scaled to predicted volume
        using the "predict" method.
        :param structure: pymatgen structure object
        :return: scaled pymatgen structure object
        """
        new_volume = self.predict(structure)
        new_structure = structure.copy()
        new_structure.scale_lattice(new_volume)

        return new_structure

