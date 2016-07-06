import warnings
from matminer.descriptors.composition_features import get_pymatgen_eldata_lst, get_std


class VolumePredictor:
    def __init__(self, cutoff=4, ionic_factor=0.125):
        """
        Predicts volume; given a structure, finds the minimum volume such that
        no two sites are closer than sum of their atomic radii. The sum of
        atomic radii is modified for ionicity using an ionic_factor that
        depends on electronegativity difference.
        Args:
            cutoff: (float) cutoff for site pairs (added to site radius)
                in Angstrom. Increase if your initial structure guess
                is extremely bad (atoms way too far apart). In all other cases,
                increasing cutoff gives same answer but at lower performance.
        """
        self.cutoff = cutoff
        if ionic_factor > 0.25:
            raise ValueError("specified ionic factor is out of range!")
        self.ionic_factor = ionic_factor

    def predict(self, structure):
        """
        Given a structure, returns back a volume
        Args:
            structure: (Structure)
        Returns:
            (float) expected volume of structure
        """

        if not structure.is_ordered:
            raise ValueError("VolumePredictorSimple requires "
                             "ordered structures!")

        smallest_distance = None
        ionic_mix = get_std(get_pymatgen_eldata_lst(structure.composition, 'X')) * 0.50
        # ionic_mix = abs(el1.X - el2.X) * self.ionic_factor

        for site in structure:
            el1 = site.specie
            if el1.atomic_radius:
                x = structure.get_neighbors(site,
                                            el1.atomic_radius + self.cutoff)

                for site2, dist in x:
                    el2 = site2.specie
                    if el2.atomic_radius:
                        r1 = el1.average_ionic_radius * ionic_mix +\
                             el1.atomic_radius * (1-ionic_mix) if \
                            el1.average_ionic_radius else el1.atomic_radius

                        r2 = el2.average_ionic_radius * ionic_mix +\
                             el2.atomic_radius * (1-ionic_mix) if \
                            el2.average_ionic_radius else el2.atomic_radius

                        expected_dist = float(r1 + r2)

                        if not smallest_distance or dist/expected_dist \
                                < smallest_distance:
                            smallest_distance = dist/expected_dist
                    else:
                        warnings.warn("VolumePredictor: no atomic radius data for {}".
                              format(el2))

            else:
                warnings.warn("VolumePredictor: no atomic radius data for {}".
                              format(el1))

        if not smallest_distance:
            raise ValueError("Could not find any bonds in this material!")

        return structure.volume * (1/smallest_distance)**3








