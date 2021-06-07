"""
Miscellaneous site featurizers.
"""
import numpy as np

from matminer.featurizers.base import BaseFeaturizer
from scipy.spatial import ConvexHull
from pymatgen.core import Structure
from pymatgen.analysis.local_env import (
    LocalStructOrderParams,
    VoronoiNN,
    solid_angle,
    vol_tetra,
)
import pymatgen.analysis.local_env

from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_nearest_neighbors
from matminer.utils.data import MagpieData


class IntersticeDistribution(BaseFeaturizer):
    """
    Interstice distribution in the neighboring cluster around an atom site.

    The interstices are categorized to distance, area and volume interstices.
    Each of these metrics is a measures of the relative amount of empty space
    around each atom as determined using atomic sphere models. The distance
    interstice is the fraction of a bonding line unoccupied by the atom spheres;
    The area interstice is the unoccupied area within the triangulated surface
    formed by atom triplets in convex hull formed by neighbors, and the volume
    interstice is the unoccupied portion of a tetrahedron formed between the
    central atom and neighbor atom triplets. Please refer to the original paper
    for more details (Wang et al. Nat Commun 10, 5537 (2019))

    For amorphous alloys (metallic glasses), the coordination environments are
    anisotropic, which can be reflected in the inequality of the interstices
    present around an atom. To describe the anisotropy, here we derive statistics
    of the interstices to featurize the interstice distribution around the atom.
    Other methods can be grouping the interstices into histogram grids of fixed
    bins and the features are then a vector of the values of the histograms.

    User note:
    This class is particularly designed for featuring the site-specific packing
    heterogeneity in metallic glasses, especially the all-metallic-element ones.
    If non-metallic-elements are present in the structures, the interstice
    estimates may have larger deviation from actual values (despite this
    deviation is systematic and thus the interstice estimates can still be
    used to represent the packing heterogeneity).

    Args:
        cutoff (float): cutoff distance in determining the potential
            neighbors for Voronoi tessellation analysis. (default: 6.5)
        interstice_types (str or [str]): interstice distribution types,
            support sub-list of ['dist', 'area', 'vol'].
        stats ([str]): statistics of distance/area/volume interstices.
        radius_type (str): source of radius estimate. (default: "MiracleRadius")
    """

    def __init__(self, cutoff=6.5, interstice_types=None, stats=None, radius_type="MiracleRadius"):
        self.cutoff = cutoff
        self.interstice_types = ["dist", "area", "vol"] if interstice_types is None else interstice_types
        if isinstance(self.interstice_types, str):
            self.interstice_types = [self.interstice_types]
        if all(t not in self.interstice_types for t in ["dist", "area", "vol"]):
            raise ValueError("interstice_types only support sub-list of " "['dist', 'area', 'vol']")
        self.stats = ["mean", "std_dev", "minimum", "maximum"] if stats is None else stats
        self.radius_type = radius_type

    def featurize(self, struct, idx):
        """
        Get interstice distribution fingerprints of site with given index in
        input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            interstice_fps ([float]): Interstice distribution fingerprints.
        """
        interstice_fps = list()

        # Get the nearest neighbors using Voronoi tessellation
        n_w = VoronoiNN(cutoff=self.cutoff).get_voronoi_polyhedra(struct, idx).values()

        nn_coords = np.array([nn["site"].coords for nn in n_w])

        # Get center atom's radius and its nearest neighbors' radii
        center_r = MagpieData().get_elemental_properties([struct[idx].specie], self.radius_type)[0] / 100
        nn_els = [nn["site"].specie for nn in n_w]
        nn_rs = np.array(MagpieData().get_elemental_properties(nn_els, self.radius_type)) / 100

        # Get indices of atoms forming the simplices of convex hull
        convex_hull_simplices = ConvexHull(nn_coords).simplices

        if "dist" in self.interstice_types:
            nn_dists = [nn["face_dist"] * 2 for nn in n_w]
            interstice_dist_list = IntersticeDistribution.analyze_dist_interstices(center_r, nn_rs, nn_dists)
            interstice_fps += [PropertyStats().calc_stat(interstice_dist_list, stat) for stat in self.stats]

        if "area" in self.interstice_types:
            interstice_area_list = IntersticeDistribution.analyze_area_interstice(
                nn_coords, nn_rs, convex_hull_simplices
            )
            interstice_fps += [PropertyStats().calc_stat(interstice_area_list, stat) for stat in self.stats]

        if "vol" in self.interstice_types:
            interstice_vol_list = IntersticeDistribution.analyze_vol_interstice(
                struct[idx].coords, nn_coords, center_r, nn_rs, convex_hull_simplices
            )
            interstice_fps += [PropertyStats().calc_stat(interstice_vol_list, stat) for stat in self.stats]
        return interstice_fps

    @staticmethod
    def analyze_dist_interstices(center_r, nn_rs, nn_dists):
        """Analyze the distance interstices between center atom and neighbors.
        Args:
            center_r (float): central atom's radius.
            nn_rs ([float]): Nearest Neighbors' radii.
            nn_dists ([float]): Nearest Neighbors' distances.
        Returns:
            dist_interstice_list ([float]): Distance interstice list.
        """
        dist_interstice_list = list()
        for nn_dist, nn_r in zip(nn_dists, nn_rs):
            dist_interstice_list.append(nn_dist / (center_r + nn_r) - 1)
        return dist_interstice_list

    @staticmethod
    def analyze_area_interstice(nn_coords, nn_rs, convex_hull_simplices):
        """Analyze the area interstices in the neighbor convex hull facets.
        Args:
            nn_coords (array-like, shape (N, 3)): Nearest Neighbors' coordinates
            nn_rs ([float]): Nearest Neighbors' radii.
            convex_hull_simplices (array-like, shape (M, 3)): Indices of points
                forming the simplicial facets of convex hull.
        Returns:
            area_interstice_list ([float]): Area interstice list.
        """
        area_interstice_list = list()

        triplet_set = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        for facet_indices in convex_hull_simplices:
            facet_coords = nn_coords[facet_indices]
            facet_rs = nn_rs[facet_indices]
            triangle_angles = list()
            for triplet in triplet_set:
                a = facet_coords[triplet[1]] - facet_coords[triplet[0]]
                b = facet_coords[triplet[2]] - facet_coords[triplet[0]]
                t_a = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                triangle_angles.append(t_a)

            # calculate neighbors' packed area in the facet
            packed_area = 0
            for t_a, nn_r in zip(triangle_angles, facet_rs):
                packed_area += t_a / 2 * pow(nn_r, 2)

            triangle_area = 0.5 * np.linalg.norm(
                np.cross(
                    np.array(facet_coords[0]) - np.array(facet_coords[2]),
                    np.array(facet_coords[1]) - np.array(facet_coords[2]),
                )
            )

            area_interstice = 1 - packed_area / triangle_area  # in fraction
            area_interstice_list.append(area_interstice if area_interstice > 0 else 0)
        return area_interstice_list

    @staticmethod
    def analyze_vol_interstice(center_coords, nn_coords, center_r, nn_rs, convex_hull_simplices):
        """Analyze the volume interstices in the tetrahedra formed by center
        atom and neighbor convex hull triplets.
        Args:
            center_coords ([float]): Central atomic coordinates.
            nn_coords (array-like, shape (N, 3)): Nearest Neighbors' coordinates
            center_r (float): central atom's radius.
            nn_rs ([float]): Nearest Neighbors' radii.
            convex_hull_simplices (array-like, shape (M, 3)): Indices of points
                forming the simplicial facets of convex hull.
        Returns:
            volume_interstice_list ([float]): Volume interstice list.
        """
        volume_interstice_list = list()

        triplet_set = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        for facet_indices in convex_hull_simplices:
            facet_coords = nn_coords[facet_indices]
            facet_rs = nn_rs[facet_indices]
            solid_angles = list()
            for triplet in triplet_set:
                s_a = solid_angle(
                    facet_coords[triplet[0]],
                    np.array(
                        [
                            facet_coords[triplet[1]],
                            facet_coords[triplet[2]],
                            center_coords,
                        ]
                    ),
                )
                solid_angles.append(s_a)

            # calculate neighbors' packed volume in the tetrahedron
            packed_volume = 0
            for s_a, nn_r in zip(solid_angles, facet_rs):
                packed_volume += s_a / 3 * pow(nn_r, 3)

            # add center atom's volume in the tetrahedron
            center_solid_angle = solid_angle(center_coords, facet_coords)
            packed_volume += center_solid_angle / 3 * pow(center_r, 3)

            volume = vol_tetra(center_coords, *facet_coords)

            volume_interstice = 1 - packed_volume / volume
            volume_interstice_list.append(volume_interstice if volume_interstice > 0 else 0)
        return volume_interstice_list

    def feature_labels(self):
        labels = list()
        labels += ["Interstice_dist_%s" % stat for stat in self.stats] if "dist" in self.interstice_types else []
        labels += ["Interstice_area_%s" % stat for stat in self.stats] if "area" in self.interstice_types else []
        labels += ["Interstice_vol_%s" % stat for stat in self.stats] if "vol" in self.interstice_types else []
        return labels

    def citations(self):
        return [
            "@article{Wang2019,"
            "title = {A transferable machine-learning framework linking "
            "interstice distribution and plastic heterogeneity in metallic "
            "glasses}, "
            "author = {Qi Wang and Anubhav Jain},"
            "journal = {Nature Communications}, year = {2019}, "
            "pages = {5537}, volume = {10}, "
            "doi = {10.1038/s41467-019-13511-9}, "
            "url = {https://www.nature.com/articles/s41467-019-13511-9}}"
        ]

    def implementors(self):
        return ["Qi Wang"]


class CoordinationNumber(BaseFeaturizer):
    """
    Number of first nearest neighbors of a site.

    Determines the number of nearest neighbors of a site using one of
    pymatgen's NearNeighbor classes. These nearest neighbor calculators
    can return weights related to the proximity of each neighbor to this
    site. It is possible to take these weights into account to prevent
    the coordination number from changing discontinuously with small
    perturbations of a structure, either by summing the total weights
    or using the normalization method presented by
    [Ward et al.](http://link.aps.org/doi/10.1103/PhysRevB.96.014107)

    Features:
        CN_[method] - Coordination number computed using a certain method
            for calculating nearest neighbors.
    """

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
            CoordinationNumber from a preset.
        """
        nn_ = getattr(pymatgen.analysis.local_env, preset)
        return CoordinationNumber(nn_(**kwargs))

    def __init__(self, nn=None, use_weights="none"):
        """Initialize the featurizer

        Args:
            nn (NearestNeighbor) - Method used to determine coordination number
            use_weights (string) - Method used to account for weights of neighbors:
                'none' - Do not use weights when computing coordination number
                'sum' - Use sum of weights as the coordination number
                'effective' - Compute the 'effective coordination number', which
                    is computed as :math:`\\frac{(\sum_n w_n)^2)}{\sum_n w_n^2}`
        """
        self.nn = nn or VoronoiNN()
        self.use_weights = use_weights

    def featurize(self, struct, idx):
        """
        Get coordintion number of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            [float] - Coordination number
        """
        if self.use_weights is None or self.use_weights == "none":
            return [self.nn.get_cn(struct, idx, use_weights=False)]
        elif self.use_weights == "sum":
            return [self.nn.get_cn(struct, idx, use_weights=True)]
        elif self.use_weights == "effective":
            # TODO: Should this weighting code go in pymatgen? I'm not sure if it even necessary to distinguish it from the 'sum' method -lw
            nns = get_nearest_neighbors(self.nn, struct, idx)
            weights = [n["weight"] for n in nns]
            return [np.sum(weights) ** 2 / np.sum(np.power(weights, 2))]
        else:
            raise ValueError("Weighting method not recognized: " + str(self.use_weights))

    def feature_labels(self):
        # TODO: Should names contain weighting scheme? -lw
        return ["CN_{}".format(self.nn.__class__.__name__)]

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
        return ["Nils E. R. Zimmermann", "Logan Ward"]
