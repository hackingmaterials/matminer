from __future__ import division, unicode_literals, print_function

import itertools
from math import pi, fabs
from operator import itemgetter
import warnings

import numpy as np
import scipy.constants as const

from pymatgen.analysis.defects.point_defects import \
    ValenceIonicRadiusEvaluator
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder as VCF
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import OPSiteFingerprint, OPSiteFingerprint_alt
from matminer.featurizers.stats import PropertyStats

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, ' \
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>'
# ("@article{label, title={}, volume={}, DOI={}, number={}, pages={}, journal={}, author={}, year={}}")

ANG_TO_BOHR = const.value('Angstrom star') / const.value('Bohr radius')


# To do:
# - Use local_env-based neighbor finding
#   once this is part of the stable Pymatgen version.
# - Use more than 1 method for MinimumRelativeDistance

class DensityFeatures(BaseFeaturizer):

    def __init__(self, desired_features=None):
        self.features = ["density", "vpa", "packing fraction"] if not \
            desired_features else desired_features

    def featurize(self, s):
        output = []

        if "density" in self.features:
            output.append(s.density)

        if "vpa" in self.features:
            if not s.is_ordered:
                raise ValueError("Disordered structure support not built yet.")
            output.append(s.volume / len(s))

        if "packing fraction" in self.features:
            if not s.is_ordered:
                raise ValueError("Disordered structure support not built yet.")
            total_rad = 0
            for site in s:
                total_rad += site.specie.atomic_radius ** 3
            output.append(4 * pi * total_rad / (3 * s.volume))

        return output

    def feature_labels(self):
        all_features = ["density", "vpa", "packing fraction"]  # enforce order
        return [x for x in all_features if x in self.features]

    def citations(self):
        return [""]

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]


class GlobalSymmetryFeatures(BaseFeaturizer):

    crystal_idx = {"triclinic": 7,
                   "monoclinic": 6,
                   "orthorhombic": 5,
                   "tetragonal": 4,
                   "trigonal": 3,
                   "hexagonal": 2,
                   "cubic": 1
                   }

    def __init__(self, desired_features = None):
        self.features = ["spacegroup_num", "crystal_system",
                         "crystal_system_int", "is_centrosymmetric"] if not \
            desired_features else desired_features

    def featurize(self, s):
        sga = SpacegroupAnalyzer(s)
        output = []

        if "spacegroup_num" in self.features:
            output.append(sga.get_space_group_number())

        if "crystal_system" in self.features:
            output.append(sga.get_crystal_system())

        if "crystal_system_int" in self.features:
            output.append(GlobalSymmetryFeatures.crystal_idx[
                              sga.get_crystal_system()])

        if "is_centrosymmetric" in self.features:
            output.append(sga.is_laue())

        return output

    def feature_labels(self):
        all_features = ["spacegroup_num", "crystal_system",
                        "crystal_system_int", "is_centrosymmetric"]  # enforce order
        return [x for x in all_features if x in self.features]

    def citations(self):
        return [""]

    def implementors(self):
        return ["Anubhav Jain"]


class RadialDistributionFunction(BaseFeaturizer):
    """
    Calculate the radial distribution function (RDF) of a crystal
    structure.
    Args:
        cutoff: (float) distance up to which to calculate the RDF.
        bin_size: (float) size of each bin of the (discrete) RDF.
    """

    def __init__(self, cutoff=20.0, bin_size=0.1):
        self.cutoff = cutoff
        self.bin_size = bin_size

    def featurize(self, s):
        """
        Get RDF of the input structure.
        Args:
            s: Pymatgen Structure object.

        Returns:
            rdf, dist: (tuple of arrays) the first element is the
                    normalized RDF, whereas the second element is
                    the inner radius of the RDF bin.
        """
        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)
        all_distances = np.concatenate(
            tuple(map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)))

        # Compute a histogram
        rdf_dict = {}
        dist_hist, dist_bins = np.histogram(
            all_distances, bins=np.arange(
                0, self.cutoff + self.bin_size, self.bin_size), density=False)

        # Normalize counts
        shell_vol = 4.0 / 3.0 * pi * (np.power(
            dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = s.num_sites / s.volume
        rdf = dist_hist / shell_vol / number_density
        return [{'distances': dist_bins[:-1], 'distribution': rdf}]

    def feature_labels(self):
        return ["radial distribution function"]

    def citations(self):
        return ("")

    def implementors(self):
        return ("Saurabh Bajaj")


class PartialRadialDistributionFunction(BaseFeaturizer):
    """
    Compute the partial radial distribution function (PRDF) of a crystal
    structure, which is the radial distibution function
    broken down for each pair of atom types.  The PRDF was proposed as a
    structural descriptor by [Schutt *et al.*]
    (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118)
    Args:
        cutoff: (float) distance up to which to calculate the RDF.
        bin_size: (float) size of each bin of the (discrete) RDF.
    """

    def __init__(self, cutoff=20.0, bin_size=0.1):
        self.cutoff = cutoff
        self.bin_size = bin_size

    def featurize(self, s):
        """
        Get PRDF of the input structure.
        Args:
            s: Pymatgen Structure object.

        Returns:
            prdf, dist: (tuple of arrays) the first element is a
                    dictionary where keys are tuples of element
                    names and values are PRDFs.
        """

        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the composition of the array
        composition = s.composition.fractional_composition.to_reduced_dict

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        # Sort neighbors by type
        distances_by_type = {}
        for p in itertools.product(composition.keys(), composition.keys()):
            distances_by_type[p] = []

        def get_symbol(site):
            return site.specie.symbol if isinstance(site.specie,
                                                    Element) else site.specie.element.symbol

        for site, nlst in zip(s.sites, neighbors_lst):  # Each list is a list for each site
            my_elem = get_symbol(site)

            for neighbor in nlst:
                rij = neighbor[1]
                n_elem = get_symbol(neighbor[0])
                # LW 3May17: Any better ideas than appending each element at a time?
                distances_by_type[(my_elem, n_elem)].append(rij)

        # Compute and normalize the prdfs
        prdf = {}
        dist_bins = np.arange(0, self.cutoff + self.bin_size, self.bin_size)
        shell_volume = 4.0 / 3.0 * pi * (
            np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        for key, distances in distances_by_type.items():
            # Compute histogram of distances
            dist_hist, dist_bins = np.histogram(distances,
                                                bins=dist_bins, density=False)
            # Normalize
            n_alpha = composition[key[0]] * s.num_sites
            rdf = dist_hist / shell_volume / n_alpha

            prdf[key] = {'distances': dist_bins, 'distribution': rdf}

        return [prdf]

    def feature_labels(self):
        return ["partial radial distribution functions"]

    def citations(self):
        return ("")

    def implementors(self):
        return ("Saurabh Bajaj")


class RadialDistributionFunctionPeaks(BaseFeaturizer):
    """
    Determine the location of the highest peaks in the radial distribution
    function (RDF) of a structure.
    Args:
        n_peaks: (int) number of the top peaks to return .
    """

    def __init__(self, n_peaks=2):
        self.n_peaks = n_peaks

    def featurize(self, rdf):
        """
        Get location of highest peaks in RDF.
    
        Args:
            rdf: (ndarray) RDF as obtained from the
                    RadialDistributionFunction class.
    
        Returns: (ndarray) distances of highest peaks in descending order
                of the peak height
        """

        return [[rdf[0]['distances'][i] for i in np.argsort(
            rdf[0]['distribution'])[-self.n_peaks:]][::-1]]

    def feature_labels(self):
        return ["radial distribution function peaks"]

    def citations(self):
        return ("")

    def implementors(self):
        return ("Saurabh Bajaj")


class ElectronicRadialDistributionFunction(BaseFeaturizer):
    """
    Calculate the crystal structure-inherent
    electronic radial distribution function (ReDF) according to
    Willighagen et al., Acta Cryst., 2005, B61, 29-36.
    The ReDF is a structure-integral RDF (i.e., summed over
    all sites) in which the positions of neighboring sites
    are weighted by electrostatic interactions inferred
    from atomic partial charges. Atomic charges are obtained
    from the ValenceIonicRadiusEvaluator class.
    Args:
        cutoff: (float) distance up to which the ReDF is to be
                calculated (default: longest diagaonal in
                primitive cell).
        dr: (float) width of bins ("x"-axis) of ReDF (default: 0.05 A).
    """

    def __init__(self, cutoff=None, dr=0.05):
        self.cutoff = cutoff
        self.dr = dr

    def featurize(self, s):
        """
        Get ReDF of input structure.

        Args:
            s: input Structure object.

        Returns: (dict) a copy of the electronic radial distribution
                functions (ReDF) as a dictionary. The distance list
                ("x"-axis values of ReDF) can be accessed via key
                'distances'; the ReDF itself is accessible via key
                'redf'.
        """
        if self.dr <= 0:
            raise ValueError("width of bins for ReDF must be >0")

        # Make structure primitive.
        struct = SpacegroupAnalyzer(s).find_primitive() or s

        # Add oxidation states.
        struct = ValenceIonicRadiusEvaluator(struct).structure

        if self.cutoff is None:
            # Set cutoff to longest diagonal.
            a = struct.lattice.matrix[0]
            b = struct.lattice.matrix[1]
            c = struct.lattice.matrix[2]
            self.cutoff = max(
                [np.linalg.norm(a + b + c), np.linalg.norm(-a + b + c),
                 np.linalg.norm(a - b + c), np.linalg.norm(a + b - c)])

        nbins = int(self.cutoff / self.dr) + 1
        redf_dict = {"distances": np.array(
            [(i + 0.5) * self.dr for i in range(nbins)]),
            "distribution": np.zeros(nbins, dtype=np.float)}

        for site in struct.sites:
            this_charge = float(site.specie.oxi_state)
            neighs_dists = struct.get_neighbors(site, self.cutoff)
            for neigh, dist in neighs_dists:
                neigh_charge = float(neigh.specie.oxi_state)
                bin_index = int(dist / self.dr)
                redf_dict["distribution"][bin_index] += (
                                                            this_charge * neigh_charge) / (
                                                            struct.num_sites * dist)

        return [redf_dict]

    def feature_labels(self):
        return ["electronic radial distribution function"]

    def citations(self):
        return ("@article{title={Method for the computational comparison"
                " of crystal structures}, volume={B61}, pages={29-36},"
                " DOI={10.1107/S0108768104028344},"
                " journal={Acta Crystallographica Section B},"
                " author={Willighagen, E. L. and Wehrens, R. and Verwer,"
                " P. and de Gelder R. and Buydens, L. M. C.}, year={2005}}")

    def implementors(self):
        return ("Nils E. R. Zimmermann")


class CoulombMatrix(BaseFeaturizer):
    """
    Generate the Coulomb matrix, M, of the input
    structure (or molecule).  The Coulomb matrix was put forward by
    Rupp et al. (Phys. Rev. Lett. 108, 058301, 2012) and is defined by
    off-diagonal elements M_ij = Z_i*Z_j/|R_i-R_j|
    and diagonal elements 0.5*Z_i^2.4, where Z_i and R_i denote
    the nuclear charge and the position of atom i, respectively.

    Args:
        diag_elems: (bool) flag indicating whether (True, default) to use
                    the original definition of the diagonal elements;
                    if set to False, the diagonal elements are set to zero.
    """

    def __init__(self, diag_elems=True):
        self.diag_elems = diag_elems

    def featurize(self, s):
        """
        Get Coulomb matrix of input structure.
    
        Args:
            s: input Structure (or Molecule) object.
    
        Returns:
            m: (Nsites x Nsites matrix) Coulomb matrix.
        """
        m = [[] for site in s.sites]
        z = []
        for site in s.sites:
            if isinstance(site, Specie):
                z.append(Element(site.element.symbol).Z)
            else:
                z.append(Element(site.species_string).Z)
        for i in range(s.num_sites):
            for j in range(s.num_sites):
                if i == j:
                    if self.diag_elems:
                        m[i].append(0.5 * z[i] ** 2.4)
                    else:
                        m[i].append(0)
                else:
                    d = s.get_distance(i, j) * ANG_TO_BOHR
                    m[i].append(z[i] * z[j] / d)
        return [np.array(m)]

    def feature_labels(self):
        return ["coulomb matrix"]

    def citations(self):
        return ("@article{rupp_tkatchenko_muller_vonlilienfeld_2012, title={"
                "Fast and accurate modeling of molecular atomization energies"
                " with machine learning}, volume={108},"
                " DOI={10.1103/PhysRevLett.108.058301}, number={5},"
                " pages={058301}, journal={Physical Review Letters}, author={"
                "Rupp, Matthias and Tkatchenko, Alexandre and M\"uller,"
                " Klaus-Robert and von Lilienfeld, O. Anatole}, year={2012}}")

    def implementors(self):
        return ["Nils E. R. Zimmermann"]


class SineCoulombMatrix(BaseFeaturizer):
    """
    This function generates a variant of the Coulomb matrix developed
    for periodic crystals by Faber et al. (Inter. J. Quantum Chem.
    115, 16, 2015). It is identical to the Coulomb matrix, except
    that the inverse distance function is replaced by the inverse of a
    sin**2 function of the vector between the sites which is periodic
    in the dimensions of the structure lattice. See paper for details.

    Args:
        diag_elems (bool): flag indication whether (True, default) to use
                the original definition of the diagonal elements;
                if set to False, the diagonal elements are set to 0
    """

    def __init__(self, diag_elems=True):
        self.diag_elems = diag_elems

    def featurize(self, s):
        """
        Args:
            s (Structure or Molecule): input structure (or molecule)

        Returns:
            (Nsites x Nsites matrix) Sine matrix.
        """
        sites = s.sites
        Zs = np.array([site.specie.Z for site in sites])
        sin_mat = np.zeros((len(sites), len(sites)))
        coords = np.array([site.frac_coords for site in sites])
        lattice = s.lattice.matrix
        pi = np.pi

        for i in range(len(sin_mat)):
            for j in range(len(sin_mat)):
                if i == j:
                    if self.diag_elems:
                        sin_mat[i][i] = 0.5 * Zs[i] ** 2.4
                elif i < j:
                    vec = coords[i] - coords[j]
                    coord_vec = np.sin(pi * vec) ** 2
                    trig_dist = np.linalg.norm((np.matrix(coord_vec) * lattice).A1) * ANG_TO_BOHR
                    sin_mat[i][j] = Zs[i] * Zs[j] / trig_dist
                else:
                    sin_mat[i][j] = sin_mat[j][i]
        return [sin_mat]

    def feature_labels(self):
        return ["sine coulomb matrix"]

    def citations(self):
        return ("@article {QUA:QUA24917,"
                "author = {Faber, Felix and Lindmaa, Alexander and von Lilienfeld, O. Anatole and Armiento, Rickard},"
                "title = {Crystal structure representations for machine learning models of formation energies},"
                "journal = {International Journal of Quantum Chemistry},"
                "volume = {115},"
                "number = {16},"
                "issn = {1097-461X},"
                "url = {http://dx.doi.org/10.1002/qua.24917},"
                "doi = {10.1002/qua.24917},"
                "pages = {1094--1101},"
                "keywords = {machine learning, formation energies, representations, crystal structure, periodic systems},"
                "year = {2015},"
                "}")

    def implementors(self):
        return ["Kyle Bystrom"]


class OrbitalFieldMatrix(BaseFeaturizer):
    """
    This function generates an orbital field matrix (OFM) as developed
    by Pham et al (arXiv, May 2017). Each atom is described by a 32-element
    vector (or 39-element vector, see period tag for details) uniquely
    representing the valence subshell. A 32x32 (39x39) matrix is formed
    by multiplying two atomic vectors. An OFM for an atomic environment is the
    sum of these matrices for each atom the center atom coordinates with
    multiplied by a distance function (In this case, 1/r times the weight of
    the coordinating atom in the Voronoi Polyhedra method). The OFM of a structure
    or molecule is the average of the OFMs for all the sites in the structure.

    Args:
        period_tag (bool): In the original OFM, an element is represented
                by a vector of length 32, where each element is 1 or 0,
                which represents the valence subshell of the element.
                With period_tag=True, the vector size is increased
                to 39, where the 7 extra elements represent the period
                of the element. Note lanthanides are treated as period 6,
                actinides as period 7. Default False as in the original paper.

    ...attribute:: size
        Either 32 or 39, the size of the vectors used to describe elements.
    """

    def __init__(self, period_tag = False):
        my_ohvs = {}
        if period_tag:
            self.size = 39
        else:
            self.size = 32
        for Z in range(1, 95):
            el = Element.from_Z(Z)
            my_ohvs[Z] = self.get_ohv(el, period_tag)
            my_ohvs[Z] = np.matrix(my_ohvs[Z])
        self.ohvs = my_ohvs

    def get_ohv(self, sp, period_tag):
        """
        Get the "one-hot-vector" for pymatgen Element sp. This 32 or 39-length
        vector represents the valence shell of the given element.
        Args:
            sp (Element): element whose ohv should be returned
            period_tag (bool): If true, the vector contains items
                    corresponding to the period of the element

        Returns:
            my_ohv (numpy array length 39 if period_tag, else 32): ohv for sp
        """
        el_struct = sp.full_electronic_structure
        ohd = {j: {i + 1: 0 for i in range(2 * (2 * j + 1))} for j in range(4)}
        nume = 0
        shell_num = 0
        max_n = el_struct[-1][0]
        while shell_num < len(el_struct):
            if el_struct[-1 - shell_num][0] < max_n - 2:
                shell_num += 1
                continue
            elif el_struct[-1 - shell_num][0] < max_n - 1 and el_struct[-1 - shell_num][1] != u'f':
                shell_num += 1
                continue
            elif el_struct[-1 - shell_num][0] < max_n and (
                            el_struct[-1 - shell_num][1] != u'd' and el_struct[-1 - shell_num][1] != u'f'):
                shell_num += 1
                continue
            curr_shell = el_struct[-1 - shell_num]
            if curr_shell[1] == u's':
                l = 0
            elif curr_shell[1] == u'p':
                l = 1
            elif curr_shell[1] == u'd':
                l = 2
            elif curr_shell[1] == u'f':
                l = 3
            ohd[l][curr_shell[2]] = 1
            nume += curr_shell[2]
            shell_num += 1
        my_ohv = np.zeros(self.size, np.int)
        k = 0
        for j in range(4):
            for i in range(2 * (2 * j + 1)):
                my_ohv[k] = ohd[j][i + 1]
                k += 1
        if period_tag:
            row = sp.row
            if row > 7:
                row -= 2
            my_ohv[row+31] = 1
        return my_ohv

    def get_single_ofm(self, site, site_dict):
        """
        Gets the orbital field matrix for a single chemical environment,
        where site is the center atom whose environment is characterized and
        site_dict is a dictionary of site : weight, where the weights are the
        Voronoi Polyhedra weights of the corresponding coordinating sites.

        Args:
            site (Site): center atom
            site_dict (dict of Site:float): chemical environment

        Returns:
            atom_ofm (size X size numpy matrix): ofm for site
        """
        ohvs = self.ohvs
        atom_ofm = np.matrix(np.zeros((self.size, self.size)))
        ref_atom = ohvs[site.specie.Z]
        for other_site in site_dict:
            scale = site_dict[other_site]
            other_atom = ohvs[other_site.specie.Z]
            atom_ofm += other_atom.T * ref_atom * scale / site.distance(other_site) / ANG_TO_BOHR
        return atom_ofm

    def get_atom_ofms(self, struct, symm=False):
        """
        Calls get_single_ofm for every site in struct. If symm=True,
        get_single_ofm is called for symmetrically distinct sites, and
        counts is constructed such that ofms[i] occurs counts[i] times
        in the structure

        Args:
            struct (Structure): structure for find ofms for
            symm (bool): whether to calculate ofm for only symmetrically
                    distinct sites

        Returns:
            ofms ([size X size matrix] X len(struct)): ofms for struct
            if symm:
                ofms ([size X size matrix] X number of symmetrically distinct sites):
                    ofms for struct
                counts: number of identical sites for each ofm
        """
        ofms = []
        vcf = VCF(struct, allow_pathological=True)
        if symm:
            symm_struct = SpacegroupAnalyzer(struct).get_symmetrized_structure()
            indices = [lst[0] for lst in symm_struct.equivalent_indices]
            counts = [len(lst) for lst in symm_struct.equivalent_indices]
        else:
            indices = [i for i in range(len(struct.sites))]
        for index in indices:
            ofms.append(self.get_single_ofm(struct.sites[index], \
                                            vcf.get_voronoi_polyhedra(index)))
        if symm:
            return ofms, counts
        return ofms

    def get_mean_ofm(self, ofms, counts):
        """
        Averages a list of ofms, weights by counts
        """
        ofms = [ofm * c for ofm, c in zip(ofms, counts)]
        return sum(ofms) / sum(counts)

    def get_structure_ofm(self, struct):
        """
        Calls get_mean_ofm on the results of get_atom_ofms
        to give a size X size matrix characterizing a structure
        """
        ofms, counts = self.get_atom_ofms(struct, True)
        return self.get_mean_ofm(ofms, counts)

    def featurize(self, s):
        """
        Makes a supercell for structure s (to protect sites
        from coordinating with themselves), and then finds the mean
        of the orbital field matrices of each site to characterize
        a structure

        Args:
            s (Structure): structure to characterize

        Returns:
            mean_ofm (size X size matrix): orbital field matrix
                    characterizing s
        """
        s *= [3, 3, 3]
        ofms, counts = self.get_atom_ofms(s, True)
        mean_ofm = self.get_mean_ofm(ofms, counts)
        return [mean_ofm]

    def feature_labels(self):
        return ["orbital field matrix"]

    def citations(self):
        return ("@ARTICLE{2017arXiv170501043P,"
                "author = {{Pham}, T. L. and {Kino}, H. and {Terakura}, K. and {Miyake}, T. and "
                "{Takigawa}, I. and {Tsuda}, K. and {Dam}, H. C.},"
                "title = \"{Machine learning reveals orbital interaction in crystalline materials}\","
                "journal = {ArXiv e-prints},"
                "archivePrefix = \"arXiv\","
                "eprint = {1705.01043},"
                "primaryClass = \"cond-mat.mtrl-sci\","
                "keywords = {Condensed Matter - Materials Science},"
                "year = 2017,"
                "month = may,"
                "adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170501043P},"
                "adsnote = {Provided by the SAO/NASA Astrophysics Data System}"
                "}")

    def implementors(self):
        return ["Kyle Bystrom"]


class MinimumRelativeDistances(BaseFeaturizer):
    """
    Determines the relative distance of each site to its closest
    neighbor. We use the relative distance,
    f_ij = r_ij / (r^atom_i + r^atom_j), as a measure rather than the
    absolute distances, r_ij, to account for the fact that different
    atoms/species have different sizes.  The function uses the
    valence-ionic radius estimator implemented in Pymatgen.
    Args:
        cutoff: (float) (absolute) distance up to which tentative
                closest neighbors (on the basis of relative distances)
                are to be determined.
    """

    def __init__(self, cutoff=10.0):
        self.cutoff = cutoff

    def featurize(self, s, cutoff=10.0):
        """
        Get minimum relative distances of all sites of the input structure.
    
        Args:
            s: Pymatgen Structure object.

        Returns:
            min_rel_dists: (list of floats) list of all minimum relative
                    distances (i.e., for all sites).
        """
        vire = ValenceIonicRadiusEvaluator(s)
        min_rel_dists = []
        for site in vire.structure:
            min_rel_dists.append(min([dist / (
                vire.radii[site.species_string] +
                vire.radii[neigh.species_string]) for neigh, dist in \
                                      vire.structure.get_neighbors(site, self.cutoff)]))
        return [min_rel_dists[:]]

    def feature_labels(self):
        return ["minimum relative distance of each site"]

    def citations(self):
        return ("")

    def implementors(self):
        return ("Nils E. R. Zimmermann")


class OPStructureFingerprint(BaseFeaturizer):
    """
    Calculates all order parameters (OPs) for all sites in a crystal
    structure.
    Args:
        op_site_fp (OPSiteFingerprint): defines the types of order
            parameters to be calculated.
        stats ([str]): list of weighted statistics to compute for each feature.
            If stats is None, for each order parameter, a list is returned that
            contains the calculated parameter for each site in the structure.
            *Note for nth mode, stat must be 'n*_mode'; e.g. stat='2nd_mode'
        min_oxi (int): minimum site oxidation state for inclusion (e.g.,
            zero means metals/cations only)
        max_oxi (int): maximum site oxidation state for inclusion
    """
    def __init__(self, op_site_fp=None, stats=('mean', 'std_dev', 'minimum',
                                               'maximum'), min_oxi=None,
                 max_oxi=None):

        self.op_site_fp = OPSiteFingerprint() if op_site_fp is None \
            else op_site_fp
        self._labels = self.op_site_fp.feature_labels()
        self.stats = tuple([stats]) if type(stats) == str else stats
        if self.stats and '_mode' in ''.join(self.stats):
            nmodes = 0
            for stat in self.stats:
                if '_mode' in stat and int(stat[0]) > nmodes:
                    nmodes = int(stat[0])
            self.nmodes = nmodes

        self.min_oxi = min_oxi
        self.max_oxi = max_oxi

    def featurize(self, s):
        """
        Calculate all sites' local structure order parameters (LSOPs).

        Args:
            s: Pymatgen Structure object.

            Returns:
                opvals: (2D array of floats) LSOP values of all sites'
                (1st dimension) order parameters (2nd dimension). 46 order
                parameters are computed per site: q_cn (coordination
                number), q_lin, 35 x q_bent (starting with a target angle
                of 5 degrees and, increasing by 5 degrees, until 175 degrees),
                q_tet, q_oct, q_bcc, q_2, q_4, q_6, q_reg_tri, q_sq, q_sq_pyr.
        """
        opvals = [[] for t in self._labels]
        for i, site in enumerate(s.sites):
            if (self.min_oxi is None or site.specie.oxi_state >= self.min_oxi) \
                    and (self.max_oxi is None or site.specie.oxi_state >= self.max_oxi):
                opvalstmp = self.op_site_fp.featurize(s, i)
                for j, opval in enumerate(opvalstmp):
                    if opval is None:
                        opvals[j].append(0.0)
                    else:
                        opvals[j].append(opval)

        if self.stats:
            opstats = []
            for op in opvals:
                if '_mode' in ''.join(self.stats):
                    modes = PropertyStats().n_numerical_modes(
                            op, self.nmodes, 0.01)
                for stat in self.stats:
                    if '_mode' in stat:
                        opstats.append(modes[int(stat[0])-1])
                    else:
                        opstats.append(PropertyStats().calc_stat(op, stat))

            return opstats
        else:
            return opvals

    def feature_labels(self):
        if self.stats:
            labels = []
            for attr in self._labels:
                for stat in self.stats:
                    labels.append('%s %s' % (stat, attr))
            return labels
        else:
            return self._labels

    def citations(self):
        return ('@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}')

    def implementors(self):
        return (['Nils E. R. Zimmermann', 'Alireza Faghaninia', 'Anubhav Jain'])


def get_op_stats_vector_diff(s1, s2, max_dr=0.2, ddr=0.01, ddist=0.01):
    """
    Determine the difference vector between two order parameter-statistics
    feature vector resulting from two input structures.

    Args:
        s1 (Structure): first input structure.
        s2 (Structure): second input structure.
        max_dr (float): maximum neighbor-finding parameter to be tested.
        ddr (float): step size for increasing neighbor-finding parameter.
        ddist (float): bin size for histogramming distances of varying dr.

    Returns: (float, [float]) optimal neighbor-finding parameter
        and difference vector between order
        parameter-statistics feature vectors obtained from the
        two input structures (s1 - s2).
    """
    # Compute OP stats vector distances for varying neigh-find paras.
    dr = []
    dist = []
    delta = []
    nbins = int(max_dr/ddr) + 1
    for i in range(nbins):
        dr.append(float(i+1)*ddr)
        opsf = OPStructureFingerprint(op_site_fp=OPSiteFingerprint(dr=dr[i]))
        delta.append(np.array(
            opsf.featurize(s1)) - np.array(opsf.featurize(s2)))
        dist.append(np.linalg.norm(delta[i]))

    # Compute distance histogram, determine peak, and location
    # of smallest dr with peak value.
    nbins = int(max(dist) / ddist) + 1
    hist, bin_edges = np.histogram(
        dist, bins=[float(i)*ddist for i in range(nbins)],
        normed=False, weights=None, density=False)
    idx = list(hist).index(max(hist))
    dist_peak = 0.5 * (bin_edges[idx] + bin_edges[idx+1])
    idx = -1
    for i, d in enumerate(dist):
        if fabs(d - dist_peak) <= ddist:
            idx = i
            break

    return dr[idx], delta[idx]


def get_op_stats_vector_diff_alt(s1, s2, angle_weight=1):
    """
    Compute structure distance using an alternate (test) algorithm. Docs are
    minimal for now.
    """
    site_f = OPSiteFingerprint_alt()
    structure_f = OPStructureFingerprint(op_site_fp=site_f, stats=("mean",))

    f1 = structure_f.featurize(s1)
    f2 = structure_f.featurize(s2)

    angle_distance = 0
    if angle_weight > 0:
        # compute angle between feature vectors
        # TODO: add StackOverflow link
        f1_u = f1 / np.linalg.norm(f1)  # unit vector
        f2_u = f2 / np.linalg.norm(f2)  # unit vector
        angle_distance = np.arccos(np.clip(np.dot(f1_u, f2_u), -1.0, 1.0))

    euclidean_weight = 1 - angle_weight
    euclidean_distance = 0
    if euclidean_weight > 0:
        euclidean_distance = np.linalg.norm(np.array(f1) - np.array(f2))

    return (angle_weight * angle_distance) + \
           (euclidean_weight * euclidean_distance)



