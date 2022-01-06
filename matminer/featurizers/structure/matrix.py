"""
Structure featurizers generating a matrix for each structure.

Most matrix structure featurizers contain the ability to flatten matrices to be dataframe-friendly.
"""
import numpy as np
import pymatgen.analysis.local_env as pmg_le
import scipy.constants as const
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.exceptions import NotFittedError

from matminer.featurizers.base import BaseFeaturizer

ANG_TO_BOHR = const.value("Angstrom star") / const.value("Bohr radius")


class CoulombMatrix(BaseFeaturizer):
    """
    The Coulomb matrix, a representation of nuclear coulombic interaction.

    Generate the Coulomb matrix, M, of the input structure (or molecule). The
    Coulomb matrix was put forward by Rupp et al. (Phys. Rev. Lett. 108, 058301,
    2012) and is defined by off-diagonal elements M_ij = Z_i*Z_j/|R_i-R_j| and
    diagonal elements 0.5*Z_i^2.4, where Z_i and R_i denote the nuclear charge
    and the position of atom i, respectively.

    Coulomb Matrix features are flattened (for ML-readiness) by default. Use
    fit before featurizing to use flattened features. To return the matrix form,
    set flatten=False.

    Args:
        diag_elems (bool): flag indication whether (True, default) to use
            the original definition of the diagonal elements; if set to False,
            the diagonal elements are set to 0
        flatten (bool): If True, returns a flattened vector based on eigenvalues
            of the matrix form. Otherwise, returns a matrix object (single
            feature), which will likely need to be processed further.
    """

    def __init__(self, diag_elems=True, flatten=True):
        self.diag_elems = diag_elems
        self.flatten = flatten
        self._max_eigs = None

    def _check_fitted(self):
        if self.flatten and not self._max_eigs:
            raise NotFittedError("Please fit the CoulombMatrix before " "featurizing if using flatten=True.")

    def fit(self, X, y=None):
        """
        Fit the Coulomb Matrix to a list of structures.

        Args:
            X ([Structure]): A list of pymatgen structures.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self
        """
        if self.flatten:
            n_sites = [structure.num_sites for structure in X]
            # CM makes sites x sites matrix; max eigvals for n x n matrix is n
            self._max_eigs = max(n_sites)
        return self

    def featurize(self, s):
        """
        Get Coulomb matrix of input structure.

        Args:
            s: input Structure (or Molecule) object.

        Returns:
            m: (Nsites x Nsites matrix) Coulomb matrix.
        """
        self._check_fitted()
        m = np.zeros((s.num_sites, s.num_sites))
        atomic_numbers = []
        for site in s.sites:
            if isinstance(site.specie, Element):
                atomic_numbers.append(site.specie.Z)
            else:
                atomic_numbers.append(site.specie.element.Z)
        for i in range(s.num_sites):
            for j in range(s.num_sites):
                if i == j:
                    if self.diag_elems:
                        m[i, j] = 0.5 * atomic_numbers[i] ** 2.4
                    else:
                        m[i, j] = 0
                else:
                    d = s.get_distance(i, j) * ANG_TO_BOHR
                    m[i, j] = atomic_numbers[i] * atomic_numbers[j] / d
        cm = np.array(m)

        if self.flatten:
            eigs, _ = np.linalg.eig(cm)
            zeros = np.zeros((self._max_eigs,))
            zeros[: len(eigs)] = eigs
            return zeros
        else:
            return [cm]

    def feature_labels(self):
        self._check_fitted()
        if self.flatten:
            return [f"coulomb matrix eig {i}" for i in range(self._max_eigs)]
        else:
            return ["coulomb matrix"]

    def citations(self):
        return [
            "@article{rupp_tkatchenko_muller_vonlilienfeld_2012, title={"
            "Fast and accurate modeling of molecular atomization energies"
            " with machine learning}, volume={108},"
            " DOI={10.1103/PhysRevLett.108.058301}, number={5},"
            " pages={058301}, journal={Physical Review Letters}, author={"
            'Rupp, Matthias and Tkatchenko, Alexandre and M"uller,'
            " Klaus-Robert and von Lilienfeld, O. Anatole}, year={2012}}"
        ]

    def implementors(self):
        return ["Nils E. R. Zimmermann", "Alex Dunn"]


class SineCoulombMatrix(BaseFeaturizer):
    """
    A variant of the Coulomb matrix developed for periodic crystals.

    This function generates a variant of the Coulomb matrix developed
    for periodic crystals by Faber et al. (Inter. J. Quantum Chem.
    115, 16, 2015). It is identical to the Coulomb matrix, except
    that the inverse distance function is replaced by the inverse of a
    sin**2 function of the vector between the sites which is periodic
    in the dimensions of the structure lattice. See paper for details.

    Coulomb Matrix features are flattened (for ML-readiness) by default. Use
    fit before featurizing to use flattened features. To return the matrix form,
    set flatten=False.

    Args:
        diag_elems (bool): flag indication whether (True, default) to use
            the original definition of the diagonal elements; if set to False,
            the diagonal elements are set to 0
        flatten (bool): If True, returns a flattened vector based on eigenvalues
            of the matrix form. Otherwise, returns a matrix object (single
            feature), which will likely need to be processed further.
    """

    def __init__(self, diag_elems=True, flatten=True):
        self.diag_elems = diag_elems
        self.flatten = flatten
        self._max_eigs = None

    def _check_fitted(self):
        if self.flatten and not self._max_eigs:
            raise NotFittedError("Please fit the SineCoulombMatrix before " "featurizing if using flatten=True.")

    def fit(self, X, y=None):
        """
        Fit the Sine Coulomb Matrix to a list of structures.

        Args:
            X ([Structure]): A list of pymatgen structures.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self
        """
        if self.flatten:
            nsites = [structure.num_sites for structure in X]
            self._max_eigs = max(nsites)
        return self

    def featurize(self, s):
        """
        Args:
            s (Structure or Molecule): input structure (or molecule)

        Returns:
            (Nsites x Nsites matrix) Sine matrix or
        """
        self._check_fitted()
        sites = s.sites
        atomic_numbers = np.array([site.specie.Z for site in sites])
        sin_mat = np.zeros((len(sites), len(sites)))
        coords = np.array([site.frac_coords for site in sites])
        lattice = s.lattice.matrix

        for i in range(len(sin_mat)):
            for j in range(len(sin_mat)):
                if i == j:
                    if self.diag_elems:
                        sin_mat[i][i] = 0.5 * atomic_numbers[i] ** 2.4
                elif i < j:
                    vec = coords[i] - coords[j]
                    coord_vec = np.sin(np.pi * vec) ** 2
                    trig_dist = np.linalg.norm((np.matrix(coord_vec) * lattice).A1) * ANG_TO_BOHR
                    sin_mat[i][j] = atomic_numbers[i] * atomic_numbers[j] / trig_dist
                else:
                    sin_mat[i][j] = sin_mat[j][i]
        if self.flatten:
            eigs, _ = np.linalg.eig(sin_mat)
            zeros = np.zeros((self._max_eigs,))
            zeros[: len(eigs)] = eigs
            return zeros
        else:
            return [sin_mat]

    def feature_labels(self):
        self._check_fitted()
        if self.flatten:
            return [f"sine coulomb matrix eig {i}" for i in range(self._max_eigs)]
        else:
            return ["sine coulomb matrix"]

    def citations(self):
        return [
            "@article {QUA:QUA24917,"
            "author = {Faber, Felix and Lindmaa, Alexander and von "
            "Lilienfeld, O. Anatole and Armiento, Rickard},"
            "title = {Crystal structure representations for machine "
            "learning models of formation energies},"
            "journal = {International Journal of Quantum Chemistry},"
            "volume = {115},"
            "number = {16},"
            "issn = {1097-461X},"
            "url = {http://dx.doi.org/10.1002/qua.24917},"
            "doi = {10.1002/qua.24917},"
            "pages = {1094--1101},"
            "keywords = {machine learning, formation energies, "
            "representations, crystal structure, periodic systems},"
            "year = {2015},"
            "}"
        ]

    def implementors(self):
        return ["Kyle Bystrom", "Alex Dunn"]


class OrbitalFieldMatrix(BaseFeaturizer):
    """
    Representation based on the valence shell electrons of neighboring atoms.

    Each atom is described by a 32-element vector (or 39-element vector, see
    period tag for details) uniquely representing the valence subshell.
    A 32x32 (39x39) matrix is formed by multiplying two atomic vectors.
    An OFM for an atomic environment is the sum of these matrices for each atom
    the center atom coordinates with multiplied by a distance function
    (In this case, 1/r times the weight of the coordinating atom in the Voronoi
     Polyhedra method). The OFM of a structure or molecule is the average of the
     OFMs for all the sites in the structure.

    Args:
        period_tag (bool): In the original OFM, an element is represented
            by a vector of length 32, where each element is 1 or 0,
            which represents the valence subshell of the element.
            With period_tag=True, the vector size is increased
            to 39, where the 7 extra elements represent the period
            of the element. Note lanthanides are treated as period 6,
            actinides as period 7. Default False as in the original paper.
        flatten (bool): Flatten the avg OFM to a 1024-vector (if period_tag
            False) or a 1521-vector (if period_tag=True).

    ...attribute:: size
        Either 32 or 39, the size of the vectors used to describe elements.

    Reference:
        `Pham et al. _Sci Tech Adv Mat_. 2017 <http://dx.doi.org/10.1080/14686996.2017.1378060>_`
    """

    def __init__(self, period_tag=False, flatten=True):
        """Initialize the featurizer

        Args:
            period_tag (bool): In the original OFM, an element is represented
                    by a vector of length 32, where each element is 1 or 0,
                    which represents the valence subshell of the element.
                    With period_tag=True, the vector size is increased
                    to 39, where the 7 extra elements represent the period
                    of the element. Note lanthanides are treated as period 6,
                    actinides as period 7. Default False as in the original paper.
        """
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
        self.flatten = flatten

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
            elif el_struct[-1 - shell_num][0] < max_n - 1 and el_struct[-1 - shell_num][1] != "f":
                shell_num += 1
                continue
            elif el_struct[-1 - shell_num][0] < max_n and (
                el_struct[-1 - shell_num][1] != "d" and el_struct[-1 - shell_num][1] != "f"
            ):
                shell_num += 1
                continue
            curr_shell = el_struct[-1 - shell_num]
            if curr_shell[1] == "s":
                l = 0
            elif curr_shell[1] == "p":
                l = 1
            elif curr_shell[1] == "d":
                l = 2
            elif curr_shell[1] == "f":
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
            my_ohv[row + 31] = 1
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
            scale = other_site["weight"]
            other_atom = ohvs[other_site["site"].specie.Z]
            atom_ofm += other_atom.T * ref_atom * scale / site.distance(other_site["site"]) / ANG_TO_BOHR
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
        vnn = pmg_le.VoronoiNN(allow_pathological=True)
        if symm:
            symm_struct = SpacegroupAnalyzer(struct).get_symmetrized_structure()
            indices = [lst[0] for lst in symm_struct.equivalent_indices]
            counts = [len(lst) for lst in symm_struct.equivalent_indices]
        else:
            indices = [i for i in range(len(struct.sites))]
        for index in indices:
            ofms.append(self.get_single_ofm(struct.sites[index], vnn.get_nn_info(struct, index)))
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
        if self.flatten:
            return mean_ofm.A.flatten()
        else:
            return [mean_ofm.A]

    def feature_labels(self):
        if self.flatten:
            slabels = [f"s^{i}" for i in range(1, 3)]
            plabels = [f"p^{i}" for i in range(1, 7)]
            dlabels = [f"d^{i}" for i in range(1, 11)]
            flabels = [f"f^{i}" for i in range(1, 15)]
            labelset_1D = slabels + plabels + dlabels + flabels

            # account for period tags
            if self.size == 39:
                period_labels = [f"period {i}" for i in range(1, 8)]
                labelset_1D += period_labels

            labelset_2D = []
            for l1 in labelset_1D:
                for l2 in labelset_1D:
                    labelset_2D.append("OFM: " + l1 + " - " + l2)
            return labelset_2D
        else:
            return ["orbital field matrix"]

    def citations(self):
        return [
            "@article{LamPham2017,"
            "author = {{Lam Pham}, Tien and Kino, Hiori and Terakura, Kiyoyuki and "
            "Miyake, Takashi and Tsuda, Koji and Takigawa, Ichigaku and {Chi Dam}, Hieu},"
            "doi = {10.1080/14686996.2017.1378060},"
            "journal = {Science and Technology of Advanced Materials},"
            "month = {dec},"
            "number = {1},"
            "pages = {756--765},"
            r"publisher = {Taylor {\&} Francis},"
            "title = {{Machine learning reveals orbital interaction in materials}},"
            "url = {https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1378060},"
            "volume = {18},"
            "year = {2017}"
            "}"
        ]

    def implementors(self):
        return ["Kyle Bystrom", "Alex Dunn"]
