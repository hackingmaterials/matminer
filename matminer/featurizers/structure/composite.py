"""
Structure featurizers producing more than one kind of structue feature data.
"""
import os
import math
import json
import itertools
from operator import itemgetter

import numpy as np
from pymatgen.core import Structure, Lattice

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.structure.order import DensityFeatures


class JarvisCFID(BaseFeaturizer):
    """
    Classical Force-Field Inspired Descriptors (CFID) from Jarvis-ML.

    Chemo-structural descriptors from five different sub-methods,cincluding
    pairwise radial, nearest neighbor, bond-angle, dihedral-angle and
    core-charge distributions. With all descriptors enabled, there are 1,557
    features per structure.

    Adapted from the nist/jarvis package hosted at:
    https://github.com/usnistgov/jarvis

    Find details at: https://journals.aps.org/prmaterials/abstract/10.1103/
        PhysRevMaterials.2.083801

    Args/Features:
        use_cell (bool): Use structure cell descriptors (4 features, based
            on DensityFeatures and log volume per atom).
        use_chem (bool): Use chemical composition descriptors (438 features)
        use_chg (bool): Use core charge descriptors (378 features)
        use_adf (bool): Use angular distribution function (179 features x 2, one
             set of features for each cutoff).
        use_rdf (bool): Use radial distribution function (100 features)
        use_ddf (bool): Use dihedral angle distribution function (179 features)
        use_nn (bool): Use nearest neighbors (100 descriptors)
    """

    def __init__(
        self,
        use_cell=True,
        use_chem=True,
        use_chg=True,
        use_rdf=True,
        use_adf=True,
        use_ddf=True,
        use_nn=True,
    ):

        self.use_cell = use_cell
        self.use_chem = use_chem
        self.use_chg = use_chg
        self.use_adf = use_adf
        self.use_rdf = use_rdf
        self.use_ddf = use_ddf
        self.use_nn = use_nn

        basedir = os.path.dirname(os.path.realpath(__file__))
        jdir = os.path.join(basedir, "../../utils/data_files/jarvis/")
        chgfile = os.path.join(jdir, "element_charges.json")
        chemfile = os.path.join(jdir, "element_chem.json")

        with open(chgfile, "r") as f:
            self.el_chrg_json = json.load(f)
        with open(chemfile, "r") as f:
            self.el_chem_json = json.load(f)

        labels = []
        if self.use_chem:
            labels += list(["jml_" + s for s in self.el_chem_json["Al"].keys()])
        if self.use_cell:
            labels += ["jml_pack_frac", "jml_vpa", "jml_density", "jml_log_vpa"]
        if self.use_chg:
            labels += ["jml_mean_charge_{}".format(i) for i in range(1, 379)]
        if self.use_rdf:
            labels += ["jml_rdf_{}".format(i) for i in range(1, 101)]
        if self.use_adf:
            for lvl in [1, 2]:
                labels += ["jml_adf{}_{}".format(lvl, i) for i in range(1, 180)]
        if self.use_ddf:
            labels += ["jml_ddf_{}".format(i) for i in range(1, 180)]
        if self.use_nn:
            labels += ["jml_nn_{}".format(i) for i in range(1, 101)]
        self.labels = labels

    def featurize(self, s):
        """
        Get chemo-structural CFID decriptors

        Args:
            s: Structure object

        Returns:
              (np.ndarray) Final descriptors
        """
        s = self._clean_structure(s)
        descriptors = []
        el_dict = s.composition.get_el_amt_dict()

        if self.use_chem:
            arr = []
            for k in el_dict.keys():
                des = self.get_chem(k)
                arr.append(des)
            mean_chem = np.mean(arr, axis=0)
            descriptors.append(mean_chem)

        if self.use_cell:
            log_vpa = round(math.log(float(s.volume) / float(s.composition.num_atoms)), 5)
            dffer = DensityFeatures(desired_features=["packing fraction", "vpa", "density"])
            feats = dffer.featurize(s)
            cell = np.array([log_vpa] + feats)
            descriptors.append(cell)

        if self.use_chg:
            chgarr = []
            for k in el_dict.keys():
                chg = self.get_chg(k)
                chgarr.append(chg)
            mean_chg = np.mean(chgarr, axis=0)
            descriptors.append(mean_chg)

        if any([self.use_rdf, self.use_adf, self.use_ddf, self.use_nn]):
            adf_1, adf_2, ddf, rdf, nn = self.get_distributions(structure=s)
            # 1st and 2nd cutoff ADFs
            adf_1 = np.array(adf_1)
            adf_2 = np.array(adf_2)
            rdf = np.array(rdf)
            ddf = np.array(ddf)
            nn = np.array(nn)
            if self.use_rdf:
                descriptors.append(rdf)
            if self.use_adf:
                descriptors.append(adf_1)
                descriptors.append(adf_2)
            if self.use_ddf:
                descriptors.append(ddf)
            if self.use_nn:
                descriptors.append(nn)
        flat = list(itertools.chain.from_iterable(descriptors))
        return np.array(flat).astype(float)

    def feature_labels(self):
        return self.labels

    def get_distributions(self, structure, c_size=10.0, max_cut=5.0):
        """
        Get radial and angular distribution functions

        Args:
            structure: Structure object
            c_size: max. cell size
            max_cut: max. bond cut-off for angular distribution
        Retruns:
             adfa, adfb, ddf, rdf, bondo
             Angular distribution upto first cut-off
             Angular distribution upto second cut-off
             Dihedral angle distribution upto first cut-off
             Radial distribution funcion
             Bond order distribution
        """
        x, y, z = self._get_rdf(structure)
        arr = []
        for i, j in zip(x, z):
            if j > 0.0:
                arr.append(i)
        box = structure.lattice.matrix
        rcut_buffer = 0.11
        io1, io2, io3 = 0, 1, 2
        delta = arr[io2] - arr[io1]
        while delta < rcut_buffer and arr[io2] < max_cut:
            io1 = io1 + 1
            io2 = io2 + 1
            io3 = io3 + 1
            delta = arr[io2] - arr[io1]
        rcut1 = (arr[io2] + arr[io1]) / float(2.0)
        rcut = self._cutoff_from_combinations(structure=structure)
        delta = arr[io3] - arr[io2]
        while delta < rcut_buffer and arr[io3] < max_cut and arr[io2] < max_cut:
            io2 = io2 + 1
            io3 = io3 + 1
            delta = arr[io3] - arr[io2]
        rcut2 = float(arr[io3] + arr[io2]) / float(2.0)
        dim1 = int(float(c_size) / float(max(abs(box[0])))) + 1
        dim2 = int(float(c_size) / float(max(abs(box[1])))) + 1
        dim3 = int(float(c_size) / float(max(abs(box[2])))) + 1
        dim = [dim1, dim2, dim3]
        dim = np.array(dim)
        coords = structure.frac_coords
        lat = np.zeros((3, 3))
        lat[0][0] = dim[0] * box[0][0]
        lat[0][1] = dim[0] * box[0][1]
        lat[0][2] = dim[0] * box[0][2]
        lat[1][0] = dim[1] * box[1][0]
        lat[1][1] = dim[1] * box[1][1]
        lat[1][2] = dim[1] * box[1][2]
        lat[2][0] = dim[2] * box[2][0]
        lat[2][1] = dim[2] * box[2][1]
        lat[2][2] = dim[2] * box[2][2]
        all_symbs = [i.symbol for i in structure.species]
        nat = len(coords)
        new_nat = nat * dim[0] * dim[1] * dim[2]
        new_coords = np.zeros((new_nat, 3))
        new_symbs = []
        count = 0
        for i in range(nat):
            for j in range(dim[0]):
                for k in range(dim[1]):
                    for l in range(dim[2]):
                        new_coords[count][0] = (coords[i][0] + j) / float(dim[0])
                        new_coords[count][1] = (coords[i][1] + k) / float(dim[1])
                        new_coords[count][2] = (coords[i][2] + l) / float(dim[2])
                        new_symbs.append(all_symbs[i])
                        count = count + 1
        nat = new_nat
        coords = new_coords
        znm = 0
        nn = np.zeros((nat), dtype="int")
        max_n = 500  # maximum number of neighbors
        dist = np.zeros((max_n, nat))
        nn_id = np.zeros((max_n, nat), dtype="int")
        bondx = np.zeros((max_n, nat))
        bondy = np.zeros((max_n, nat))
        bondz = np.zeros((max_n, nat))
        dim05 = [float(1 / 2.0) for i in dim]
        for i in range(nat):
            for j in range(i + 1, nat):
                diff = coords[i] - coords[j]
                for v in range(3):
                    if np.fabs(diff[v]) >= dim05[v]:
                        diff[v] = diff[v] - np.sign(diff[v])
                new_diff = np.dot(diff, lat)
                dd = np.linalg.norm(new_diff)
                if dd < rcut and dd >= 0.1:
                    nn_index = nn[i]  # index of the neighbor
                    nn[i] = nn[i] + 1
                    dist[nn_index][i] = dd  # nn_index counter id
                    nn_id[nn_index][i] = j  # exact id
                    bondx[nn_index][i] = new_diff[0]
                    bondy[nn_index][i] = new_diff[1]
                    bondz[nn_index][i] = new_diff[2]
                    nn_index1 = nn[j]  # index of the neighbor
                    nn[j] = nn[j] + 1
                    dist[nn_index1][j] = dd  # nn_index counter id
                    nn_id[nn_index1][j] = i  # exact id
                    bondx[nn_index1][j] = -new_diff[0]
                    bondy[nn_index1][j] = -new_diff[1]
                    bondz[nn_index1][j] = -new_diff[2]
        ang_at = {}
        for i in range(nat):
            for in1 in range(nn[i]):
                for in2 in range(in1 + 1, nn[i]):
                    nm = dist[in1][i] * dist[in2][i]
                    if nm != 0:
                        rrx = bondx[in1][i] * bondx[in2][i]
                        rry = bondy[in1][i] * bondy[in2][i]
                        rrz = bondz[in1][i] * bondz[in2][i]
                        cos = float(rrx + rry + rrz) / float(nm)
                        if cos <= -1.0:
                            cos = cos + 0.000001
                        if cos >= 1.0:
                            cos = cos - 0.000001
                        deg = math.degrees(math.acos(cos))
                        ang_at.setdefault(round(deg, 3), []).append(i)
                    else:
                        znm = znm + 1
        angs = np.array([float(i) for i in ang_at.keys()])
        norm = np.array([float(len(i)) / float(len(set(i))) for i in ang_at.values()])
        binrng = np.arange(1, 181.0, 1)
        ang_hist1, _ = np.histogram(angs, weights=norm, bins=binrng, density=False)
        # 1st neighbors
        nn = np.zeros((nat), dtype="int")
        max_n = 500  # maximum number of neighbors
        dist = np.zeros((max_n, nat))
        nn_id = np.zeros((max_n, nat), dtype="int")
        bondx = np.zeros((max_n, nat))
        bondy = np.zeros((max_n, nat))
        bondz = np.zeros((max_n, nat))
        dim05 = [float(1 / 2.0) for i in dim]
        for i in range(nat):
            for j in range(i + 1, nat):
                diff = coords[i] - coords[j]
                for v in range(3):
                    if np.fabs(diff[v]) >= dim05[v]:
                        diff[v] = diff[v] - np.sign(diff[v])
                new_diff = np.dot(diff, lat)
                dd = np.linalg.norm(new_diff)
                if dd < rcut1 and dd >= 0.1:
                    nn_index = nn[i]  # index of the neighbor
                    nn[i] = nn[i] + 1
                    dist[nn_index][i] = dd  # nn_index counter id
                    nn_id[nn_index][i] = j  # exact id
                    bondx[nn_index, i] = new_diff[0]
                    bondy[nn_index, i] = new_diff[1]
                    bondz[nn_index, i] = new_diff[2]
                    nn_index1 = nn[j]  # index of the neighbor
                    nn[j] = nn[j] + 1
                    dist[nn_index1][j] = dd  # nn_index counter id
                    nn_id[nn_index1][j] = i  # exact id
                    bondx[nn_index1, j] = -new_diff[0]
                    bondy[nn_index1, j] = -new_diff[1]
                    bondz[nn_index1, j] = -new_diff[2]
        dih_at = {}
        for i in range(nat):
            for in1 in range(nn[i]):
                j1 = nn_id[in1][i]
                if j1 > i:
                    # angles between i,j, k=nn(i), l=nn(j)
                    # all other nn of i that are not j
                    for in2 in range(nn[i]):
                        j2 = nn_id[in2][i]
                        if j2 != j1:
                            # all other nn of j that are not i
                            for in3 in range(nn[j1]):
                                j3 = nn_id[in3][j1]
                                if j3 != i:
                                    v1, v2, v3 = [], [], []
                                    v1.append(bondx[in2][i])
                                    v1.append(bondy[in2][i])
                                    v1.append(bondz[in2][i])
                                    v2.append(-bondx[in1][i])
                                    v2.append(-bondy[in1][i])
                                    v2.append(-bondz[in1][i])
                                    v3.append(-bondx[in3][j1])
                                    v3.append(-bondy[in3][j1])
                                    v3.append(-bondz[in3][j1])
                                    v23 = np.cross(v2, v3)
                                    v12 = np.cross(v1, v2)
                                    theta = math.degrees(
                                        math.atan2(
                                            np.linalg.norm(v2) * np.dot(v1, v23),
                                            np.dot(v12, v23),
                                        )
                                    )
                                    if theta < 0.00001:
                                        theta = -theta
                                    dih_at.setdefault(round(theta, 3), []).append(i)
        dih = np.array([float(i) for i in dih_at.keys()])
        norm = np.array([float(len(i)) / float(len(set(i))) for i in dih_at.values()])
        dih_hist1, _ = np.histogram(dih, weights=norm, bins=binrng, density=False)
        # 2nd neighbors
        znm = 0
        nn = np.zeros((nat), dtype="int")
        max_n = 250  # maximum number of neighbors
        dist = np.zeros((max_n, nat))
        nn_id = np.zeros((max_n, nat), dtype="int")
        bondx = np.zeros((max_n, nat))
        bondy = np.zeros((max_n, nat))
        bondz = np.zeros((max_n, nat))
        dim05 = [float(1 / 2.0) for _ in dim]
        for i in range(nat):
            for j in range(i + 1, nat):
                diff = coords[i] - coords[j]
                for v in range(3):
                    if np.fabs(diff[v]) >= dim05[v]:
                        diff[v] = diff[v] - np.sign(diff[v])
                new_diff = np.dot(diff, lat)
                dd = np.linalg.norm(new_diff)
                if dd < rcut2 and dd >= 0.1:
                    nn_index = nn[i]  # index of the neighbor
                    nn[i] = nn[i] + 1
                    dist[nn_index][i] = dd  # nn_index counter id
                    nn_id[nn_index][i] = j  # exact id
                    bondx[nn_index, i] = new_diff[0]
                    bondy[nn_index, i] = new_diff[1]
                    bondz[nn_index, i] = new_diff[2]
                    nn_index1 = nn[j]  # index of the neighbor
                    nn[j] = nn[j] + 1
                    dist[nn_index1][j] = dd  # nn_index counter id
                    nn_id[nn_index1][j] = i  # exact id
                    bondx[nn_index1, j] = -new_diff[0]
                    bondy[nn_index1, j] = -new_diff[1]
                    bondz[nn_index1, j] = -new_diff[2]
        ang_at = {}
        for i in range(nat):
            for in1 in range(nn[i]):
                for in2 in range(in1 + 1, nn[i]):
                    nm = dist[in1][i] * dist[in2][i]
                    if nm != 0:
                        rrx = bondx[in1][i] * bondx[in2][i]
                        rry = bondy[in1][i] * bondy[in2][i]
                        rrz = bondz[in1][i] * bondz[in2][i]
                        cos = float(rrx + rry + rrz) / float(nm)
                        if cos <= -1.0:
                            cos = cos + 0.000001
                        if cos >= 1.0:
                            cos = cos - 0.000001
                        deg = math.degrees(math.acos(cos))
                        ang_at.setdefault(round(deg, 3), []).append(i)
                    else:
                        znm = znm + 1
        angs = np.array([float(i) for i in ang_at.keys()])
        norm = np.array([float(len(i)) / float(len(set(i))) for i in ang_at.values()])
        ang_hist2, _ = np.histogram(angs, weights=norm, bins=binrng, density=False)
        # adf_1, adf_2, ddf, rdf, bond-order/nn
        return ang_hist1, ang_hist2, dih_hist1, y, z

    def get_chg(self, element):
        """
        Get charge descriptors for an element

        Args:
             element: element name
        Returns:
               arr: descriptor array values
        """
        try:
            arr = self.el_chrg_json[element][0][1]
        except (KeyError, IndexError):
            arr = []
        return arr

    def get_chem(self, element):
        """
        Get chemical descriptors for an element

        Args:
             element: element name
        Returns:
               arr: descriptor array value
        """
        try:
            d = self.el_chem_json[element]
            arr = []
            for v in d.values():
                arr.append(v)
            arr = np.array(arr).astype(float)
        except (KeyError, IndexError):
            arr = []
        return arr

    def citations(self):
        return [
            "@article{PhysRevMaterials.2.083801, "
            "title = {Machine learning with force-field-inspired "
            "descriptors for materials: Fast screening and mapping "
            "energy landscape},"
            "author = {Choudhary, Kamal and DeCost, Brian and Tavazza, "
            "Francesca},"
            "journal = {Phys. Rev. Materials},"
            "volume = {2},"
            "issue = {8},"
            "pages = {083801},"
            "numpages = {8},"
            "year = {2018},"
            "month = {Aug},"
            "publisher = {American Physical Society},"
            "doi = {10.1103/PhysRevMaterials.2.083801}, "
            "url = "
            "{https://link.aps.org/doi/10.1103/PhysRevMaterials.2.083801}}"
        ]

    def implementors(self):
        return ["Alex Dunn", "Kamal Choudhary"]

    def _cutoff_from_combinations(self, structure=None, cutoff=10.0):
        """
        Get the cutoff, ensuring that no elemental combination is left out.

        Args:
             structure (Structure): A pymatgen structure obj
             cutoff (float): maximum cutoff in Angstrom
        Returns:
            (float) max-cutoff in Angstroms to ensure all the element
                combinations are included
        """
        neighbors_lst = structure.get_all_neighbors(cutoff)
        comb = self._element_combinations(structure=structure)
        info = {}
        for c in comb:
            for i, ii in enumerate(neighbors_lst):
                for j in ii:
                    comb1 = str(structure[i].specie) + str("-") + str(j[0].specie)
                    comb2 = str(j[0].specie) + str("-") + str(structure[i].specie)
                    if comb1 == c or comb2 == c:
                        info.setdefault(c, []).append(j[1])
        for i in info.items():
            i[1].sort()
        cut_off = {}
        for i, j in info.items():
            cut_off[i] = self._flatten(arr=j, tol=0.1)
        return max(cut_off.items(), key=itemgetter(1))[1]

    @staticmethod
    def _element_combinations(structure):
        """
        Get element combinations for a Structure object

        Args:
            structure: Structure object
        Returns:
               comb: combinations
        """
        sym = structure.symbol_set
        tmp = map("-".join, itertools.product(sym, repeat=2))
        comb = list(set([str("-".join(sorted(i.split("-")))) for i in tmp]))
        return comb

    @staticmethod
    def _get_rdf(structure=None, cutoff=10.0, intvl=0.1):
        """
        Get total radial distribution function

        Args:
             structure (Structure): pymatgen structure object
             cutoff (float): Maximum distance for binning
             intvl (float): Bin size
        Returns:
               bins (np.array): The bins of the distribution
               dist (np.array): The distribution
               scaled_dist (np.array): The scaled distribution
        """
        neighbors_lst = structure.get_all_neighbors(cutoff)
        mapper = map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)
        all_distances = np.concatenate(tuple(mapper))
        binrng = np.arange(0, cutoff + intvl, intvl)
        # equivalent to bond-order
        dist_hist, dist_bins = np.histogram(all_distances, bins=binrng, density=False)
        shell_vol = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = structure.num_sites / structure.volume
        rdf = dist_hist / shell_vol / number_density / len(neighbors_lst)
        bins = dist_bins[:-1]
        dist = [round(i, 4) for i in rdf]
        scaled_dist = dist_hist / float(len(structure))
        return bins, dist, scaled_dist

    @staticmethod
    def _flatten(arr, tol=0.1):
        """
        Determine first cut-off

        Args:
             arr: array
             tol: toelrance
        Return:
              rcut: cut-off for a given tolerance tol,
              because sometimes RDF peaks could be very close
        """
        rcut_buffer = tol
        io1, io2, io3 = 0, 1, 2
        delta = arr[io2] - arr[io1]
        while delta < rcut_buffer and io3 < len(arr):
            io1 = io1 + 1
            io2 = io2 + 1
            io3 = io3 + 1
            delta = arr[io2] - arr[io1]
        rcut = (arr[io2] + arr[io1]) / float(2.0)
        return rcut

    @staticmethod
    def _clean_structure(s=None, tol=8.0):
        """
        Check if there is vacuum, if so get actual size of the structure
        and the add vaccum of size tol to make sure structures
        are independent of user defined vacuum

        Args:
             s: Structure object
             tol: vacuum tolerance
        Returns:
               s: re-structure structure with tol vacuum
        """
        coords = s.cart_coords
        range_x = max(coords[:, 0]) - min(coords[:, 0])
        range_y = max(coords[:, 1]) - min(coords[:, 1])
        range_z = max(coords[:, 2]) - min(coords[:, 2])
        a = s.lattice.matrix[0][0]
        b = s.lattice.matrix[1][1]
        c = s.lattice.matrix[2][2]
        if abs(a - range_x) > tol:
            a = range_x + tol
        if abs(b - range_y) > tol:
            b = range_y + tol
        if abs(c - range_z) > tol:
            c = range_z + tol
        arr = Lattice(
            [
                [a, s.lattice.matrix[0][1], s.lattice.matrix[0][2]],
                [s.lattice.matrix[1][0], b, s.lattice.matrix[1][2]],
                [s.lattice.matrix[2][0], s.lattice.matrix[2][1], c],
            ]
        )
        s = Structure(arr, s.species, coords, coords_are_cartesian=True)
        s.remove_oxidation_states()
        return s
