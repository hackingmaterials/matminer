"""
Construct ElM2D plot of a list of inorganic compostions via Element Movers Distance.

Copyright (C) 2021  Cameron Hargreaves

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

--------------------------------------------------------------------------------

Python Parser Source: https://github.com/Zapaan/python-chemical-formula-parser

Periodic table JSON data: https://github.com/Bowserinator/Periodic-Table-JSON,
updated to include the Pettifor number and modified Pettifor number from
https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011

Network simplex source modified to use numba from
https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/algorithms/flow/networksimplex.html#network_simplex

Requires umap which may be installed via:
    conda install -c conda-forge umap-learn
"""
from operator import attrgetter
from copy import deepcopy

from numba import cuda

from multiprocessing import cpu_count

import numpy as np
import pickle as pk

import umap

from dist_matrix.njit_dist_matrix_full import dist_matrix as cpu_dist_matrix

from matminer.featurizers.base import BaseFeaturizer

# overriden by ElM2D class if self.target is not None
use_cuda = cuda.is_available()
if use_cuda:
    target = "cuda"
else:
    target = "cpu"

if use_cuda:
    from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
else:
    gpu_dist_matrix = None


class ElM2DFeaturizer(BaseFeaturizer):
    """
    Create intercompound EMD distance matrix and embedding via list of formulas.

    Embedding types are:
        - PCA
        - UMAP
    """

    def __init__(
        self,
        formula_list=None,
        n_proc=None,
        n_components=2,
        verbose=True,
        chunksize=1,
        umap_kwargs={},
        emd_algorithm="wasserstein",
        target=None,
    ):
        """
        Initialize parameters for Element Mover's Distance.

        Parameters
        ----------
        formula_list : list of str, optional
            List of chemical formulas, by default None
        n_proc : int, optional
            Number of processors to use (deprecated), by default None
        n_components : int, optional
            Number of embedding dimensions, by default 2
        verbose : bool, optional
            Whether to output verbose information, by default True
        chunksize : int, optional
            Size of chunks for multiprocessing (deprecated), by default 1
        umap_kwargs : dict, optional
            Arguments to pass into umap_kwargs, by default {}
        emd_algorithm : str, optional
            How to compute the earth mover's distances, by default "wasserstein"
        target : str, optional
            Compute device to use: "cuda" or "cpu". If None, defaults to
            fit() "target". If fit() target value is also None, uses "cuda"
            if compatible GPU is available, otherwise "cpu", by default None
        """
        self.verbose = verbose

        if n_proc is None:
            self.n_proc = cpu_count()
        else:
            self.n_proc = n_proc

        self.formula_list = formula_list  # Input formulae
        # fmt: off
        # modified pettifor scale
        self.periodic_tab = {"D": 102, "T": 102, "H": 102, "He": 0, "Li": 11, 
        "Be": 76, "B": 85, "C": 86, "N": 87, "O": 96, "F": 101, 
        "Ne": 1, "Na": 10, "Mg": 72, "Al": 77, "Si": 84, "P": 88, 
        "S": 95, "Cl": 100, "Ar": 2, "K": 9, "Ca": 15, "Sc": 47, 
        "Ti": 50, "V": 53, "Cr": 54, "Mn": 71, "Fe": 70, "Co": 69, 
        "Ni": 68, "Cu": 67, "Zn": 73, "Ga": 78, "Ge": 83, "As": 89, 
        "Se": 94, "Br": 99, "Kr": 3, "Rb": 8, "Sr": 14, "Y": 20, "Zr": 48, 
        "Nb": 52, "Mo": 55, "Tc": 58, "Ru": 60, "Rh": 62, "Pd": 64, "Ag": 66, 
        "Cd": 74, "In": 79, "Sn": 82, "Sb": 90, "Te": 93, "I": 98, "Xe": 4, 
        "Cs": 7, "Ba": 13, "La": 31, "Ce": 30, "Pr": 29, "Nd": 28, "Pm": 27, 
        "Sm": 26, "Eu": 16, "Gd": 25, "Tb": 24, "Dy": 23, "Ho": 22, "Er": 21, 
        "Tm": 19, "Yb": 17, "Lu": 18, "Hf": 49, "Ta": 51, "W": 56, "Re": 57, 
        "Os": 59, "Ir": 61, "Pt": 63, "Au": 65, "Hg": 75, "Tl": 80, "Pb": 81, 
        "Bi": 91, "Po": 92, "At": 97, "Rn": 5, "Fr": 6, "Ra": 12, "Ac": 32, 
        "Th": 33, "Pa": 34, "U": 35, "Np": 36, "Pu": 37, "Am": 38, "Cm": 39, 
        "Bk": 40, "Cf": 41, "Es": 42, "Fm": 43, "Md": 44, "No": 45, "Lr": 46, 
        "Rf": 0, "Db": 0, "Sg": 0, "Bh": 0, "Hs": 0, "Mt": 0, "Ds": 0, "Rg": 0, 
        "Cn": 0, "Nh": 0, "Fl": 0, "Mc": 0, "Lv": 0, "Ts": 0, "Og": 0, "Uue": 0}
        # fmt: on

        self.chunksize = chunksize

        self.umap_kwargs = umap_kwargs

        self.umap_kwargs["n_components"] = n_components
        self.umap_kwargs["metric"] = "precomputed"

        self.input_mat = None  # Pettifor vector representation of formula
        self.embedder = None  # For accessing UMAP object
        self.embedding = None  # Stores the last embedded coordinates
        self.dm = None  # Stores distance matrix
        self.emd_algorithm = emd_algorithm
        self.target = target  # "cuda" or "cpu"

    def fit(self, X, target=None):
        """
        Construct and store an ElMD distance matrix.

        Take an input vector, either of a precomputed distance matrix, or
        an iterable of strings of composition formula, construct an ElMD distance
        matrix and store to self.dm.

        Parameters
        ----------
        X : list of str OR 2D array
            A list of compound formula strings, or a precomputed distance matrix. If
            using a precomputed distance matrix, ensure self.metric == "precomputed"


        Returns
        -------
        None.

        """
        self.formula_list = X
        n = len(X)

        if self.verbose:
            print(f"Fitting {self.metric} kernel matrix")
        if self.metric == "precomputed":
            self.dm = X

        else:
            if self.verbose:
                print("Constructing distances")
            elif self.emd_algorithm == "wasserstein":
                self.dm = self.EM2D(X, X, target=target)

    def fit_transform(self, X, y=None, how="UMAP", n_components=2, target=None):
        """
        Successively call fit and transform.

        Parameters
        ----------
        X : list of str
            Compositions to embed.
        y : 1D numerical array, optional
            Target values to use for supervised UMAP embedding. The default is None.
        how : str, optional
            How to perform embedding ("UMAP" or "PCA"). The default is "UMAP".
        n_components : int, optional
            Number of dimensions to embed to. The default is 2.

        Returns
        -------
        embedding : TYPE
            DESCRIPTION.

        """
        self.fit(X, target=target)
        embedding = self.transform(
            how=how, n_components=self.umap_kwargs["n_components"], y=y
        )
        return embedding

    def transform(self, how="UMAP", n_components=2, y=None):
        """
        Call the selected embedding method (UMAP or PCA) and embed.

        Parameters
        ----------
        how : str, optional
            How to perform embedding ("UMAP" or "PCA"). The default is "UMAP".
            The default is "UMAP".
        n_components : int, optional
            Number of dimensions to embed to. The default is 2.
        y : 1D numerical array, optional
            Target values to use for supervised UMAP embedding. The default is None.

        Returns
        -------
        2D array
            UMAP or PCA embedding.

        """
        if self.dm is None:
            print("No distance matrix computed, run fit() first")
            return

        n = self.umap_kwargs["n_components"]
        if how == "UMAP":
            if y is None:
                if self.verbose:
                    print(f"Constructing UMAP Embedding to {n} dimensions")
                self.embedder = umap.UMAP(**self.umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm)

            else:
                y = y.to_numpy(dtype=float)
                if self.verbose:
                    print(
                        f"Constructing UMAP Embedding to {n} dimensions, with \
                            a targeted embedding"
                    )
                self.embedder = umap.UMAP(**self.umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm, y)

        elif how == "PCA":
            if self.verbose:
                print(f"Constructing PCA Embedding to {n} dimensions")
            self.embedding = self.PCA(n_components=self.umap_kwargs["n_components"])
            if self.verbose:
                print("Finished Embedding")

        return self.embedding

    def EM2D(self, formulas, formulas2=None, target=None):
        """
        Earth Mover's 2D distances. See also EMD.

        Parameters
        ----------
        formulas : list of str
            First list of formulas for which to compute distances. If only formulas
            is specified, then a `pdist`-like array is returned, i.e. pairwise
            distances within a single set.
        formulas2 : list of str, optional
                Second list of formulas, which if specified, causes `cdist`-like
                behavior (i.e. pairwise distances between two sets).

        Returns
        -------
        2D array
            Pairwise distances.

        """
        isXY = formulas2 is None
        # E = ElMD(metric=self.metric)

        def gen_ratio_vector(comp):
            """Create a numpy array from a composition dictionary."""
            if isinstance(comp, str):
                comp = self._parse_formula(comp)
                comp = self._normalise_composition(comp)

            sorted_keys = sorted(comp.keys())
            comp_labels = [self._get_position(k) for k in sorted_keys]
            comp_ratios = [comp[k] for k in sorted_keys]

            indices = np.array(comp_labels, dtype=np.int64)
            ratios = np.array(comp_ratios, dtype=np.float64)

            numeric = np.zeros(shape=len(E.periodic_tab), dtype=np.float64)
            numeric[indices] = ratios

            return numeric

        def gen_ratio_vectors(comps):
            return np.array([gen_ratio_vector(comp) for comp in comps])

        U_weights = gen_ratio_vectors(formulas)
        if isXY:
            V_weights = gen_ratio_vectors(formulas2)

        self.lookup, self.periodic_tab = attrgetter("lookup", "periodic_tab")(E)

        def get_mod_petti(x):
            mod_petti = [
                self.periodic_tab[self.lookup[a]] if b > 0 else 0
                for a, b in enumerate(x)
            ]  # FIXME: apparently might output an array of strings
            return mod_petti

        def get_mod_pettis(X):
            # NOTE: in case output as strings, convert to float
            mod_pettis = np.array([get_mod_petti(x) for x in X]).astype(float)
            return mod_pettis

        U = get_mod_pettis(U_weights)
        if isXY:
            V = get_mod_pettis(V_weights)

        # decide whether to use cpu or cuda version
        if target is None:
            if (self.target is None or not cuda.is_available()) or self.target == "cpu":
                target = "cpu"
            elif self.target == "cuda" or cuda.is_available():
                target = "cuda"

        if isXY:
            if target == "cpu":
                distances = cpu_dist_matrix(
                    U,
                    V=V,
                    U_weights=U_weights,
                    V_weights=V_weights,
                    metric="wasserstein",
                )
            elif target == "cuda":
                distances = gpu_dist_matrix(
                    U,
                    V=V,
                    U_weights=U_weights,
                    V_weights=V_weights,
                    metric="wasserstein",
                )
        else:
            if target == "cpu":
                distances = cpu_dist_matrix(
                    U, U_weights=U_weights, metric="wasserstein"
                )
            elif target == "cuda":
                distances = gpu_dist_matrix(
                    U, U_weights=U_weights, metric="wasserstein"
                )

        # package
        self.U = U
        self.U_weights = U_weights

        if isXY:
            self.V = V
            self.V_weights = V_weights

        return distances

    def PCA(self, n_components=5):
        """
        Perform multidimensional scaling (MDS) on a matrix of interpoint distances.

        This finds a set of low dimensional points that have similar interpoint
        distances.
        Source: https://github.com/stober/mds/blob/master/src/mds.py
        """
        if self.dm == []:
            raise Exception(
                "No distance matrix computed, call fit_transform with a list of \
                    compositions, or load a saved matrix with load_dm()"
            )

        (n, n) = self.dm.shape

        if self.verbose:
            print(f"Constructing {n}x{n_components} Gram matrix")
        E = -0.5 * self.dm ** 2

        # Use this matrix to get column and row means
        Er = np.mat(np.mean(E, 1))
        Es = np.mat(np.mean(E, 0))

        # From Principles of Multivariate Analysis: A User's Perspective (page 107).
        F = np.array(E - np.transpose(Er) - Es + np.mean(E))

        if self.verbose:
            print("Computing Eigen Decomposition")
        [U, S, V] = np.linalg.svd(F)

        Y = U * np.sqrt(S)

        if self.verbose:
            print("PCA Projected Points Computed")
        self.mds_points = Y

        return Y[:, :n_components]

    def _parse(self, formula):
        """
        Return the molecule dict and length of parsed part.

        Recurse on opening brackets to parse the subpart and
        return on closing ones because it is the end of said subpart.
        """
        q = []
        mol = {}
        i = 0

        while i < len(formula):
            # Using a classic loop allow for manipulating the cursor
            token = formula[i]

            if token in self.CLOSERS:
                # Check for an index for this part
                m = re.match("\d+\.*\d*|\.\d*", formula[i + 1 :])
                if m:
                    weight = float(m.group(0))
                    i += len(m.group(0))
                else:
                    weight = 1

                submol = self._dictify(re.findall(self.ATOM_REGEX, "".join(q)))
                return self._fuse(mol, submol, weight), i

            elif token in self.OPENERS:
                submol, l = self._parse(formula[i + 1 :])
                mol = self._fuse(mol, submol)
                # skip the already read submol
                i += l + 1
            else:
                q.append(token)

            i += 1

        # Fuse in all that's left at base level
        return (
            self._fuse(mol, self._dictify(re.findall(self.ATOM_REGEX, "".join(q)))),
            i,
        )

    def _parse_formula(self, formula):
        """Parse the formula and return a dict with occurences of each atom."""
        if not self._is_balanced(formula):
            raise ValueError("Your brackets not matching in pairs ![{]$[&?)]}!]")

        return self._parse(formula)[0]

    def _normalise_composition(self, input_comp):
        """Sum up the numbers in our counter to get total atom count."""
        composition = deepcopy(input_comp)
        # check it has been processed
        if isinstance(composition, str):
            composition = self._parse_formula(composition)

        atom_count = sum(composition.values(), 0.0)

        for atom in composition:
            composition[atom] /= atom_count

        return composition

    def _get_position(self, element):
        """
        Return either the x, y coordinate of an elements position, or the
        x-coordinate on the Pettifor numbering system as a 2-dimensional
        """
        keys = list(self.periodic_tab.keys())

        try:
            atomic_num = keys.index(element)
            return atomic_num

        except:
            if self.strict_parsing:
                raise KeyError(
                    f"One of the elements in {self.composition} is not in the {self.metric} dictionary. Try a different representation or use strict_parsing=False"
                )
            else:
                return -1

    def __repr__(self):
        """Summary of ElM2D object: length, diversity, and max distance if dm exists."""
        if self.dm is not None:
            return f"ElM2D(size={len(self.formula_list)},  \
                chemical_diversity={np.mean(self.dm)} +/- {np.std(self.dm)}, \
                    maximal_distance={np.max(self.dm)})"
        else:
            return "ElM2D()"

    def export_dm(self, path):
        """Export distance matrix as .csv to path."""
        np.savetxt(path, self.dm, delimiter=",")

    def import_dm(self, path):
        """Import distance matrix from .csv file located at path."""
        self.dm = np.loadtxt(path, delimiter=",")

    def export_embedding(self, path):
        """Export embedding as .csv file to path."""
        np.savetxt(path, self.embedding, delimiter=",")

    def import_embedding(self, path):
        """Import embedding from .csv file located at path."""
        self.embedding = np.loadtxt(path, delimiter=",")

    def _pool_featurize(self, comp):
        """Extract the feature vector for a given composition (comp)."""
        return ElMD(comp, metric=self.metric).feature_vector

    def featurize(self, formula_list=None, how="mean"):
        """Featurize a list of formulas."""
        if formula_list is None and self.formula_list is None:
            raise Exception("You must enter a list of compositions first")

        elif formula_list is None:
            formula_list = self.formula_list

        elif self.formula_list is None:
            self.formula_list = formula_list

        print(
            f"Constructing compositionally weighted {self.metric} feature vectors \
                for each composition"
        )
        vectors = map(self._pool_featurize, formula_list)

        print("Complete")

        return np.array(vectors)

    def save(self, filepath):
        """
        Save all variables except for the distance matrix.

        Parameters
        ----------
        filepath : str
            Filepath for which to save the pickle.

        Returns
        -------
        None.

        """
        save_dict = {k: v for k, v in self.__dict__.items()}
        f_handle = open(filepath + ".pk", "wb")
        pk.dump(save_dict, f_handle)
        f_handle.close()

    def load(self, filepath):
        """
        Load variables from pickle file.

        Parameters
        ----------
        filepath : str
            Filepath for which to load the pickle.

        Returns
        -------
        None.

        """
        f_handle = open(filepath + ".pk", "rb")
        load_dict = pk.load(f_handle)
        f_handle.close()

        for k, v in load_dict.items():
            self.__dict__[k] = v
