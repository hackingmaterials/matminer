"""
Structure featurizers based on bonding.
"""
import itertools
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from pymatgen.core import Structure
from pymatgen.analysis import bond_valence
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
import pymatgen.analysis.local_env as pmg_le

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_all_nearest_neighbors
from matminer.utils.data import IUCrBondValenceData
from matminer.featurizers.structure.matrix import SineCoulombMatrix


class BondFractions(BaseFeaturizer):
    """
    Compute the fraction of each bond in a structure, based on NearestNeighbors.

    For example, in a structure with 2 Li-O bonds and 3 Li-P bonds:

    Li-0: 0.4
    Li-P: 0.6

    Features:

    BondFractions must be fit with iterable of structures before featurization in
    order to define the allowed bond types (features). To do this, pass a list
    of allowed_bonds. Otherwise, fit based on a list of structures. If
    allowed_bonds is defined and BondFractions is also fit, the intersection
    of the two lists of possible bonds is used.

    For dataframes containing structures of various compositions, a unified
    dataframe is returned which has the collection of all possible bond types
    gathered from all structures as columns. To approximate bonds based on
    chemical rules (ie, for a structure which you'd like to featurize but has
    bonds not in the allowed set), use approx_bonds = True.

    BondFractions is based on the "sum over bonds" in the Bag of Bonds approach,
    based on a method by Hansen et. al "Machine Learning Predictions of Molecular
    Properties: Accurate Many-Body Potentials and Nonlocality in Chemical Space"
    (2015).

    Args:
        nn (NearestNeighbors): A Pymatgen nearest neighbors derived object. For
            example, pymatgen.analysis.local_env.VoronoiNN().
        bbv (float): The 'bad bond values', values substituted for
            structure-bond combinations which can not physically exist, but
            exist in the unified dataframe. For example, if a dataframe contains
            structures of BaLiP and BaTiO3, determines the value to place in
            the Li-P column for the BaTiO3 row; by default, is 0.
        no_oxi (bool): If True, the featurizer will be agnostic to oxidation
            states, which prevents oxidation states from  differentiating
            bonds. For example, if True, Ca - O is identical to Ca2+ - O2-,
            Ca3+ - O-, etc., and all of them will be included in Ca - O column.
        approx_bonds (bool): If True, approximates the fractions of bonds not
            in allowed_bonds (forbidden bonds) with similar allowed bonds.
            Chemical rules are used to determine which bonds are most 'similar';
            particularly, the Euclidean distance between the 2-tuples of the
            bonds in Mendeleev no. space is minimized for the approximate
            bond chosen.
        token (str): The string used to separate species in a bond, including
            spaces. The token must contain at least one space and cannot have
            alphabetic characters in it, and should be padded by spaces. For
            example, for the bond Cs+ - Cl-, the token is ' - '. This determines
            how bonds are represented in the dataframe.
        allowed_bonds ([str]): A listlike object containing bond types as
            strings. For example, Cs - Cl, or Li+ - O2-. Ions and elements
            will still have distinct bonds if (1) the bonds list originally
            contained them and (2) no_oxi is False. These must match the
            token specified.
    """

    def __init__(
        self,
        nn=pmg_le.CrystalNN(),
        bbv=0,
        no_oxi=False,
        approx_bonds=False,
        token=" - ",
        allowed_bonds=None,
    ):
        self.nn = nn
        self.bbv = bbv
        self.no_oxi = no_oxi
        self.approx_bonds = approx_bonds

        if " " not in token:
            raise ValueError("A space must be present in the token.")

        if any([str.isalnum(i) for i in token]):
            raise ValueError("The token cannot have any alphanumeric " "characters.")

        token_els = token.split(" ")
        if len(token_els) != 3 and token != " ":
            raise ValueError(
                "The token must either be a space or be padded by" "single spaces with no spaces in between."
            )

        self.token = token

        if allowed_bonds is None:
            self.allowed_bonds = allowed_bonds
            self.fitted_bonds_ = allowed_bonds
        else:
            self.allowed_bonds = self._sanitize_bonds(allowed_bonds)
            self.fitted_bonds_ = self._sanitize_bonds(allowed_bonds)

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use one of the standard instances of a given NearNeighbor class.
        Pass args to __init__, such as allowed_bonds, using this method as well.

        Args:
            preset (str): preset type ("CrystalNN", "VoronoiNN", "JmolNN",
            "MiniumDistanceNN", "MinimumOKeeffeNN", or "MinimumVIRENN").

        Returns:
            CoordinationNumber from a preset.
        """
        nn = getattr(pmg_le, preset)
        return BondFractions(nn(), **kwargs)

    def fit(self, X, y=None):
        """
        Define the bond types allowed to be returned during each featurization.
        Bonds found during featurization which are not allowed will be omitted
        from the returned dataframe or matrix.

        Fit BondFractions by either passing an iterable of structures to
        training_data or by defining the bonds explicitly with allowed_bonds
        in __init__.

        Args:
            X (Series/list): An iterable of pymatgen Structure
                objects which will be used to determine the allowed bond
                types.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self

        """
        if not hasattr(X, "__getitem__"):
            raise ValueError("X must be an iterable of pymatgen Structures")

        X = X.values if isinstance(X, pd.Series) else X

        if not all([isinstance(x, Structure) for x in X]):
            raise ValueError("Each structure must be a pymatgen Structure " "object.")

        sanitized = self._sanitize_bonds(self.enumerate_all_bonds(X))

        if self.allowed_bonds is None:
            self.fitted_bonds_ = sanitized
        else:
            self.fitted_bonds_ = [b for b in sanitized if b in self.allowed_bonds]
            if len(self.fitted_bonds_) == 0:
                warnings.warn(
                    "The intersection between the allowed bonds "
                    "and the fitted bonds is zero. There's no bonds"
                    "to be featurized!"
                )

        return self

    def enumerate_bonds(self, s):
        """
        Lists out all the bond possibilities in a single structure.

        Args:
            s (Structure): A pymatgen structure

        Returns:
            A list of bond types in 'Li-O' form, where the order of the
            elements in each bond type is alphabetic.
        """
        els = s.composition.elements
        het_bonds = list(itertools.combinations(els, 2))
        het_bonds = [tuple(sorted([str(i) for i in j])) for j in het_bonds]
        hom_bonds = [(str(el), str(el)) for el in els]
        bond_types = [k[0] + self.token + k[1] for k in het_bonds + hom_bonds]
        return sorted(bond_types)

    def enumerate_all_bonds(self, structures):
        """
        Identify all the unique, possible bonds types of all structures present,
        and create the 'unified' bonds list.

        Args:
             structures (list/ndarray): List of pymatgen Structures

        Returns:
            A tuple of unique, possible bond types for an entire list of
            structures. This tuple is used to form the unified feature labels.
        """
        bond_types = []
        for s in structures:
            bts = self.enumerate_bonds(s)
            for bt in bts:
                if bt not in bond_types:
                    bond_types.append(bt)
        return tuple(sorted(bond_types))

    def featurize(self, s):
        """
        Quantify the fractions of each bond type in a structure.

        For collections of structures, bonds types which are not found in a
        particular structure (e.g., Li-P in BaTiO3) are represented as NaN.

        Args:
            s (Structure): A pymatgen Structure object

        Returns:
            (list) The feature list of bond fractions, in the order of the
                alphabetized corresponding bond names.
        """

        self._check_fitted()

        bond_types = tuple(self.enumerate_bonds(s))
        bond_types = self._sanitize_bonds(bond_types)
        bonds = {k: 0.0 for k in bond_types}
        tot_bonds = 0.0

        # if we find a bond in allowed_bonds not in bond_types, mark as bbv
        for b in self.fitted_bonds_:
            if b not in bond_types:
                if self.bbv is None:
                    bonds[b] = float("nan")
                else:
                    bonds[b] = self.bbv

        for i, _ in enumerate(s.sites):
            nearest = self.nn.get_nn(s, i)
            origin = s.sites[i].specie

            for neigh in nearest:
                btup = tuple(sorted([str(origin), str(neigh.specie)]))
                b = self._sanitize_bonds(btup[0] + self.token + btup[1])
                # The bond will not be in bonds if it is a forbidden bond
                # (when a local bond is not in allowed_bonds)
                tot_bonds += 1.0
                if b in bonds:
                    bonds[b] += 1.0

        if self.approx_bonds:
            bonds = self._approximate_bonds(bonds)

        # If allowed_bonds caused no bonds to be present, all bonds will be 0.
        # Prevent division by zero error.
        tot_bonds = tot_bonds or 1.0

        # if we find a bond in bond_types not in allowed_bonds, skip
        return [bonds[b] / tot_bonds for b in self.fitted_bonds_]

    def feature_labels(self):
        """
        Returns the list of allowed bonds. Throws an error if the featurizer
        has not been fit.
        """
        self._check_fitted()
        return [b + " bond frac." for b in self.fitted_bonds_]

    def _check_fitted(self):
        """
        Ensure the Featurizer has been fit to the dataframe
        """
        if self.fitted_bonds_ is None:
            raise NotFittedError(
                "BondFractions must have a list of allowed bonds."
                " Either pass in a list of bonds to the "
                'initializer with allowed_bonds, use "fit" with'
                " a list of structures, or do both to sets the "
                "intersection of the two as the allowed list."
            )

    def _sanitize_bonds(self, bonds):
        """
        Prevent errors and/or bond duplicates from badly formatted allowed_bonds

        Args:
            bonds (str/[str]): An iterable of bond types, specified as strings
                with the general format "El - Sp", where El or Sp can be specie
                or an element with pymatgen's str representation of a bond. For
                example, a Cesium Chloride bond could be represented as either
                "Cs-Cl" or "Cs+-Cl-" or "Cl-Cs" or "Cl--Cs+". "bond frac." may
                be present at the end of each bond, as it will be sanitized.
                Can also be a single string bond type.
        Returns:
            bonds ([str]): A listlike object containing alphabetized bond types.
                Note that ions and elements will still have distinct bonds if
                the bonds list originally contained them.
        """
        if isinstance(bonds, str):
            single = True
            bonds = [bonds]
        else:
            single = False
            try:
                bonds = list(bonds)
            except:
                # In the case of a series object
                bonds = bonds.tolist()

        for i, bond in enumerate(bonds):
            if not isinstance(bond, str):
                raise TypeError(
                    "Bonds must be specified as strings between "
                    "elements or species with the token in between, "
                    "for example Cl - Cs"
                )
            if self.token not in bond:
                raise ValueError('Token "{}" not found in bond: {}'.format(self.token, bond))
            bond = bond.replace(" bond frac.", "")
            species = sorted(bond.split(self.token))

            if self.no_oxi:
                alphabetized = self.token.join(species)
                species = self._species_from_bondstr(alphabetized)
                species = [str(s.element) for s in species]

            bonds[i] = self.token.join(species)
        bonds = list(OrderedDict.fromkeys(bonds))

        if single:
            return bonds[0]
        else:
            return tuple(sorted(bonds))

    def _species_from_bondstr(self, bondstr):
        """
        Create a 2-tuple of species objects from a bond string.

        Args:
            bondstr (str): A string representing a bond between elements or
                species, or a combination of the two. For example, "Cl- - Cs+".

        Returns:
            ((Species)): A tuple of pymatgen Species objects in alphabetical
                order.
        """
        species = []
        for ss in bondstr.split(self.token):
            try:
                species.append(Specie.from_string(ss))
            except ValueError:
                d = {"element": ss, "oxidation_state": 0}
                species.append(Specie.from_dict(d))
        return tuple(species)

    def _approximate_bonds(self, local_bonds):
        """
        Approximate a structure's bonds if the structure contains bonds not in
        allowed_bonds.

        Local bonds are approximated according to the "nearest" bonds present in
        allowed_bonds (the unified list). Nearness is measured by the euclidean
        distance (diff) in mendeleev number of each element. For example a Na-O
        bond could be approximated as a Li-O bond ( distance is sqrt(0^2 + 1^2)
         = 1).

        Args:
            local_bonds (dict): The bonds present in the structure with the bond
                types as keys ("Cl- - Cs+") and the bond fraction as values
                (0.7).

        Returns:
            abonds_data (dict): A dictionary of the unified (allowed) bonds
                with the bond names as keys and the corresponding bond fractions
                (whether approximated or true) as values.

        """

        # At this stage, local_bonds may contain unified bonds which
        # are nan.

        abonds_data = {k: 0.0 for k in self.fitted_bonds_}
        abonds_species = {k: None for k in self.fitted_bonds_}
        for ub in self.fitted_bonds_:
            species = self._species_from_bondstr(ub)
            abonds_species[ub] = tuple(species)
        # keys are pairs of species, values are bond names in unified_bonds
        abonds_species = {v: k for k, v in abonds_species.items()}

        for lb in local_bonds.keys():
            local_bonds[lb] = 0.0 if np.isnan(local_bonds[lb]) else local_bonds[lb]

            if lb in self.fitted_bonds_:
                abonds_data[lb] += local_bonds[lb]
            else:
                lbs = self._species_from_bondstr(lb)

                nearest = []
                d_min = None
                for abss in abonds_species.keys():

                    # The distance between bonds is euclidean. To get a good
                    # measure of the coordinate between mendeleev numbers for
                    # each specie, we use the minumum difference. ie, for
                    # finding the distance between Na-O and O-Li, we would
                    # not want the distance between (Na and O) and (O and Li),
                    # we want the distance between (Na and Li) and (O and O).

                    u_mends = sorted([j.element.mendeleev_no for j in abss])
                    l_mends = sorted([j.element.mendeleev_no for j in lbs])

                    d0 = u_mends[0] - l_mends[0]
                    d1 = u_mends[1] - l_mends[1]

                    d = (d0 ** 2.0 + d1 ** 2.0) ** 0.5
                    if not d_min:
                        d_min = d
                        nearest = [abss]
                    elif d < d_min:
                        # A new best approximation has been found
                        d_min = d
                        nearest = [abss]
                    elif d == d_min:
                        # An equivalent approximation has been found
                        nearest += [abss]
                    else:
                        pass

                # Divide bond fraction equally among all equiv. approximate bonds
                bond_frac = local_bonds[lb] / len(nearest)
                for n in nearest:
                    # Get the name of the approximate bond from the map
                    ab = abonds_species[n]

                    # Add the bond frac to that/those nearest bond(s)
                    abonds_data[ab] += bond_frac
        return abonds_data

    def implementors(self):
        return ["Alex Dunn"]

    def citations(self):
        return [
            "@article{doi:10.1021/acs.jpclett.5b00831, "
            "author = {Hansen, Katja and Biegler, "
            "Franziska and Ramakrishnan, Raghunathan and Pronobis, Wiktor"
            "and von Lilienfeld, O. Anatole and Muller, Klaus-Robert and"
            "Tkatchenko, Alexandre},"
            "title = {Machine Learning Predictions of Molecular Properties: "
            "Accurate Many-Body Potentials and Nonlocality in Chemical Space},"
            "journal = {The Journal of Physical Chemistry Letters},"
            "volume = {6},"
            "number = {12},"
            "pages = {2326-2331},"
            "year = {2015},"
            "doi = {10.1021/acs.jpclett.5b00831}, "
            "note ={PMID: 26113956},"
            "URL = {http://dx.doi.org/10.1021/acs.jpclett.5b00831}"
            "}"
        ]


class BagofBonds(BaseFeaturizer):
    """
    Compute a Bag of Bonds vector, as first described by Hansen et al. (2015).

    The Bag of Bonds approach is based creating an even-length vector from a
    Coulomb matrix output. Practically, it represents the Coloumbic interactions
    between each possible set of sites in a structure as a vector.

    BagofBonds must be fit to an iterable of structures using the "fit" method
    before featurization can occur. This is because the bags and the maximum
    lengths of each bag must be set prior to featurization. We recommend
    fitting and featurizing on the same data to maintain consistency
    between generated feature sets. This can be done using the fit_transform
    method (for lists of structures) or the fit_featurize_dataframe method
    (for dataframes).

    BagofBonds is based on a method by Hansen et. al "Machine Learning
    Predictions of Molecular Properties: Accurate Many-Body Potentials and
    Nonlocality in Chemical Space" (2015).

    Args:
        coulomb_matrix (BaseFeaturizer): A featurizer object containing a
            "featurize" method which returns a matrix of size nsites x nsites.
            Good choices are CoulombMatrix() or SineCoulombMatrix(), with the
            flatten=False parameter set.
        token (str): The string used to separate species in a bond, including
            spaces. The token must contain at least one space and cannot have
            alphabetic characters in it, and should be padded by spaces. For
            example, for the bond Cs+ - Cl-, the token is ' - '. This determines
            how bonds are represented in the dataframe.

    """

    def __init__(self, coulomb_matrix=SineCoulombMatrix(flatten=False), token=" - "):
        self.coulomb_matrix = coulomb_matrix
        self.token = token
        self.bag_lens = None
        self.ordered_bonds = None

    def _check_fitted(self):
        if not self.bag_lens or not self.ordered_bonds:
            raise NotFittedError(
                "BagofBonds not fitted to any list of "
                "structures! Use the 'fit' method to define "
                "the bags and the maximum length of each bag."
            )

    def fit(self, X, y=None):
        """
        Define the bags using a list of structures.

        Both the names of the bags (e.g., Cs-Cl) and the maximum lengths of
        the bags are set with fit.

        Args:
            X (Series/list): An iterable of pymatgen Structure
                objects which will be used to determine the allowed bond
                types and bag lengths.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self
        """
        unpadded_bobs = [self.bag(s, return_baglens=True) for s in X]
        bonds = [list(bob.keys()) for bob in unpadded_bobs]
        bonds = np.unique(sum(bonds, []))
        baglens = [0] * len(bonds)

        for i, bond in enumerate(bonds):
            for bob in unpadded_bobs:
                if bond in bob:
                    baglen = bob[bond]
                    baglens[i] = max((baglens[i], baglen))
        self.bag_lens = dict(zip(bonds, baglens))
        # Sort the bags by bag length, with the shortest coming first.
        self.ordered_bonds = [b[0] for b in sorted(self.bag_lens.items(), key=lambda bl: bl[1])]
        return self

    def bag(self, s, return_baglens=False):
        """
        Convert a structure into a bag of bonds, where each bag has no padded
        zeros. using this function will give the 'raw' bags, which when
        concatenated, will have different lengths.

        Args:
            s (Structure): A pymatgen Structure or IStructure object. May also
                work with a
            return_baglens (bool): If True, returns the bag of bonds with as
                a dictionary with the number of bonds as values in place
                of the vectors of coulomb matrix vals. If False, calculates
                Coulomb matrix values and returns 'raw' bags.

        Returns:
            (dict) A bag of bonds, where the keys are sorted tuples of pymatgen
                Site objects representing bonds or sites, and the values are the
                Coulomb matrix values for that bag.
        """
        sites = s.sites
        nsites = len(sites)
        bonds = np.zeros((nsites, nsites), dtype=object)
        for i, si in enumerate(sites):
            for j, sj in enumerate(sites):
                el0, el1 = si.specie, sj.specie
                if isinstance(el0, Specie):
                    el0 = el0.element
                if isinstance(el1, Specie):
                    el1 = el1.element
                if i == j:
                    bonds[i, j] = (el0,)
                else:
                    bonds[i, j] = tuple(sorted((el0, el1)))

        if return_baglens:
            bags = {b: 0 for b in np.unique(bonds)}
        else:
            cm = self.coulomb_matrix.featurize(s)[0]
            bags = {b: [] for b in np.unique(bonds)}

        for i in range(nsites):
            for j in range(nsites):
                bond = bonds[i, j]
                if return_baglens:
                    # Only return length of bag
                    bags[bond] = bags[bond] + 1
                else:
                    # Calculate bond "strength"
                    cmval = cm[i, j]
                    bags[bond].append(cmval)

        if return_baglens:
            return bags
        else:
            # We must sort the magnitude of bonds in each bag
            return {bond: sorted(bags[bond]) for bond in bags}

    def featurize(self, s):
        """
        Featurizes a structure according to the bag of bonds method.
        Specifically, each structure is first bagged by flattening the
        Coulomb matrix for the structure. Then, it is zero-padded according to
        the maximum number of bonds in each bag, for the set of bags that
        BagofBonds was fit with.

        Args:
            s (Structure): A pymatgen structure object

        Returns:
            (list): The Bag of Bonds vector for the input structure
        """
        self._check_fitted()
        unpadded_bob = self.bag(s)
        padded_bob = {bag: [0.0] * int(length) for bag, length in self.bag_lens.items()}

        for bond in unpadded_bob:
            if bond not in list(self.bag_lens.keys()):
                raise ValueError("{} is not in the fitted " "bonds/sites!".format(bond))
            baglen_s = len(unpadded_bob[bond])
            baglen_fit = self.bag_lens[bond]

            if baglen_s > baglen_fit:
                raise ValueError(
                    "The bond {} has more entries than was "
                    "fitted for (i.e., there are more {} bonds"
                    " in structure {} ({}) than the fitted set"
                    " allows ({}).".format(bond, bond, s, baglen_s, baglen_fit)
                )
            elif baglen_s < baglen_fit:
                padded_bob[bond] = unpadded_bob[bond] + [0.0] * (baglen_fit - baglen_s)
            else:
                padded_bob[bond] = unpadded_bob[bond]

        # Ensure the bonds are printed in correct order
        bob = [padded_bob[bond] for bond in self.ordered_bonds]
        return list(sum(bob, []))

    def feature_labels(self):
        self._check_fitted()
        labels = []
        for bag in self.ordered_bonds:
            if len(bag) == 1:
                basename = str(bag[0]) + " site #"
            else:
                basename = str(bag[0]) + self.token + str(bag[1]) + " bond #"
            bls = [basename + str(i) for i in range(self.bag_lens[bag])]
            labels += bls
        return labels

    def implementors(self):
        return ["Alex Dunn"]

    def citations(self):
        return [
            "@article{doi:10.1021/acs.jpclett.5b00831, "
            "author = {Hansen, Katja and Biegler, "
            "Franziska and Ramakrishnan, Raghunathan and Pronobis, Wiktor"
            "and von Lilienfeld, O. Anatole and Muller, Klaus-Robert and"
            "Tkatchenko, Alexandre},"
            "title = {Machine Learning Predictions of Molecular Properties: "
            "Accurate Many-Body Potentials and Nonlocality in Chemical Space},"
            "journal = {The Journal of Physical Chemistry Letters},"
            "volume = {6},"
            "number = {12},"
            "pages = {2326-2331},"
            "year = {2015},"
            "doi = {10.1021/acs.jpclett.5b00831}, "
            "note ={PMID: 26113956},"
            "URL = {http://dx.doi.org/10.1021/acs.jpclett.5b00831}"
            "}"
        ]


class GlobalInstabilityIndex(BaseFeaturizer):
    """
    The global instability index of a structure.

    The default is to use IUCr 2016 bond valence parameters for computing
    bond valence sums. If the structure has disordered site occupancies
    or non-integer valences on sites, pymatgen's bond valence sum method
    can be used instead.

    Note that pymatgen's bond valence sum method is prone to error unless
    the correct scale factor is supplied. A scale factor based on testing
    with perovskites is used here.
    TODO: Use scipy to optimize scale factor for minimizing GII

    Based on the following publication:

    'Structural characterization of R2BaCuO5 (R = Y, Lu, Yb, Tm, Er, Ho,
        Dy, Gd, Eu and Sm) oxides by X-ray and neutron diffraction',
        A.Salinas-Sanchez, J.L.Garcia-Muñoz, J.Rodriguez-Carvajal,
        R.Saez-Puche, and J.L.Martinez, Journal of Solid State Chemistry,
        100, 201-211 (1992),
        https://doi.org/10.1016/0022-4596(92)90094-C

    Args:
        r_cut: Float, how far to search for neighbors when computing bond valences
        disordered_pymatgen: Boolean, whether to fall back on pymatgen's bond
            valence sum method for disordered structures

    Features:
        The global instability index is the square root of the sum of squared
            differences of the bond valence sums from the formal valences
            averaged over all atoms in the unit cell.
    """

    def __init__(self, r_cut=4.0, disordered_pymatgen=False):

        bv = IUCrBondValenceData()
        self.bv_values = bv.params
        self.r_cut = r_cut
        self.disordered_pymatgen = disordered_pymatgen

    def precheck(self, struct):
        """
        Bond valence methods require atom pairs with oxidation states.

        Additionally, check if at least the first and last site's species
        have a entry in the bond valence parameters.

        Args:
            struct: Pymatgen Structure
        """

        anions = ["O", "N", "F", "Cl", "Br", "S", "Se", "I", "Te", "P", "H", "As"]

        for site in struct:
            # Fail if site doesn't have either attribute
            if not hasattr(site, "species"):
                return False
            if isinstance(site.species.elements[0], Element):
                return False

        elems = [str(x.element) for x in struct.composition.elements]

        # If compound is not ionically bonded, it is going to fail
        if not any([e in anions for e in elems]):
            return False
        valences = [site.species.elements[0].oxi_state for site in struct]

        # If the oxidation states are technically provided but any are 0, fails
        if not all(valences):
            return False

        if len(struct) > 200:
            warnings.warn("Computing bond valence sums for over 200 sites. " "Featurization might be very slow!")

        # Check that all cation-anion pairs are tabulated
        specs = struct.composition.elements.copy()
        while len(specs) > 1:
            spec1 = specs.pop()
            elem1 = str(spec1.element)
            val_1 = spec1.oxi_state
            for spec2 in specs:
                elem2 = str(spec2.element)
                val_2 = spec2.oxi_state
                if np.sign(val_1) == -1 and np.sign(val_2) == 1:
                    try:
                        self.get_bv_params(elem2, elem1, val_2, val_1)
                    except IndexError:
                        return False
        return True

    def featurize(self, struct):
        """
        Get global instability index.

        Args:
            struct: Pymatgen Structure object
        Returns:
            [gii]: Length 1 list with float value
        """

        if struct.is_ordered:
            gii = self.calc_gii_iucr(struct)
            if gii > 0.6:
                warnings.warn(
                    "GII extremely large. Table parameters may " "not be suitable or structure may be unusual."
                )

        else:
            if self.disordered_pymatgen:
                gii = self.calc_gii_pymatgen(struct, scale_factor=0.965)
                if gii > 0.6:
                    warnings.warn(
                        "GII extremely large. Pymatgen method may not be " "suitable or structure may be unusual."
                    )
                return [gii]
            else:
                raise ValueError("Structure must be ordered for table lookup method.")

        return [gii]

    def get_equiv_sites(self, s, site):
        """Find identical sites from analyzing space group symmetry."""
        sga = SpacegroupAnalyzer(s, symprec=0.01)
        sg = sga.get_space_group_operations
        sym_data = sga.get_symmetry_dataset()
        equiv_atoms = sym_data["equivalent_atoms"]
        wyckoffs = sym_data["wyckoffs"]
        sym_struct = SymmetrizedStructure(s, sg, equiv_atoms, wyckoffs)
        equivs = sym_struct.find_equivalent_sites(site)
        return equivs

    def calc_bv_sum(self, site_val, site_el, neighbor_list):
        """Computes bond valence sum for site.
        Args:
            site_val (Integer): valence of site
            site_el (String): element name
            neighbor_list (List): List of neighboring sites and their distances
        """
        bvs = 0
        for neighbor_info in neighbor_list:
            neighbor = neighbor_info[0]
            dist = neighbor_info[1]
            neighbor_val = neighbor.species.elements[0].oxi_state
            neighbor_el = str(neighbor.species.element_composition.elements[0])
            if neighbor_val % 1 != 0 or site_val % 1 != 0:
                raise ValueError("Some sites have non-integer valences.")
            try:
                if np.sign(site_val) == 1 and np.sign(neighbor_val) == -1:
                    params = self.get_bv_params(
                        cation=site_el,
                        anion=neighbor_el,
                        cat_val=site_val,
                        an_val=neighbor_val,
                    )
                    bvs += self.compute_bv(params, dist)
                elif np.sign(site_val) == -1 and np.sign(neighbor_val) == 1:
                    params = self.get_bv_params(
                        cation=neighbor_el,
                        anion=site_el,
                        cat_val=neighbor_val,
                        an_val=site_val,
                    )
                    bvs -= self.compute_bv(params, dist)
            except:
                raise ValueError(
                    "BV parameters for {} with valence {} and {} {} not "
                    "found in table"
                    "".format(site_el, site_val, neighbor_el, neighbor_val)
                )
        return bvs

    def calc_gii_iucr(self, s):
        """Computes global instability index using tabulated bv params.

        Args:
            s: Pymatgen Structure object
        Returns:
            gii: Float, the global instability index
        """
        elements = [str(i) for i in s.composition.element_composition.elements]
        if elements[0] == elements[-1]:
            raise ValueError("No oxidation states with single element.")

        bond_valence_sums = []
        cutoff = self.r_cut
        pairs = s.get_all_neighbors(r=cutoff)
        site_val_sums = {}  # Cache bond valence deviations

        for i, neighbor_list in enumerate(pairs):
            site = s[i]
            equivs = self.get_equiv_sites(s, site)
            flag = False

            # If symm. identical site has cached bond valence sum difference,
            # use it to avoid unnecessary calculations
            for item in equivs:
                if item in site_val_sums:
                    bond_valence_sums.append(site_val_sums[item])
                    site_val_sums[site] = site_val_sums[item]
                    flag = True
                    break
            if flag:
                continue
            site_val = site.species.elements[0].oxi_state
            site_el = str(site.species.element_composition.elements[0])
            bvs = self.calc_bv_sum(site_val, site_el, neighbor_list)

            site_val_sums[site] = bvs - site_val
        gii = np.linalg.norm(list(site_val_sums.values())) / np.sqrt(len(site_val_sums))
        return gii

    # Cache bond valence parameters
    @lru_cache(maxsize=512)
    def get_bv_params(self, cation, anion, cat_val, an_val):
        """Lookup bond valence parameters from IUPAC table.
        Args:
            cation: String, cation element
            anion: String, anion element
            cat_val: Integer, cation formal valence
            an_val: Integer, anion formal valence
        Returns:
            bond_val_list: dataframe of bond valence parameters
        """
        bv_data = self.bv_values
        bond_val_list = bv_data[
            (bv_data["Atom1"] == cation)
            & (bv_data["Atom1_valence"] == cat_val)
            & (bv_data["Atom2"] == anion)
            & (bv_data["Atom2_valence"] == an_val)
        ]
        # If multiple values exist, take first one
        return bond_val_list.iloc[0]

    @staticmethod
    def compute_bv(params, dist):
        """Compute bond valence from parameters.
        Args:
            params: Dataframe with Ro and B parameters
            dist: Float, distance to neighboring atom
        Returns:
            bv: Float, bond valence
        """
        bv = np.exp((params["Ro"] - dist) / params["B"])
        return bv

    def calc_gii_pymatgen(self, struct, scale_factor=0.965):
        """Calculates global instability index using Pymatgen's bond valence sum
        Args:
            struct: Pymatgen Structure object
            scale_factor: Float, tunable scale factor for bond valence
        Returns:
            gii: Float, global instability index
        """
        deviations = []
        cutoff = self.r_cut
        if struct.is_ordered:
            for site in struct:
                nn = struct.get_neighbors(site, r=cutoff)
                bvs = bond_valence.calculate_bv_sum(site, nn, scale_factor=scale_factor)
                deviations.append(bvs - site.species.elements[0].oxi_state)
            gii = np.linalg.norm(deviations) / np.sqrt(len(deviations))
        else:
            for site in struct:
                nn = struct.get_neighbors(site, r=cutoff)
                bvs = bond_valence.calculate_bv_sum_unordered(site, nn, scale_factor=scale_factor)
                min_diff = min([bvs - spec.oxi_state for spec in site.species.elements])
                deviations.append(min_diff)
            gii = np.linalg.norm(deviations) / np.sqrt(len(deviations))
        return gii

    def feature_labels(self):
        return ["global instability index"]

    def implementors(self):
        return ["Nicholas Wagner", "Nenian Charles", "Alex Dunn"]

    def citations(self):
        return [
            "@article{PhysRevB.87.184115,"
            "title = {Structural characterization of R2BaCuO5 (R = Y, Lu, Yb, Tm, Er, Ho,"
            " Dy, Gd, Eu and Sm) oxides by X-ray and neutron diffraction},"
            "author = {Salinas-Sanchez, A. and Garcia-Muñoz, J.L. and Rodriguez-Carvajal, "
            "J. and Saez-Puche, R. and Martinez, J.L.},"
            "journal = {Journal of Solid State Chemistry},"
            "volume = {100},"
            "issue = {2},"
            "pages = {201-211},"
            "year = {1992},"
            "doi = {10.1016/0022-4596(92)90094-C},"
            "url = {https://doi.org/10.1016/0022-4596(92)90094-C}}",
        ]


class StructuralHeterogeneity(BaseFeaturizer):
    """
    Variance in the bond lengths and atomic volumes in a structure

    These features are based on several statistics derived from the Voronoi
    tessellation of a structure. The first set of features relate to the
    variance in the average bond length across all atoms in the structure.
    The second relate to the variance of bond lengths between each neighbor
    of each atom. The final feature is the variance in Voronoi cell sizes
    across the structure.

    We define the 'average bond length' of a site as the weighted average of
    the bond lengths for all neighbors. By default, the weight is the
    area of the face between the sites.

    The 'neighbor distance variation' is defined as the weighted mean absolute
    deviation in both length for all neighbors of a particular site. As before,
    the weight is according to face area by default. For this statistic, we
    divide the mean absolute deviation by the mean neighbor distance for that
    site.

    Features:
        mean absolute deviation in relative bond length - Mean absolute deviation
            in the average bond lengths for all sites, divided by the
            mean average bond length
        max relative bond length - Maximum average bond length, divided by the
            mean average bond length
        min relative bond length - Minimum average bond length, divided by the
            mean average bond length
        [stat] neighbor distance variation - Statistic (e.g., mean) of the
            neighbor distance variation
        mean absolute deviation in relative cell size - Mean absolute deviation
            in the Voronoi cell volume across all sites in the structure.
            Divided by the mean Voronoi cell volume.

    References:
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.024104>`_
    """

    def __init__(self, weight="area", stats=("minimum", "maximum", "range", "mean", "avg_dev")):
        self.weight = weight
        self.stats = stats

    def featurize(self, strc):
        # Compute the Voronoi tessellation of each site
        voro = VoronoiNN(extra_nn_info=True, weight=self.weight)
        nns = get_all_nearest_neighbors(voro, strc)

        # Compute the mean bond length of each atom, and the mean
        #   variation within each cell
        mean_bond_lengths = np.zeros((len(strc),))
        bond_length_var = np.zeros_like(mean_bond_lengths)
        for i, nn in enumerate(nns):
            weights = [n["weight"] for n in nn]
            lengths = [n["poly_info"]["face_dist"] * 2 for n in nn]
            mean_bond_lengths[i] = PropertyStats.mean(lengths, weights)

            # Compute the mean absolute deviation of the bond lengths
            bond_length_var[i] = PropertyStats.avg_dev(lengths, weights) / mean_bond_lengths[i]

        # Normalize the bond lengths by the average of the whole structure
        #   This is done to make the attributes length-scale-invariant
        mean_bond_lengths /= mean_bond_lengths.mean()

        # Compute statistics related to bond lengths
        features = [
            PropertyStats.avg_dev(mean_bond_lengths),
            mean_bond_lengths.max(),
            mean_bond_lengths.min(),
        ]
        features += [PropertyStats.calc_stat(bond_length_var, stat) for stat in self.stats]

        # Compute the variance in volume
        cell_volumes = [sum(x["poly_info"]["volume"] for x in nn) for nn in nns]
        features.append(PropertyStats.avg_dev(cell_volumes) / np.mean(cell_volumes))

        return features

    def feature_labels(self):
        fl = [
            "mean absolute deviation in relative bond length",
            "max relative bond length",
            "min relative bond length",
        ]
        fl += [stat + " neighbor distance variation" for stat in self.stats]
        fl.append("mean absolute deviation in relative cell size")
        return fl

    def citations(self):
        return [
            "@article{Ward2017,"
            "author = {Ward, Logan and Liu, Ruoqian "
            "and Krishna, Amar and Hegde, Vinay I. "
            "and Agrawal, Ankit and Choudhary, Alok "
            "and Wolverton, Chris},"
            "doi = {10.1103/PhysRevB.96.024104},"
            "journal = {Physical Review B},"
            "pages = {024104},"
            "title = {{Including crystal structure attributes "
            "in machine learning models of formation energies "
            "via Voronoi tessellations}},"
            "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.024104},"
            "volume = {96},year = {2017}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class MinimumRelativeDistances(BaseFeaturizer):
    """
    Determines the relative distance of each site to its closest neighbor.

    We use the relative distance,
    f_ij = r_ij / (r^atom_i + r^atom_j), as a measure rather than the
    absolute distances, r_ij, to account for the fact that different
    atoms/species have different sizes.  The function uses the
    valence-ionic radius estimator implemented in Pymatgen.


    The features can be flattened so a uniform-length vector is returned for
    each material, regardless of the number of sites in each structure.
    Returning flat output REQUIRES fitting (using self.fit(...)). If fit,
    structures having fewer sites than the max sites among the fitting
    structures are extended with NaNs; structures with more sites are truncated.
    To return non-flat (i.e., requiring further processing) features so that
    no features are NaN and no distances are truncated, use flatten=False.

    Features:
        If using flatten=True:
        site #{number} min. rel. dist. (float): The minimum relative distance of
            site {number}
        site #{number} specie (str): The string representing the specie at site
            {number}
        site #{number} neighbor specie(s) (str, tuple(str)): The neighbor specie
            used to determine the minimum relative distance with respect to site
            {number}. If multiple neighbor sites have equivalent minimum
            relative distances,all these sites are listed in a tuple.

        If using flatten=False:
        minimum relative distance of each site ([float]): List of the minimum
            relative distance for each site. Structures with different numbers
            of sites will return a different length vector.
    Args:
        cutoff (float): (absolute) distance up to which tentative closest
            neighbors (on the basis of relative distances) are to be determined.
        flatten (bool): If True, returns a uniform length feature vector for
            each structure regardless of the number of sites in the structure.
            If True, you must call .fit() before featurizing.
        include_distances (bool): Include the numerical minimum relative
            distance in the returned features. Only used if flatten=True.
        include_species (bool): Include the species for each site and the
            species of the neighbor (as determined by minimum rel. distance).
            Only used as flatten=True.
    """

    def __init__(self, cutoff=10.0, flatten=True, include_distances=True, include_species=True):
        if not include_distances and not include_species:
            raise ValueError("Featurizer must return distances, species, or both.")

        self.include_distances = include_distances
        self.include_species = include_species
        self.cutoff = cutoff
        self.flatten = flatten
        self._max_sites = None

    def _check_fitted(self):
        if not self._max_sites and self.flatten:
            raise NotFittedError("If using flatten=True, MinimumRelativeDistances must be fit " "before using.")

    def fit(self, X, y=None):
        """
        Fit the MRD featurizer to a list of structures.
        Args:
            X ([Structure]): A list of pymatgen structures.
            y : unused (added for consistency with overridden method signature)
        Returns:
            self
        """
        self._max_sites = max([len(s.sites) for s in X])
        return self

    def featurize(self, s):
        """
        Get minimum relative distances of all sites of the input structure.

        Args:
            s: Pymatgen Structure object.

        Returns:
            dists_relative_min: (list of floats) list of all minimum relative
                    distances (i.e., for all sites).
        """
        self._check_fitted()
        vire = ValenceIonicRadiusEvaluator(s)
        n_sites = len(s.sites)
        parent_site_species = [None] * n_sites
        neighbor_site_species = [None] * n_sites
        dists_relative_min = [None] * n_sites

        for i, site in enumerate(vire.structure):
            dists_relative = []
            neigh_species_relative = []
            for nnsite, dist, *_ in vire.structure.get_neighbors(site, self.cutoff):
                r_site = vire.radii[site.species_string]
                r_neigh = vire.radii[nnsite.species_string]
                radii_dist = r_site + r_neigh
                d_relative = dist / radii_dist
                dists_relative.append(d_relative)
                neigh_species_relative.append(site.species_string)

            dists_relative = np.asarray(dists_relative)
            drmin = dists_relative.min()
            dists_relative_min[i] = drmin
            dists_relative_min_ix = np.where(dists_relative == drmin)

            neigh_species_equiv = np.asarray(neigh_species_relative)[dists_relative_min_ix]

            parent_site_species[i] = site.species_string

            if len(neigh_species_equiv) == 1:
                neighbor_site_species[i] = neigh_species_equiv[0]
            else:
                neighbor_site_species[i] = tuple(neigh_species_equiv)

        if self.flatten:
            features = []

            for i in range(self._max_sites):
                site_features = []
                if i <= n_sites - 1:
                    if self.include_distances:
                        site_features.append(dists_relative_min[i])
                    if self.include_species:
                        site_features.append(parent_site_species[i])
                        site_features.append(neighbor_site_species[i])
                else:
                    site_features = [np.nan] * (int(self.include_distances) + 2 * int(self.include_species))
                features += site_features
            return features

        else:
            return [dists_relative_min]

    def feature_labels(self):
        self._check_fitted()

        if self.flatten:
            labels = []
            for i in range(self._max_sites):
                site_labels = []
                if self.include_distances:
                    site_labels.append(f"site #{i} min. rel. dist.")
                if self.include_species:
                    site_labels.append(f"site #{i} specie")
                    site_labels.append(f"site #{i} neighbor specie(s)")
                labels += site_labels
            return labels
        else:
            return ["minimum relative distance of each site"]

    def citations(self):
        return [
            "@article{Zimmermann2017,"
            "author = {Zimmermann, Nils E. R. and Horton, Matthew K."
            " and Jain, Anubhav and Haranczyk, Maciej},"
            "doi = {10.3389/fmats.2017.00034},"
            "journal = {Frontiers in Materials},"
            "pages = {34},"
            "title = {{Assessing Local Structure Motifs Using Order"
            " Parameters for Motif Recognition, Interstitial"
            " Identification, and Diffusion Path Characterization}},"
            "url = {https://www.frontiersin.org/articles/10.3389/fmats.2017.00034},"
            "volume = {4},"
            "year = {2017}"
            "}"
        ]

    def implementors(self):
        return ["Nils E. R. Zimmermann", "Alex Dunn"]
