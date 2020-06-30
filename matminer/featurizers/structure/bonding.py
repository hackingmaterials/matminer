from __future__ import division, unicode_literals, print_function

import itertools
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from pymatgen import Structure
from pymatgen.core.periodic_table import Specie
import pymatgen.analysis.local_env as pmg_le

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.structure import SineCoulombMatrix

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

    def __init__(self, nn=pmg_le.CrystalNN(), bbv=0, no_oxi=False,
                 approx_bonds=False, token=' - ', allowed_bonds=None):
        self.nn = nn
        self.bbv = bbv
        self.no_oxi = no_oxi
        self.approx_bonds = approx_bonds

        if " " not in token:
            raise ValueError("A space must be present in the token.")

        if any([str.isalnum(i) for i in token]):
            raise ValueError("The token cannot have any alphanumeric "
                             "characters.")

        token_els = token.split(" ")
        if len(token_els) != 3 and token != " ":
            raise ValueError("The token must either be a space or be padded by"
                             "single spaces with no spaces in between.")

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
            raise ValueError("Each structure must be a pymatgen Structure "
                             "object.")

        sanitized = self._sanitize_bonds(self.enumerate_all_bonds(X))

        if self.allowed_bonds is None:
            self.fitted_bonds_ = sanitized
        else:
            self.fitted_bonds_ = [b for b in sanitized if
                                  b in self.allowed_bonds]
            if len(self.fitted_bonds_) == 0:
                warnings.warn("The intersection between the allowed bonds "
                              "and the fitted bonds is zero. There's no bonds"
                              "to be featurized!")

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
                'BondFractions must have a list of allowed bonds.'
                ' Either pass in a list of bonds to the '
                'initializer with allowed_bonds, use "fit" with'
                ' a list of structures, or do both to sets the '
                'intersection of the two as the allowed list.')

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
                raise TypeError("Bonds must be specified as strings between "
                                "elements or species with the token in between, "
                                "for example Cl - Cs")
            if not self.token in bond:
                raise ValueError('Token "{}" not found in bond: {}'.format(
                    self.token, bond))
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
                d = {'element': ss, 'oxidation_state': 0}
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
            local_bonds[lb] = 0.0 if np.isnan(local_bonds[lb]) else local_bonds[
                lb]

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
        return ["@article{doi:10.1021/acs.jpclett.5b00831, "
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
                "}"]


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

    def __init__(self, coulomb_matrix=SineCoulombMatrix(flatten=False),
                 token=' - '):
        self.coulomb_matrix = coulomb_matrix
        self.token = token
        self.bag_lens = None
        self.ordered_bonds = None

    def _check_fitted(self):
        if not self.bag_lens or not self.ordered_bonds:
            raise NotFittedError("BagofBonds not fitted to any list of "
                                 "structures! Use the 'fit' method to define "
                                 "the bags and the maximum length of each bag.")

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
        self.ordered_bonds = [b[0] for b in sorted(self.bag_lens.items(),
                                                   key=lambda bl: bl[1])]
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
        padded_bob = {bag: [0.0] * int(length) for bag, length in
                      self.bag_lens.items()}

        for bond in unpadded_bob:
            if bond not in list(self.bag_lens.keys()):
                raise ValueError("{} is not in the fitted "
                                 "bonds/sites!".format(bond))
            baglen_s = len(unpadded_bob[bond])
            baglen_fit = self.bag_lens[bond]

            if baglen_s > baglen_fit:
                raise ValueError("The bond {} has more entries than was "
                                 "fitted for (i.e., there are more {} bonds"
                                 " in structure {} ({}) than the fitted set"
                                 " allows ({}).".format(bond, bond, s, baglen_s,
                                                        baglen_fit))
            elif baglen_s < baglen_fit:
                padded_bob[bond] = unpadded_bob[bond] + \
                                   [0.0] * (baglen_fit - baglen_s)
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
        return ["@article{doi:10.1021/acs.jpclett.5b00831, "
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
                "}"]
