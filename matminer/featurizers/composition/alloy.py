"""
Composition featurizers specialized for use with alloys.
"""

import collections
import os
from functools import reduce

import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import get_el_sp

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import (
    MagpieData,
    CohesiveEnergyData,
    MixingEnthalpy,
)
from matminer.featurizers.composition.packing import AtomicPackingEfficiency

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "..", "..", "utils", "data_files")


class Miedema(BaseFeaturizer):
    """
    Formation enthalpies of intermetallic compounds, from Miedema et al.

    Calculate the formation enthalpies of the intermetallic compound,
    solid solution and amorphous phase of a given composition, based on
    semi-empirical Miedema model (and some extensions), particularly for
    transitional metal alloys.

    Support elemental, binary and multicomponent alloys.
    For elemental/binary alloys, the formulation is based on the original
    works by Miedema et al. in 1980s;
    For multicomponent alloys, the formulation is basically the linear
    combination of sub-binary systems. This is reported to work well for
    ternary alloys, but needs to be careful with quaternary alloys and more.

    Args:
        struct_types (str or [str]): default='all'
            'inter': intermetallic compound; 'ss': solid solution
            'amor': amorphous phase; 'all': same for ['inter', 'ss', 'amor']
            ['inter', 'ss']: amorphous phase and solid solution
        ss_types (str or [str]): only for ss, default='min'
            'fcc': fcc solid solution; 'bcc': bcc solid solution
            'hcp': hcp solid solution;
            'no_latt': solid solution with no specific structure type
            'min': min value of ['fcc', 'bcc', 'hcp', 'no_latt']
            'all': same for ['fcc', 'bcc', 'hcp', 'no_latt']
            ['fcc', 'bcc']: fcc and bcc solid solutions
        data_source (str): source of dataset, default='Miedema'
            'Miedema': 'Miedema.csv' placed in "matminer/utils/data_files/",
            containing the following model parameters for 73 elements:
            'molar_volume', 'electron_density', 'electronegativity'
            'valence_electrons', 'a_const', 'R_const', 'H_trans'
            'compressibility', 'shear_modulus', 'melting_point'
            'structural_stability'. Please see the references for details.
    Returns:
        (list of floats) Miedema formation enthalpies (eV/atom) for input
            struct_types:
            -Miedema_deltaH_inter: for intermetallic compound
            -Miedema_deltaH_ss: for solid solution, can include 'fcc', 'bcc',
                'hcp', 'no_latt', 'min' based on input ss_types
            -Miedema_deltaH_amor: for amorphous phase
    """

    def __init__(self, struct_types="all", ss_types="min", data_source="Miedema"):
        if isinstance(struct_types, list):
            self.struct_types = struct_types
        else:
            if struct_types == "all":
                self.struct_types = ["inter", "amor", "ss"]
            else:
                self.struct_types = [struct_types]

        if isinstance(ss_types, list):
            self.ss_types = ss_types
        else:
            if ss_types == "all":
                self.ss_types = ["fcc", "bcc", "hcp", "no_latt"]
            else:
                self.ss_types = [ss_types]

        self.data_source = data_source
        if self.data_source == "Miedema":
            self.df_dataset = pd.read_csv(os.path.join(data_dir, "Miedema.csv"), index_col="element")
        else:
            raise NotImplementedError("data_source {} not implemented yet".format(data_source))

        self.element_list = [Element(estr) for estr in self.df_dataset.index]

    def precheck(self, c: Composition) -> bool:
        """
        Precheck a single entry. Miedema does not work for compositons
        containing any elments for which the Miedema model has no parameters.
        To precheck an entire dataframe (qnd automatically gather
        the fraction of structures that will pass the precheck), please use
        precheck_dataframe.

        Args:
            c (pymatgen.Composition): The composition to precheck.

        Returns:
            (bool): If True, s passed the precheck; otherwise, it failed.
        """
        return all([e in self.element_list for e in c.element_composition.elements])

    def deltaH_chem(self, elements, fracs, struct):
        """
        Chemical term of formation enthalpy
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            struct (str): 'inter', 'ss' or 'amor'
        Returns:
            deltaH_chem (float): chemical term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el["molar_volume"])
        n_ws = np.array(df_el["electron_density"])
        elec = np.array(df_el["electronegativity"])
        val = np.array(df_el["valence_electrons"])
        a = np.array(df_el["a_const"])
        r = np.array(df_el["R_const"])
        h_trans = np.array(df_el["H_trans"])

        if struct == "inter":
            gamma = 8
        elif struct == "amor":
            gamma = 5
        else:
            gamma = 0

        c_sf = fracs * np.power(v_molar, 2 / 3) / np.dot(fracs, np.power(v_molar, 2 / 3))
        f = (c_sf * (1 + gamma * np.power(np.multiply.reduce(c_sf, 0), 2)))[::-1]
        v_a = np.array(
            [
                np.power(v_molar[0], 2 / 3) * (1 + a[0] * f[0] * (elec[0] - elec[1])),
                np.power(v_molar[1], 2 / 3) * (1 + a[1] * f[1] * (elec[1] - elec[0])),
            ]
        )
        c_sf_a = fracs * v_a / np.dot(fracs, v_a)
        f_a = (c_sf_a * (1 + gamma * np.power(np.multiply.reduce(c_sf_a, 0), 2)))[::-1]

        threshold = range(3, 12)
        if (val[0] in threshold) and (val[1] in threshold):
            p = 14.1
            r = 0.0
        elif (val[0] not in threshold) and (val[1] not in threshold):
            p = 10.7
            r = 0.0
        else:
            p = 12.35
            r = np.multiply.reduce(r, 0) * p
        q = p * 9.4

        eta_ab = (
            2
            * (
                -p * np.power(elec[0] - elec[1], 2)
                - r
                + q * np.power(np.power(n_ws[0], 1 / 3) - np.power(n_ws[1], 1 / 3), 2)
            )
            / reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3))
        )

        deltaH_chem = f_a[0] * fracs[0] * v_a[0] * eta_ab + np.dot(fracs, h_trans)
        return deltaH_chem

    def deltaH_elast(self, elements, fracs):
        """
        Elastic term of formation enthalpy
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
        Returns:
            deltaH_elastic (float): elastic term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el["molar_volume"])
        n_ws = np.array(df_el["electron_density"])
        elec = np.array(df_el["electronegativity"])
        compr = np.array(df_el["compressibility"])
        shear_mod = np.array(df_el["shear_modulus"])

        alp = np.multiply(1.5, np.power(v_molar, 2 / 3)) / reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3))
        v_a = v_molar + np.array(
            [
                alp[0] * (elec[0] - elec[1]) / n_ws[0],
                alp[1] * (elec[1] - elec[0]) / n_ws[1],
            ]
        )
        alp_a = np.multiply(1.5, np.power(v_a, 2 / 3)) / reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3))

        # effective volume in alloy
        vab_a = v_molar[0] + np.array(
            [
                alp_a[0] * (elec[0] - elec[1]) / n_ws[0],
                alp_a[1] * (elec[1] - elec[0]) / n_ws[0],
            ]
        )
        vba_a = v_molar[1] + np.array(
            [
                alp_a[0] * (elec[0] - elec[1]) / n_ws[1],
                alp_a[1] * (elec[1] - elec[0]) / n_ws[1],
            ]
        )

        # H_elast A in B
        hab_elast = (2 * compr[0] * shear_mod[1] * np.power((vab_a[0] - vba_a[0]), 2)) / (
            4 * shear_mod[1] * vab_a[0] + 3 * compr[0] * vba_a[0]
        )
        # H_elast B in A
        hba_elast = (2 * compr[1] * shear_mod[0] * np.power((vba_a[1] - vab_a[1]), 2)) / (
            4 * shear_mod[0] * vba_a[1] + 3 * compr[1] * vab_a[1]
        )

        deltaH_elast = np.multiply.reduce(fracs, 0) * (fracs[1] * hab_elast + fracs[0] * hba_elast)
        return deltaH_elast

    def deltaH_struct(self, elements, fracs, latt):
        """
        Structural term of formation enthalpy, only for solid solution
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            latt (str): 'fcc', 'bcc', 'hcp' or 'no_latt'
        Returns:
            deltaH_struct (float): structural term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        val = np.array(df_el["valence_electrons"])
        struct_stab = np.array(df_el["structural_stability"])

        if latt == "fcc":
            latt_stab_dict = {
                0.0: 0.0,
                1.0: 0,
                2.0: 0,
                3.0: -2,
                4.0: -1.5,
                5.0: 9.0,
                5.5: 14.0,
                6.0: 11.0,
                7.0: -3.0,
                8.0: -9.5,
                8.5: -11.0,
                9.0: -9.0,
                10.0: -2.0,
                11.0: 1.5,
                12.0: 0.0,
                13.0: 0.0,
                14.0: 0.0,
                15.0: 0.0,
            }
        elif latt == "bcc":
            latt_stab_dict = {
                0.0: 0.0,
                1.0: 0.0,
                2.0: 0.0,
                3.0: 2.2,
                4.0: 2.0,
                5.0: -9.5,
                5.5: -14.5,
                6.0: -12.0,
                7.0: 4.0,
                8.0: 10.0,
                8.5: 11.0,
                9.0: 8.5,
                10.0: 1.5,
                11.0: 1.5,
                12.0: 0.0,
                13.0: 0.0,
                14.0: 0.0,
                15.0: 0.0,
            }
        elif latt == "hcp":
            latt_stab_dict = {
                0.0: 0.0,
                1.0: 0.0,
                2.0: 0.0,
                3.0: -2.5,
                4.0: -2.5,
                5.0: 10.0,
                5.5: 15.0,
                6.0: 13.0,
                7.0: -5.0,
                8.0: -10.5,
                8.5: -11.0,
                9.0: -8.0,
                10.0: -1.0,
                11.0: 2.5,
                12.0: 0.0,
                13.0: 0.0,
                14.0: 0.0,
                15.0: 0.0,
            }
        else:
            return 0
        latt_stab_dict = collections.OrderedDict(sorted(latt_stab_dict.items(), key=lambda t: t[0]))
        # lattice stability of different lattice_types
        val_avg = np.dot(fracs, val)
        val_bd_lower, val_bd_upper = 0, 0
        for key in latt_stab_dict.keys():
            if val_avg - key <= 0:
                val_bd_upper = key
                break
            else:
                val_bd_lower = key

        latt_stab = (val_avg - val_bd_lower) * latt_stab_dict[val_bd_upper] / (val_bd_upper - val_bd_lower) + (
            val_bd_upper - val_avg
        ) * latt_stab_dict[val_bd_lower] / (val_bd_upper - val_bd_lower)

        deltaH_struct = latt_stab - np.dot(fracs, struct_stab)
        return deltaH_struct

    def deltaH_topo(self, elements, fracs):
        """
        Topological term of formation enthalpy, only for amorphous phase
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
        Returns:
            deltaH_topo (float): topological term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        melt_point = np.array(df_el["melting_point"])

        deltaH_topo = 3.5 * np.dot(fracs, melt_point) / 1000
        return deltaH_topo

    def featurize(self, comp):
        """
        Get Miedema formation enthalpies of target structures: inter, amor,
        ss (can be further divided into 'min', 'fcc', 'bcc', 'hcp', 'no_latt'
            for different lattice_types)
        Args:
            comp: Pymatgen composition object
        Returns:
            miedema (list of floats): formation enthalpies of target structures
        """
        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        fracs = [el_amt[el] for el in elements]
        el_num = len(elements)
        # divide into a list of sub-binaries
        el_bins = []
        frac_bins = []
        for i in range(el_num - 1):
            for j in range(i + 1, el_num):
                el_bins.append([elements[i], elements[j]])
                frac_bins.append([fracs[i], fracs[j]])

        miedema = []
        for struct_type in self.struct_types:
            # inter: intermetallic compound
            if struct_type == "inter":
                deltaH_chem_inter = 0
                for i_inter, el_bin in enumerate(el_bins):
                    deltaH_chem_inter += self.deltaH_chem(el_bin, frac_bins[i_inter], "inter")
                miedema.append(deltaH_chem_inter)
            # ss: solid solution
            elif struct_type == "ss":
                deltaH_chem_ss = 0
                deltaH_elast_ss = 0
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_ss += self.deltaH_chem(el_bin, frac_bins[sub_bin], "ss")
                    deltaH_elast_ss += self.deltaH_elast(el_bin, frac_bins[sub_bin])

                for ss_type in self.ss_types:
                    if ss_type == "min":
                        deltaH_ss_all = []
                        for latt in ["fcc", "bcc", "hcp", "no_latt"]:
                            deltaH_ss_all.append(
                                deltaH_chem_ss + deltaH_elast_ss + self.deltaH_struct(elements, fracs, latt)
                            )
                        deltaH_ss_min = min(deltaH_ss_all)
                        miedema.append(deltaH_ss_min)
                    else:
                        deltaH_struct_ss = self.deltaH_struct(elements, fracs, ss_type)
                        miedema.append(deltaH_chem_ss + deltaH_elast_ss + deltaH_struct_ss)
            # amor: amorphous phase
            elif struct_type == "amor":
                deltaH_chem_amor = 0
                deltaH_topo_amor = self.deltaH_topo(elements, fracs)
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_amor += self.deltaH_chem(el_bin, frac_bins[sub_bin], "amor")
                miedema.append(deltaH_chem_amor + deltaH_topo_amor)

        # convert kJ/mol to eV/atom. The original Miedema model is in kJ/mol.
        miedema = [deltaH / 96.4853 for deltaH in miedema]
        return miedema

    def feature_labels(self):
        labels = []
        for struct_type in self.struct_types:
            if struct_type == "ss":
                for ss_type in self.ss_types:
                    labels.append("Miedema_deltaH_ss_" + ss_type)
            else:
                labels.append("Miedema_deltaH_" + struct_type)
        return labels

    def citations(self):
        miedema_citation = (
            "@article{miedema_1988, "
            "title={Cohesion in metals},"
            "author={De Boer, Frank R and Mattens, WCM "
            "and Boom, R and Miedema, AR and Niessen, AK},"
            "year={1988}}"
        )
        zhang_citation = (
            "@article{miedema_zhang_2016, "
            "title={Miedema Calculator: A thermodynamic platform "
            "for predicting formation enthalpies of alloys within "
            "framework of Miedema's Theory},"
            "author={R.F. Zhang, S.H. Zhang, Z.J. He, J. Jing and S.H. Sheng},"
            "journal={Computer Physics Communications}"
            "year={2016}}"
        )
        ternary_citation = (
            "@article{miedema_alonso_1990, "
            "title={Glass formation in ternary transition metal alloys},"
            "author={L J Gallego, J A Somoza and J A Alonso},"
            "journal={Journal of Physics: Condensed Matter}"
            "year={1990}}"
        )
        return [miedema_citation, zhang_citation, ternary_citation]

    def implementors(self):
        return ["Qi Wang", "Alireza Faghaninia"]


class YangSolidSolution(BaseFeaturizer):
    """
    Mixing thermochemistry and size mismatch terms of Yang and Zhang (2012)

    This featurizer returns two different features developed by
    .. Yang and Zhang `https://linkinghub.elsevier.com/retrieve/pii/S0254058411009357`
    to predict whether metal alloys will form metallic glasses,
    crystalline solid solutions, or intermetallics.
    The first, Omega, is related to the balance between the mixing entropy and
    mixing enthalpy of the liquid phase. The second, delta, is related to the
    atomic size mismatch between the different elements of the material.

    Features
        Yang omega - Mixing thermochemistry feature, Omega
        Yang delta - Atomic size mismatch term

    References:
        .. Yang and Zhang (2012) `https://linkinghub.elsevier.com/retrieve/pii/S0254058411009357`.
    """

    def __init__(self):
        # Load in the mixing enthalpy data
        #  Creates a lookup table of the liquid mixing enthalpies
        self.dhf_mix = MixingEnthalpy()

        # Load in a table of elemental properties
        self.elem_data = MagpieData()

    def precheck(self, c: Composition) -> bool:
        """
        Precheck a single entry. YangSolidSolution does not work for compositons
        containing any binary elment combinations for which the model has no
        parameters. We can nearly equivalently approximate this by checking
        against the unary element list.

        To precheck an entire dataframe (qnd automatically gather
        the fraction of structures that will pass the precheck), please use
        precheck_dataframe.

        Args:
            c (pymatgen.Composition): The composition to precheck.

        Returns:
            (bool): If True, s passed the precheck; otherwise, it failed.
        """
        return all([e in self.dhf_mix.valid_element_list for e in c.element_composition.elements])

    def featurize(self, comp):
        return [self.compute_omega(comp), self.compute_delta(comp)]

    def compute_omega(self, comp):
        """Compute Yang's mixing thermodynamics descriptor

        :math:`\\frac{T_m \Delta S_{mix}}{ |  \Delta H_{mix} | }`

        Where :math:`T_m` is average melting temperature,
        :math:`\Delta S_{mix}` is the ideal mixing entropy,
        and :math:`\Delta H_{mix}` is the average mixing enthalpies
        of all pairs of elements in the alloy

        Args:
            comp (Composition) - Composition to featurizer
        Returns:
            (float) Omega
        """

        # Special case: Elemental compound (entropy == 0 -> Omega == 1)
        if len(comp) == 1:
            return 0

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.fractional_composition.items())

        # Get the mean melting temperature
        mean_Tm = PropertyStats.mean(self.elem_data.get_elemental_properties(elements, "MeltingT"), fractions)

        # Get the mixing entropy
        entropy = np.dot(fractions, np.log(fractions)) * 8.314 / 1000

        # Get the mixing enthalpy
        enthalpy = 0
        for i, (e1, f1) in enumerate(zip(elements, fractions)):
            for e2, f2 in zip(elements[:i], fractions):
                enthalpy += f1 * f2 * self.dhf_mix.get_mixing_enthalpy(e1, e2)
        enthalpy *= 4

        # Make sure the enthalpy is nonzero
        #  The limit as dH->0 of omega is +\inf. A very small positive dH will approximate
        #  this limit without causing issues with infinite features
        enthalpy = max(1e-6, abs(enthalpy))

        return abs(mean_Tm * entropy / enthalpy)

    def compute_delta(self, comp):
        """Compute Yang's delta parameter

        :math:`\sqrt{\sum^n_{i=1} c_i \left( 1 - \\frac{r_i}{\\bar{r}} \\right)^2 }`

        where :math:`c_i` and :math:`r_i` are the fraction and radius of
        element :math:`i`, and :math:`\\bar{r}` is the fraction-weighted
        average of the radii. We use the radii compiled by
        .. Miracle et al. `https://www.tandfonline.com/doi/ref/10.1179/095066010X12646898728200?scroll=top`.

        Args:
            comp (Composition) - Composition to assess
        Returns:
            (float) delta

        """

        elements, fractions = zip(*comp.element_composition.items())

        # Get the radii of elements
        radii = self.elem_data.get_elemental_properties(elements, "MiracleRadius")
        mean_r = PropertyStats.mean(radii, fractions)

        # Compute the mean (1 - r/\\bar{r})^2
        r_dev = np.power(1.0 - np.divide(radii, mean_r), 2)
        return np.sqrt(PropertyStats.mean(r_dev, fractions))

    def feature_labels(self):
        return ["Yang omega", "Yang delta"]

    def citations(self):
        return [
            "@article{Yang2012,"
            "author = {Yang, X. and Zhang, Y.},"
            "doi = {10.1016/j.matchemphys.2011.11.021},"
            "journal = {Materials Chemistry and Physics},"
            "number = {2-3},"
            "pages = {233--238},"
            "title = {{Prediction of high-entropy stabilized solid-solution in multi-component alloys}},"
            "url = {http://dx.doi.org/10.1016/j.matchemphys.2011.11.021},"
            "volume = {132},year = {2012}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class WenAlloys(BaseFeaturizer):
    """
    Calculate features for alloy properties.

    Based on the work:

    "Machine learning assisted design of high entropy alloys
    with desired property" by Wen et al., Acta Materiala 170,
    109-117 (2019).

    Copyright 2020 Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

    Features:
        Yang omega
        Yang delta
        Radii local mismatch
        Radii gamma
        Configuration entropy
        Lambda entropy
        Electronegativity delta
        Electronegativity local mismatch
        VEC mean
        Mixing enthalpy
        Mean cohesive energy
        Interant electrons
        Shear modulus mean
        Shear modulus delta
        Shear modulus local mismatch
        Shear modulus strength model

    Copyright 2020 Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
    """

    def __init__(self):
        # Use of Miedema to retrieve the shear modulus
        self.data_source_miedema = Miedema(data_source="Miedema")
        self.data_source_magpie = MagpieData().all_elemental_props
        self.data_source_cohesive_energy = CohesiveEnergyData()
        self.data_source_enthalpy = MixingEnthalpy()
        self.yss = YangSolidSolution()

    def precheck(self, comp):
        return self.yss.precheck(comp)

    def featurize(self, comp):
        """
        Get elemental property attributes
        Args:
            comp: Pymatgen composition object
        Returns:
            (list): Generated Wen et al. features.
        """
        composition_dict = comp.fractional_composition.get_el_amt_dict()
        elements = list(composition_dict.keys())
        fractions = list(composition_dict.values())
        miracle_radius_stats = self.compute_magpie_summary("MiracleRadius", elements, fractions)
        atomic_weight_stats = self.compute_magpie_summary("AtomicWeight", elements, fractions)
        electronegativity = [self.data_source_miedema.df_dataset.loc[str(e)]["electronegativity"] for e in elements]
        single_VEC = [self.data_source_miedema.df_dataset.loc[str(e)]["valence_electrons"] for e in elements]
        mean_VEC = PropertyStats.mean(single_VEC, fractions)
        cohesive_energy = [self.data_source_cohesive_energy.cohesive_energy_data[str(e)] for e in elements]
        mean_cohesive_energy = PropertyStats.mean(cohesive_energy, fractions)
        shear_modulus = [self.data_source_miedema.df_dataset.loc[str(e)]["shear_modulus"] for e in elements]
        mean_shear_modulus = PropertyStats.mean(shear_modulus, fractions)

        s_unfilled = sum(
            [
                2 - self.data_source_magpie["NsUnfilled"][e]
                for e in elements
                if self.data_source_magpie["NsUnfilled"][e] != 0
            ]
        )
        p_unfilled = sum(
            [
                6 - self.data_source_magpie["NpUnfilled"][e]
                for e in elements
                if self.data_source_magpie["NpUnfilled"][e] != 0
            ]
        )
        d_unfilled = sum(
            [
                10 - self.data_source_magpie["NdUnfilled"][e]
                for e in elements
                if self.data_source_magpie["NdUnfilled"][e] != 0
            ]
        )
        f_unfilled = sum(
            [
                14 - self.data_source_magpie["NfUnfilled"][e]
                for e in elements
                if self.data_source_magpie["NfUnfilled"][e] != 0
            ]
        )
        interant_electrons = s_unfilled + p_unfilled + d_unfilled + f_unfilled
        weight_fraction = self.compute_weight_fraction(elements, comp)
        atomic_fraction = self.compute_atomic_fraction(elements, comp)
        yang_delta = self.yss.compute_delta(comp)
        yang_omega = self.yss.compute_omega(comp)
        ape = AtomicPackingEfficiency().compute_simultaneous_packing_efficiency(comp)[0]
        radii_local_mismatch = self.compute_local_mismatch(miracle_radius_stats["array"], fractions)
        radii_gamma = self.compute_gamma_radii(miracle_radius_stats)
        S_config = self.compute_configuration_entropy(fractions)
        atomic_weight_mean = atomic_weight_stats["mean"]
        wt = comp.weight
        lambda_entropy = self.compute_lambda(yang_delta, S_config)
        X_delta = self.compute_delta(electronegativity, fractions)
        X_local_mismatch = self.compute_local_mismatch(electronegativity, fractions)
        H_mixing = self.compute_enthalpy(elements, fractions)
        shear_modulus_delta = self.compute_delta(shear_modulus, fractions)
        shear_modulus_local_mismatch = self.compute_local_mismatch(shear_modulus, fractions)
        shear_modulus_strength_model = self.compute_strength_local_mismatch_shear(
            shear_modulus=shear_modulus, mean_shear_modulus=mean_shear_modulus, fractions=fractions
        )

        return [
            weight_fraction,
            atomic_fraction,
            yang_delta,
            yang_omega,
            ape,
            radii_local_mismatch,
            radii_gamma,
            S_config,
            atomic_weight_mean,
            wt,
            lambda_entropy,
            X_delta,
            X_local_mismatch,
            mean_VEC,
            H_mixing,
            mean_cohesive_energy,
            interant_electrons,
            s_unfilled,
            p_unfilled,
            d_unfilled,
            f_unfilled,
            mean_shear_modulus,
            shear_modulus_delta,
            shear_modulus_local_mismatch,
            shear_modulus_strength_model,
        ]

    @staticmethod
    def compute_local_mismatch(variable, fractions):
        """Compute local mismatch of a given variable.

        :math:`\sum^n_{i=1} \sum^n_{j=1,i \neq j}  c_i c_j | v_i - v_j |^2`

        where :math:`c_{i,j}` and :math:`v_{i,j}` are the fraction and variable of
        element :math:`i,j`.
        Args:
            variable (list): List of properties to asses
            fractions (list): List of fractions to asses
        Returns:
            (float) local mismatch
        """

        array_variable = np.array(variable)
        array_fractions = np.array(fractions)
        variable_upper_triangle = abs((array_variable[:, None] - array_variable)[np.triu_indices(len(variable), k=1)])
        fractions_upper_triangle = (array_fractions[:, None] * array_fractions)[np.triu_indices(len(fractions), k=1)]
        return sum(variable_upper_triangle * fractions_upper_triangle)

    @staticmethod
    def compute_delta(variable, fractions):
        """Compute Yang's delta parameter for a generic variable.

        :math:`\sqrt{\sum^n_{i=1} c_i \left( 1 - \\frac{v_i}{\\bar{v}} \\right)^2 }`

        where :math:`c_i` and :math:`v_i` are the fraction and variable of
        element :math:`i`, and :math:`\\bar{v}` is the fraction-weighted
        average of the variable.
        Args:
            variable (list): List of properties to asses
            fractions (list): List of fractions to asses
        Returns:
            (float) delta
        """
        mean_variable = PropertyStats.mean(variable, fractions)
        dev_variable = np.power(1.0 - np.divide(variable, mean_variable), 2)
        return np.sqrt(PropertyStats.mean(dev_variable, fractions))

    @staticmethod
    def compute_lambda(yang_delta, entropy):
        """
        Args:
            yang_delta (float): Yang Solid Solution Delta
            entropy (float): Configuration entropy

        Returns:
            float
        """
        if yang_delta != 0:
            return entropy / yang_delta ** 2
        else:
            return 0

    @staticmethod
    def compute_gamma_radii(miracle_radius_stats):
        """Compute Gamma of the radii. The solid angles of the
        atomic packing for the elements with the most significant
        and smallest atomic sizes.

        :math:`\frac{1 - \sqrt{ \frac{((r + r_{min})^2 - r^2)}{(r + r_{min})^2}}}{1 - \sqrt{ \frac{((r + r_{max})^2 - r^2)}{(r + r_{max})^2}}}`

        where :math:`r`, :math:`r_{min}` and :math:`r_{max}` are the mean radii
        min radii and max radii.

        Args:
            miracle_radius_stats (dict): Dictionary of stats for miracleradius via compute_magpie_summary

        Returns:
            (float) gamma
        """
        mrmean = miracle_radius_stats["mean"]
        mrmin = miracle_radius_stats["min"]
        mrmax = miracle_radius_stats["max"]

        numerator = 1 - np.sqrt((mrmean * mrmin + mrmin ** 2) / (mrmean + mrmin) ** 2)
        denominator = 1 - np.sqrt((mrmean * mrmax + mrmax ** 2) / (mrmean + mrmax) ** 2)
        return numerator / denominator

    @staticmethod
    def compute_configuration_entropy(fractions):
        """Compute the configuration entropy.

        :math:`R \sum^n_{i=1} c_i \ln{c_i}`

        where :math:`c_i` are the fraction of each element :math:`i`
        and :math:`R` is the ideal gas constant
        Args:
            fractions ([float]): List of element fractions
        Returns:
            (float) gamma
        """

        return np.dot(fractions, np.log(fractions)) * 8.314 / 1000

    @staticmethod
    def compute_weight_fraction(elements, composition):
        """Get weight fraction string.

        Args:
            elements ([pymatgen.Element or str]): List of elements
            composition (pymatgen.Composition): Composition

        Returns:
            (str)
        """
        weight_fraction = ""
        for single_element in elements:
            weight_fraction += single_element + str(composition.get_wt_fraction(single_element))

        return weight_fraction

    @staticmethod
    def compute_atomic_fraction(elements, composition):
        """Get atomic fraction string.

        Args:
            elements ([pymatgen.Element or str]): List of elements
            composition (pymatgen.Composition): Composition

        Returns:
            (str)
        """
        atomic_fraction = ""
        for single_element in elements:
            atomic_fraction += single_element + str(composition.get_atomic_fraction(single_element))

        return atomic_fraction

    @staticmethod
    def compute_strength_local_mismatch_shear(shear_modulus, mean_shear_modulus, fractions):
        """The local mismatch of the shear values.

        :math:`\sum^n_{i=1} \frac{c_i \frac{2(G_i - G)}{G_i + G} }{\left(1 + 0.5 |c_i \frac{2(G_i - G)}{G_i + G} \right)|}`

        where :math:`c_{i}`, :math:'G' and :math:`G_{i}` are the fraction, mean shear modulus and shear modulus of
        element :math:`i`.
        Args:
            shear_modulus ([float]): List of shear moduli of elements
            mean_shear_modulus(float): Mean of shear moduli
            fractions ([float]): List of element fractions in the composition

        Returns:
            (float) strengthening local mismatch
        """

        array_shear = np.array(shear_modulus)
        array_fractions = np.array(fractions)
        modulus_combination = (
            2 * array_fractions * (array_shear - mean_shear_modulus) / (array_shear + mean_shear_modulus)
        )
        return sum(modulus_combination / (1 + 0.5 * abs(modulus_combination)))

    def compute_magpie_summary(self, attribute_name, elements, fractions):
        """Get limited list of weighted statistics according to magpie data.

        Args:
            attribute_name (str): Name of magpie attribute to retrieve
            elements ([pymatgen.element or str]): List of elements
            fractions ([float]): List of element fractions

        Returns:
            (dict) Dictionary of element-fraction weighted statistics for attribute.
        """
        attribute = [self.data_source_magpie[attribute_name][e] for e in elements]
        return {
            "array": attribute,
            "mean": PropertyStats.mean(attribute, fractions),
            "min": PropertyStats.minimum(attribute, fractions),
            "max": PropertyStats.maximum(attribute, fractions),
        }

    def compute_enthalpy(self, elements, fractions):
        """Compute mixing enthalpy.

        Args:
            elements ([pymatgen.Element or str]): List of elements
            fractions [float]: Fractions of elements in composition

        Returns:
            (float) H_mixing
        """
        enthalpy = 0
        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements[:i]):
                enthalpy += (
                    fractions[i]
                    * fractions[j]
                    * self.data_source_enthalpy.get_mixing_enthalpy(Element(e1), Element(e2))
                )
        enthalpy *= 4
        # Make sure the enthalpy is nonzero
        #  The limit as dH->0 of omega is +\inf. A very small positive dH will approximate
        #  this limit without causing issues with infinite features
        enthalpy = max(1e-6, abs(enthalpy))
        return abs(enthalpy)

    def feature_labels(self):
        return [
            "Weight Fraction",
            "Atomic Fraction",
            "Yang delta",
            "Yang omega",
            "APE mean",
            "Radii local mismatch",
            "Radii gamma",
            "Configuration entropy",
            "Atomic weight mean",
            "Total weight",
            "Lambda entropy",
            "Electronegativity delta",
            "Electronegativity local mismatch",
            "VEC mean",
            "Mixing enthalpy",
            "Mean cohesive energy",
            "Interant electrons",
            "Interant s electrons",
            "Interant p electrons",
            "Interant d electrons",
            "Interant f electrons",
            "Shear modulus mean",
            "Shear modulus delta",
            "Shear modulus local mismatch",
            "Shear modulus strength model",
        ]

    def citations(self):
        return [
            "@article{wen2019machine,"
            "author={Wen, Cheng and Zhang, Yan and Wang, Changxin and Xue, Dezhen and Bai, Yang and Antonov, Stoichko and Dai, Lanhong and Lookman, Turab and Su, Yanjing},"
            "doi = {10.1016/j.actamat.2019.03.010},"
            "journal={Acta Materialia},"
            "pages={109--117},"
            "title={Machine learning assisted design of high entropy alloys with desired property},"
            "url = {https://doi.org/10.1016/j.actamat.2019.03.010},"
            "volume={170},year={2019}"
        ]

    def implementors(self):
        return ["M. Ross Kunz", "Jeffery Aguiar", "Alex Dunn"]
