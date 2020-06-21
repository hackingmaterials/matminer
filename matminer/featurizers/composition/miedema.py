from __future__ import division

import collections
import os
from functools import reduce

import numpy as np
import pandas as pd
from pymatgen import Element
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import get_el_sp

from matminer.featurizers.base import BaseFeaturizer

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "../..", "utils", "data_files")


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

    def __init__(self, struct_types='all', ss_types='min',
                 data_source='Miedema'):
        if isinstance(struct_types, list):
            self.struct_types = struct_types
        else:
            if struct_types == 'all':
                self.struct_types = ['inter', 'amor', 'ss']
            else:
                self.struct_types = [struct_types]

        if isinstance(ss_types, list):
            self.ss_types = ss_types
        else:
            if ss_types == 'all':
                self.ss_types = ['fcc', 'bcc', 'hcp', 'no_latt']
            else:
                self.ss_types = [ss_types]

        self.data_source = data_source
        if self.data_source == 'Miedema':
            self.df_dataset = pd.read_csv(
                os.path.join(data_dir, 'Miedema.csv'), index_col='element')
        else:
            # NOTE comprhys: Not sure why self needed
            raise NotImplementedError(f"data_source {self}{data_source} not implemented yet")
            # raise NotImplementedError('data_source {} not implemented yet'.
            #                           format(self, data_source))

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
        return all([e in self.element_list
                    for e in c.element_composition.elements])

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
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        val = np.array(df_el['valence_electrons'])
        a = np.array(df_el['a_const'])
        r = np.array(df_el['R_const'])
        h_trans = np.array(df_el['H_trans'])

        if struct == 'inter':
            gamma = 8
        elif struct == 'amor':
            gamma = 5
        else:
            gamma = 0

        c_sf = (fracs * np.power(v_molar, 2 / 3) / np.dot(fracs, np.power(v_molar, 2 / 3)))
        f = (c_sf * (1 + gamma * np.power(np.multiply.reduce(c_sf, 0), 2)))[::-1]
        v_a = np.array([np.power(v_molar[0], 2 / 3) * (1 + a[0] * f[0] * (elec[0] - elec[1])),
                        np.power(v_molar[1], 2 / 3) * (1 + a[1] * f[1] * (elec[1] - elec[0]))])
        c_sf_a = fracs * v_a / np.dot(fracs, v_a)
        f_a = (c_sf_a * (1 + gamma * np.power(np.multiply.reduce
                                              (c_sf_a, 0), 2)))[::-1]

        threshold = range(3, 12)
        if (val[0] in threshold) and (val[1] in threshold):
            p = 14.1
            r = 0.
        elif (val[0] not in threshold) and (val[1] not in threshold):
            p = 10.7
            r = 0.
        else:
            p = 12.35
            r = np.multiply.reduce(r, 0) * p
        q = p * 9.4

        eta_ab = (2 * (-p * np.power(elec[0] - elec[1], 2) - r +
                       q * np.power(np.power(n_ws[0], 1 / 3) -
                                    np.power(n_ws[1], 1 / 3), 2)) /
                  reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))

        deltaH_chem = (f_a[0] * fracs[0] * v_a[0] * eta_ab +
                       np.dot(fracs, h_trans))
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
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        compr = np.array(df_el['compressibility'])
        shear_mod = np.array(df_el['shear_modulus'])

        alp = (np.multiply(1.5, np.power(v_molar, 2 / 3)) /
               reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))
        v_a = (v_molar + np.array([alp[0] * (elec[0] - elec[1]) / n_ws[0],
                                   alp[1] * (elec[1] - elec[0]) / n_ws[1]]))
        alp_a = (np.multiply(1.5, np.power(v_a, 2 / 3)) /
                 reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))

        # effective volume in alloy
        vab_a = (v_molar[0] +
                 np.array([alp_a[0] * (elec[0] - elec[1]) / n_ws[0],
                           alp_a[1] * (elec[1] - elec[0]) / n_ws[0]]))
        vba_a = (v_molar[1] +
                 np.array([alp_a[0] * (elec[0] - elec[1]) / n_ws[1],
                           alp_a[1] * (elec[1] - elec[0]) / n_ws[1]]))

        # H_elast A in B
        hab_elast = ((2 * compr[0] * shear_mod[1] *
                      np.power((vab_a[0] - vba_a[0]), 2)) /
                     (4 * shear_mod[1] * vab_a[0] +
                      3 * compr[0] * vba_a[0]))
        # H_elast B in A
        hba_elast = ((2 * compr[1] * shear_mod[0] *
                      np.power((vba_a[1] - vab_a[1]), 2)) /
                     (4 * shear_mod[0] * vba_a[1] +
                      3 * compr[1] * vab_a[1]))

        deltaH_elast = (np.multiply.reduce(fracs, 0) *
                        (fracs[1] * hab_elast + fracs[0] * hba_elast))
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
        val = np.array(df_el['valence_electrons'])
        struct_stab = np.array(df_el['structural_stability'])

        if latt == 'fcc':
            latt_stab_dict = {0.: 0., 1.: 0, 2.: 0, 3.: -2, 4.: -1.5,
                              5.: 9., 5.5: 14., 6.: 11., 7.: -3., 8.: -9.5,
                              8.5: -11., 9.: -9., 10.: -2., 11.: 1.5,
                              12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        elif latt == 'bcc':
            latt_stab_dict = {0.: 0., 1.: 0., 2.: 0., 3.: 2.2, 4.: 2.,
                              5.: -9.5, 5.5: -14.5, 6.: -12., 7.: 4.,
                              8.: 10., 8.5: 11., 9.: 8.5, 10.: 1.5,
                              11.: 1.5, 12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        elif latt == 'hcp':
            latt_stab_dict = {0.: 0., 1.: 0., 2.: 0., 3.: -2.5, 4.: -2.5,
                              5.: 10., 5.5: 15., 6.: 13., 7.: -5.,
                              8.: -10.5, 8.5: -11., 9.: -8., 10.: -1.,
                              11.: 2.5, 12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        else:
            return 0
        latt_stab_dict = collections.OrderedDict(sorted(latt_stab_dict.items(),
                                                        key=lambda t: t[0]))
        # lattice stability of different lattice_types
        val_avg = np.dot(fracs, val)
        val_bd_lower, val_bd_upper = 0, 0
        for key in latt_stab_dict.keys():
            if val_avg - key <= 0:
                val_bd_upper = key
                break
            else:
                val_bd_lower = key

        latt_stab = ((val_avg - val_bd_lower) * latt_stab_dict[val_bd_upper] /
                     (val_bd_upper - val_bd_lower) +
                     (val_bd_upper - val_avg) * latt_stab_dict[val_bd_lower] /
                     (val_bd_upper - val_bd_lower))

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
        melt_point = np.array(df_el['melting_point'])

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
            if struct_type == 'inter':
                deltaH_chem_inter = 0
                for i_inter, el_bin in enumerate(el_bins):
                    deltaH_chem_inter += self.deltaH_chem(el_bin,
                                                          frac_bins[i_inter],
                                                          'inter')
                miedema.append(deltaH_chem_inter)
            # ss: solid solution
            elif struct_type == 'ss':
                deltaH_chem_ss = 0
                deltaH_elast_ss = 0
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_ss += self.deltaH_chem(el_bin, frac_bins[sub_bin], 'ss')
                    deltaH_elast_ss += self.deltaH_elast(el_bin, frac_bins[sub_bin])

                for ss_type in self.ss_types:
                    if ss_type == 'min':
                        deltaH_ss_all = []
                        for latt in ['fcc', 'bcc', 'hcp', 'no_latt']:
                            deltaH_ss_all.append(
                                deltaH_chem_ss + deltaH_elast_ss +
                                self.deltaH_struct(elements, fracs, latt))
                        deltaH_ss_min = min(deltaH_ss_all)
                        miedema.append(deltaH_ss_min)
                    else:
                        deltaH_struct_ss = self.deltaH_struct(elements,
                                                              fracs, ss_type)
                        miedema.append(deltaH_chem_ss + deltaH_elast_ss +
                                       deltaH_struct_ss)
            # amor: amorphous phase
            elif struct_type == 'amor':
                deltaH_chem_amor = 0
                deltaH_topo_amor = self.deltaH_topo(elements, fracs)
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_amor += self.deltaH_chem(el_bin,
                                                         frac_bins[sub_bin],
                                                         'amor')
                miedema.append(deltaH_chem_amor + deltaH_topo_amor)

        # convert kJ/mol to eV/atom. The original Miedema model is in kJ/mol.
        miedema = [deltaH / 96.4853 for deltaH in miedema]
        return miedema

    def feature_labels(self):
        labels = []
        for struct_type in self.struct_types:
            if struct_type == 'ss':
                for ss_type in self.ss_types:
                    labels.append('Miedema_deltaH_ss_' + ss_type)
            else:
                labels.append('Miedema_deltaH_' + struct_type)
        return labels

    def citations(self):
        miedema_citation = (
            '@article{miedema_1988, '
            'title={Cohesion in metals},'
            'author={De Boer, Frank R and Mattens, WCM '
            'and Boom, R and Miedema, AR and Niessen, AK},'
            'year={1988}}')
        zhang_citation = (
            '@article{miedema_zhang_2016, '
            'title={Miedema Calculator: A thermodynamic platform '
            'for predicting formation enthalpies of alloys within '
            'framework of Miedema\'s Theory},'
            'author={R.F. Zhang, S.H. Zhang, Z.J. He, J. Jing and S.H. Sheng},'
            'journal={Computer Physics Communications}'
            'year={2016}}')
        ternary_citation = (
            '@article{miedema_alonso_1990, '
            'title={Glass formation in ternary transition metal alloys},'
            'author={L J Gallego, J A Somoza and J A Alonso},'
            'journal={Journal of Physics: Condensed Matter}'
            'year={1990}}')
        return [miedema_citation, zhang_citation, ternary_citation]

    def implementors(self):
        return ['Qi Wang', 'Alireza Faghaninia']
