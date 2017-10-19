from __future__ import division, unicode_literals, print_function

import numpy as np
from numpy.linalg import norm

from matminer.featurizers.base import BaseFeaturizer
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure, \
    BandStructureSymmLine
from pymatgen.electronic_structure.dos import Dos
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class BranchPointEnergy(BaseFeaturizer):
    def __init__(self, n_vb=1, n_cb=1, calculate_band_edges=True):
        """
        Calculates the branch point energy and (optionally) an absolute band
        edge position assuming the branch point energy is the center of the gap

        Args:
            n_vb: (int) number of valence bands to include in BPE calc
            n_cb: (int) number of conduction bands to include in BPE calc
            calculate_band_edges: (bool) whether to also return band edge
                positions
        """
        self.n_vb = n_vb
        self.n_cb = n_cb
        self.calculate_band_edges = calculate_band_edges

    def featurize(self, bs, target_gap=None):
        """
        Args:
            bs: (BandStructure) Uniform (not symm line) band structure

        Returns:
            (int) branch point energy on same energy scale as BS eigenvalues
        """
        if bs.is_metal():
            raise ValueError("Cannot define a branch point energy for metals!")

        if isinstance(bs, BandStructureSymmLine):
            raise ValueError("BranchPointEnergy works only with uniform (not "
                             "line mode) band structures!")

        total_sum_energies = 0
        num_points = 0

        kpt_wts = SpacegroupAnalyzer(bs.structure).get_kpoint_weights(
            [k.frac_coords for k in bs.kpoints])

        for spin in bs.bands:
            for kpt_idx in range(len(bs.kpoints)):
                vb_energies = []
                cb_energies = []
                for band_idx in range(bs.nb_bands):
                    e = bs.bands[spin][band_idx][kpt_idx]
                    if e > bs.efermi:
                        cb_energies.append(e)
                    else:
                        vb_energies.append(e)
                vb_energies.sort(reverse=True)
                cb_energies.sort()
                total_sum_energies += (sum(
                    vb_energies[0:self.n_vb]) / self.n_vb + sum(
                    cb_energies[0:self.n_cb]) / self.n_cb) * kpt_wts[
                                          kpt_idx] / 2

                num_points += kpt_wts[kpt_idx]

        bpe = total_sum_energies / num_points

        if not self.calculate_band_edges:
            return [bpe]

        vbm = bs.get_vbm()["energy"]
        cbm = bs.get_cbm()["energy"]
        shift = 0
        if target_gap:
            # for now, equal shift to VBM / CBM
            shift = (target_gap - (cbm - vbm)) / 2

        return [bpe, (vbm - bpe - shift), (cbm - bpe + shift)]

    def feature_labels(self):

        return ["branch_point_energy", "vbm_absolute",
                "cbm_absolute"] if self.calculate_band_edges else [
            "branch point energy"]

    def citations(self):
        return ["@article{Schleife2009, author = {Schleife, A. and Fuchs, F. "
                "and R{\"{o}}dl, C. and Furthm{\"{u}}ller, J. and Bechstedt, "
                "F.}, doi = {10.1063/1.3059569}, isbn = {0003-6951}, issn = "
                "{00036951}, journal = {Applied Physics Letters}, number = {1},"
                " pages = {2009--2011}, title = {{Branch-point energies and "
                "band discontinuities of III-nitrides and III-/II-oxides "
                "from quasiparticle band-structure calculations}}, volume = "
                "{94}, year = {2009}}"]

    def implementors(self):
        return ["Anubhav Jain"]


class BandFeaturizer(BaseFeaturizer):
    """
    Featurizes a pymatgen band structure object.
    """
    def __init__(self):
        pass

    def featurize(self, bs):
        """
        Args:
            bs (pymatgen BandStructure or BandStructureSymmLine or their dict):
                The band structure to featurize. To obtain all features, bs
                should include the structure attribute.
        Returns:
             ([float]): a list of band structure features. If not bs.structure,
                features that require the structure will be returned as NaN.
            List of currently supported features:
                band_gap (eV): the difference between the CBM and VBM energy
                is_gap_direct (0.0|1.0): whether the band gap is direct or not
                direct_gap (eV): the minimum direct distance of the last
                    valence band and the first conduction band
                p_ex1_norm (Angstrom^-1): k-space distance between Gamma point
                    and k-point of VBM
                n_ex1_norm (Angstrom^-1): k-space distance between Gamma point
                    and k-point of CBM
        """

        if isinstance(bs, dict):
            bs = BandStructure.from_dict(bs)
        if bs.is_metal():
            raise ValueError("Cannot featurize a metallic band structure!")

        # preparation
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()
        band_gap = bs.get_band_gap()
        vbm_bidx, vbm_bspin = self.get_bindex_bspin(vbm, is_cbm=False)
        cbm_bidx, cbm_bspin = self.get_bindex_bspin(cbm, is_cbm=True)
        vbm_ens = np.array(bs.bands[vbm_bspin][vbm_bidx])
        cbm_ens = np.array(bs.bands[cbm_bspin][cbm_bidx])

        # featurize
        self.feat = []
        self.feat.append(('band_gap', band_gap['energy']))
        self.feat.append(('is_gap_direct', band_gap['direct']))
        self.feat.append(('direct_gap', min(cbm_ens - vbm_ens)))
        self.feat.append(('p_ex1_norm',
                norm(bs.kpoints[vbm['kpoint_index'][0]].frac_coords)))
        self.feat.append(('n_ex1_norm',
                norm(bs.kpoints[cbm['kpoint_index'][0]].frac_coords)))

        return list(x[1] for x in self.feat)

    def feature_labels(self):
        return list(x[0] for x in self.feat)

    @staticmethod
    def get_bindex_bspin(extremum, is_cbm):
        """
        Returns the band index and spin of band extremum

        Args:
            extremum (dict): dictionary containing the CBM/VBM, i.e. output of
                Bandstructure.get_cbm()
            is_cbm (bool): whether the extremum is the CBM or not
        """

        idx = int(is_cbm) - 1  # 0 for CBM and -1 for VBM
        try:
            bidx = extremum["band_index"][Spin.up][idx]
            bspin = Spin.up
        except IndexError:
            bidx = extremum["band_index"][Spin.down][idx]
            bspin = Spin.down
        return bidx, bspin

    def citations(self):
        return ['@article{in_progress, title={{In progress}} year={2017}}']

    def implementors(self):
        return ['Alireza Faghaninia', 'Anubhav Jain']


class DOSFeaturizer(BaseFeaturizer):
    """
    Featurizes a pymatgen dos object.
    """

    def __init__(self):
        pass

    def featurize(self, dos, band_depth=1, sampling_resolution=100, smear=0.25):
        """
        Args:
            dos (pymatgen DOS or their dict):
                The density of states to featurize. To obtain all features, dos
                should include the structure attribute.
            band_depth: The extent into the bands to sample the DOS
            sampling_resolution: Number of points to sample DOS at
            smear: Smearing parameter for the DOS
        Returns:
             ([float]): a list of band structure features.
            List of currently supported features:
                cbm_main_percent: (float) fraction that the main cb orbital contributes
                vbm_main_percent: (float) fraction that the main vb orbital contributes
                cb_main_character: ([0.0|1.0]) main orbital present is s p or d
                vb_main_character: ([0.0|1.0]) main orbital present is s p or d
                                    s = [1.0, 0.0, 0.0]
                                    p = [0.0, 1.0, 0.0]
                                    d = [0.0, 0.0, 1.0]
        """

        if isinstance(dos, dict):
            dos = Dos.from_dict(dos)

        # preparation
        cbm_scores, vbm_scores = self.get_cbm_vbm_scores(
            dos, band_depth, sampling_resolution, smear)

        cbm_pdos = list(cbm_scores.items())
        cbm_pdos.sort(key=lambda x: x[1])
        vbm_pdos = list(vbm_scores.items())
        vbm_pdos.sort(key=lambda x: x[1])

        cbm_main = cbm_pdos[-1]
        vbm_main = vbm_pdos[-1]

        # featurize
        self.feat = []
        self.feat.append(('cbm_main_percent', cbm_main[1]))
        self.feat.append(('vbm_main_percent', vbm_main[1]))
        self.feat.append(('cbm_main_character', self.get_orbital_character(cbm_main)))
        self.feat.append(('vbm_main_character', self.get_orbital_character(vbm_main)))

        return list(x[1] for x in self.feat)

    def feature_labels(self):
        return list(x[0] for x in self.feat)

    @staticmethod
    def get_orbital_character(PDOS):
        """
        check which orbital character (s, p, d) is present
        Args:
            PDOS: (touple) that gives pdos character and ammount
                ex: ('Ti:s', 0.75)
        return:
            [(int)]: a binary list which represents the pdos character
            s = [1.0, 0.0, 0.0]
            p = [0.0, 1.0, 0.0]
            d = [0.0, 0.0, 1.0]
        """

        if PDOS[0][-1] == 's':
            return [1., 0., 0.]
        elif PDOS[0][-1] == 'p':
            return [0., 1., 0.]
        else:
            return [0., 0., 1.]

    @staticmethod
    def get_cbm_vbm_scores(dos, band_depth, sampling_resolution, smear):
        """
        Args:
            dos (pymatgen DOS or their dict):
                The density of states to featurize. To obtain all features, dos
                should include the structure attribute.
            band_depth: The extent into the bands to sample the DOS
            sampling_resolution: Number of points to sample DOS at
            smear: smearing parameter for the DOS
        Returns:
            density of states scores (cbm_scores, vbm_scores):
                the partial density of states of both band extremum
                normalized by the total density of states in the sampled
                range
        """

        cbm, vbm = dos.get_cbm_vbm(tol=0.01)

        cbm_scores = {}
        vbm_scores = {}
        for el in dos.structure.composition.elements:
            proj = dos.get_element_spd_dos(el)
            for orb in proj:
                energies = [e for e in proj[orb].energies]
                smear_dos = proj[orb].get_smeared_densities(smear)
                dos_up = smear_dos[Spin.up]
                dos_down = smear_dos[Spin.down] if Spin.down in smear_dos\
                           else smear_dos[Spin.up]
                dos_total = [sum(id) for id in zip(dos_up, dos_down)]

                vbm_score = 0
                vbm_space = np.linspace(vbm, vbm - band_depth,
                                        num=sampling_resolution)
                for e in vbm_space:
                    vbm_score += np.interp(e, energies, dos_total)

                cbm_score = 0
                cbm_space = np.linspace(cbm, cbm + band_depth,
                                        num=sampling_resolution)
                for e in cbm_space:
                    cbm_score += np.interp(e, energies, dos_total)

                cbm_scores["{}:{}".format(el.symbol, str(orb))] = cbm_score
                vbm_scores["{}:{}".format(el.symbol, str(orb))] = vbm_score

        total_cbm = sum([cbm_scores[key] for key in cbm_scores.keys()])
        total_vbm = sum([vbm_scores[key] for key in vbm_scores.keys()])

        for key in cbm_scores.keys():
            cbm_scores[key] = cbm_scores[key] / total_cbm

        for key in vbm_scores.keys():
            vbm_scores[key] = vbm_scores[key] / total_vbm

        return (cbm_scores, vbm_scores)

    def implementors(self):
        return ['Maxwell Dylla', 'Anubhav Jain']
