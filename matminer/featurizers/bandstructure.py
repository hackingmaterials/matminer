from __future__ import division, unicode_literals, print_function

import numpy as np
from numpy.linalg import norm

from matminer.featurizers.base import BaseFeaturizer
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure, \
    BandStructureSymmLine
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def remove_duplicate_kpoints(kpts, dk=0.001):
    """
    Args:
        kpts ([numpy.array]): list of fractional coordinates of k-points
        dk (float): the tolerance below which two coordinates assumed the same
    Returns ([numpy.array]): list of fractional coordinates of k-points w/o
        duplicates. [0.5, 0.5, 0.0] and [-0.5, -0.5, 0.0] are also duplicates
    """
    rm_list = []
    kdist = [norm(k) for k in kpts]
    ktuple = list(zip(kdist, kpts))
    ktuple.sort(key=lambda x: x[0])
    kpts = [tup[1] for tup in ktuple]
    i = 0
    while i < len(kpts) - 1:
        j = i
        while j < len(kpts) - 1 and ktuple[j + 1][0] - ktuple[i][0] < dk:
            if (abs(kpts[i][0] - kpts[j + 1][0]) < dk or
                abs(kpts[i][0]) == abs(kpts[j + 1][0]) == 0.5) and \
                (abs(kpts[i][1] - kpts[j + 1][1]) < dk or
                abs(kpts[i][1]) == abs(kpts[j + 1][1]) == 0.5) and \
                (abs(kpts[i][2] - kpts[j + 1][2]) < dk or
                abs(kpts[i][2]) == abs(kpts[j + 1][2]) == 0.5):
                rm_list.append(j + 1)
            j += 1
        i += 1
    return np.delete(kpts, rm_list, axis=0)


def get_k_degen(frac_k, rotations):
    """
    returns the degeneracy of a given k-point inside the Brillouin zone.
    Args:
        frac_k (numy.array vector): fractional coordinate of the k-point
        rotations ([numpy.array]): list of rotational symmtetry matrices
    Returns (int): the number of unique equivalent k-points (i.e. degeneracy)
    """
    # all_ks = [np.dot(frac_k, rotations[i]) for i in range(len(rotations))]
    all_ks = np.dot(frac_k, rotations)
    return len(remove_duplicate_kpoints(all_ks))


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
        vbm_k = bs.kpoints[vbm['kpoint_index'][0]].frac_coords
        cbm_k = bs.kpoints[cbm['kpoint_index'][0]].frac_coords

        # featurize
        self.feat = []
        self.feat.append(('band_gap', band_gap['energy']))
        self.feat.append(('is_gap_direct', band_gap['direct']))
        self.feat.append(('direct_gap', min(cbm_ens - vbm_ens)))
        self.feat.append(('p_ex1_norm', norm(vbm_k)))
        self.feat.append(('n_ex1_norm', norm(cbm_k)))
        if bs.structure:
            sg = SpacegroupAnalyzer(bs.structure)
            rotations, _ = sg._get_symmetry()
            self.feat.append(('p_ex1_degen', get_k_degen(vbm_k, rotations)))
            self.feat.append(('n_ex1_degen', get_k_degen(cbm_k, rotations)))

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