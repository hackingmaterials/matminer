from __future__ import division, unicode_literals, print_function

import numpy as np

from matminer.featurizers.base import BaseFeaturizer
from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def norm(vector):
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5

# TODO: add a unit test

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
            bs: (BandStructure)

        Returns:
            (int) branch point energy on same energy scale as BS eigenvalues
        """
        if bs.is_metal():
            raise ValueError("Cannot define a branch point energy for metals!")

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

        return ["branch point energy", "vbm_absolute",
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
    Featurizes a pymatgen band structure object. If also the structure is fed
        to featurize method, additional features will be returned.
    Args:
        nband (int): the number of the valence or conduction bands to be
            included when featurizing the band structure
        n_extrem (int): the number of extrema (in each band) to be included
    """
    def __init__(self, nband=1, n_extrem=1):
        self.n_extrem = n_extrem
        self.nband = nband

    def featurize(self, bs):
        """
        Args:
            bs (pymatgen BandStructure or BandStructureSymmLine or their dict)
                note that if bs.structure, more features will be generated.
        Returns ([float]):
            a list of band structure features. If not bs.structure, the
                features that require the structure will be returned as NaN.
            List of currently supported features:
                band_gap (eV): the difference between the CBM and VBM energy
                is_gap_direct (0.0|1.0): whether the band gap is direct or not
                direct_gap (eV): the minimum direct distance of the last
                    valence band and the first conduction band
                *_ex#_en (eV): for example p_ex2_en is the absolute value of
                    the energy of the second valence band maximum w.r.t. VBM
                *_ex#_norm (float): e.g. p_ex1_norm is norm of the fractional
                     coordinates of the 1st valence band maximum (VBM) k-point
                NA! *_ex#_degen (float): the band degeneracy of the extremum
                NA! *_ex#_mass (float): the effective mass of the extremum

        """
        self.feat = []
        if isinstance(bs, dict):
            bsd = bs
            structure = None
        else:
            bsd = bs.as_dict()
            structure = bs.structure

        if bsd['is_metal']:
            raise ValueError("Cannot featurize a metallic band structure!")

        # preparation
        cbm = bsd['cbm']
        vbm = bsd['vbm']
        vbm_bidx, vbm_bspin = self.get_bindex_bspin(vbm)
        cbm_bidx, cbm_bspin = self.get_bindex_bspin(cbm)
        vbm_ens = np.array(bsd['bands'][str(vbm_bspin)][vbm_bidx])
        cbm_ens = np.array(bsd['bands'][str(cbm_bspin)][cbm_bidx])

        # featurize
        self.feat.append(('band_gap', bsd['band_gap']['energy']))
        self.feat.append(('is_gap_direct', bsd['band_gap']['direct']))
        self.feat.append(('direct_gap', min(cbm_ens - vbm_ens)))
        self.feat.append(('p_ex1_norm',
                norm(bsd['kpoints'][vbm['kpoint_index'][0]])))
        self.feat.append(('n_ex1_norm',
                norm(bsd['kpoints'][cbm['kpoint_index'][-1]])))
        if structure:
            pass
            # additional features such as n_ex_degen will be generated here

        return list(list(zip(*self.feat))[1])

    def feature_labels(self):
        return list(list(zip(*self.feat))[0])

    @staticmethod
    def get_bindex_bspin(extremum):
        try:
            bidx = extremum["band_index"][str(Spin.up)][-1]
            bspin = int(Spin.up)
        except KeyError:
            bidx = extremum["band_index"][str(Spin.down)][-1]
            bspin = int(Spin.down)
        return bidx, bspin

    def citations(self):
        return ['@article{in_progress, title={{In progress}} year={2017}}']

    def implementors(self):
        return ['Alireza Faghaninia', 'Anubhav Jain']