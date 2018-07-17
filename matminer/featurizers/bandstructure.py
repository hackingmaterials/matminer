from __future__ import division, unicode_literals, print_function

import numpy as np
from collections import OrderedDict
from numpy.linalg import norm
from scipy.interpolate import griddata

from matminer.featurizers.base import BaseFeaturizer
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure, \
    BandStructureSymmLine
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class BranchPointEnergy(BaseFeaturizer):
    """
    Branch point energy and absolute band edge position.

    Calculates the branch point energy and (optionally) an absolute band
    edge position assuming the branch point energy is the center of the gap

    Args:
        n_vb (int): number of valence bands to include in BPE calc
        n_cb (int): number of conduction bands to include in BPE calc
        calculate_band_edges: (bool) whether to also return band edge
            positions
        atol (float): absolute tolerance when finding equivalent fractional
            k-points in irreducible brillouin zone (IBZ) when weights is None
    """
    def __init__(self, n_vb=1, n_cb=1, calculate_band_edges=True, atol=1e-5):
        self.n_vb = n_vb
        self.n_cb = n_cb
        self.calculate_band_edges = calculate_band_edges
        self.atol = atol

    def featurize(self, bs, target_gap=None, weights=None):
        """
        Args:
            bs (BandStructure): Uniform (not symm line) band structure
            target_gap (float): if set the band gap is scissored to match this
                number
            weights ([float]): if set, its length has to be equal to bs.kpoints
                to explicitly determine the k-point weights when averaging

        Returns:
            (int) branch point energy on same energy scale as BS eigenvalues
        """
        if bs.is_metal():
            raise ValueError("Cannot define a branch point energy for metals!")

        if isinstance(bs, BandStructureSymmLine):
            raise ValueError("BranchPointEnergy works only with uniform (not "
                             "line mode) band structures!")
        vbm = bs.get_vbm()["energy"]
        cbm = bs.get_cbm()["energy"]
        shift = 0.0
        if target_gap:
            # for now, equal shift to VBM / CBM
            shift = (target_gap - (cbm - vbm)) / 2.0
        total_sum_energies = 0
        num_points = 0
        if weights is not None:
            kpt_wts = weights
        else:
            kpt_wts = SpacegroupAnalyzer(bs.structure).get_kpoint_weights(
                [k.frac_coords for k in bs.kpoints], atol=self.atol)

        for spin in bs.bands:
            for kpt_idx in range(len(bs.kpoints)):
                vb_energies = []
                cb_energies = []
                for band_idx in range(bs.nb_bands):
                    e = bs.bands[spin][band_idx][kpt_idx]
                    if e > bs.efermi:
                        cb_energies.append(e + shift)
                    else:
                        vb_energies.append(e - shift)
                vb_energies.sort(reverse=True)
                cb_energies.sort()
                total_sum_energies += (sum(
                    vb_energies[0:self.n_vb]) / self.n_vb + sum(
                    cb_energies[0:self.n_cb]) / self.n_cb) * kpt_wts[
                                          kpt_idx] / 2.0

                num_points += kpt_wts[kpt_idx]

        bpe = total_sum_energies / num_points

        if not self.calculate_band_edges:
            return [bpe]
        return [bpe, vbm-shift, cbm+shift]

    def feature_labels(self):
        """
        Returns ([str]): absolute energy levels as provided in the input
            BandStructure. "absolute" means no reference energy is subtracted
            from branch_point_energy, vbm or cbm.
        """
        return ["branch_point_energy", "vbm_absolute",
                "cbm_absolute"] if self.calculate_band_edges else [
            "branch_point_energy"]

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

    Args:
        kpoints ([1x3 numpy array]): list of fractional coordinates of
                k-points at which energy is extracted.
        find_method (str): the method for finding or interpolating for energy
            at given kpoints. It does nothing if kpoints is None.
            options are:
                'nearest': the energy of the nearest available k-point to
                    the input k-point is returned.
                'linear': the result of linear interpolation is returned
                see the documentation for scipy.interpolate.griddata
        nbands (int): the number of valence/conduction bands to be featurized
    """

    def __init__(self, kpoints=None, find_method='nearest', nbands = 2):
        self.kpoints = kpoints
        self.find_method = find_method
        self.nbands = nbands

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
                p_ex1_norm (float): k-space distance between Gamma point
                    and k-point of VBM
                n_ex1_norm (float): k-space distance between Gamma point
                    and k-point of CBM
                p_ex1_degen: degeneracy of VBM
                n_ex1_degen: degeneracy of CBM
                if kpoints is provided (e.g. for kpoints == [[0.0, 0.0, 0.0]]):
                    n_0.0;0.0;0.0_en: (energy of the first conduction band at
                        [0.0, 0.0, 0.0] - CBM energy)
                    p_0.0;0.0;0.0_en: (energy of the last valence band at
                        [0.0, 0.0, 0.0] - VBM energy)
        """
        if isinstance(bs, dict):
            bs = BandStructure.from_dict(bs)
        if bs.is_metal():
            raise ValueError("Cannot featurize a metallic band structure!")
        bs_kpts = [k.frac_coords for k in bs.kpoints]
        cvd = {'p': bs.get_vbm(), 'n': bs.get_cbm()}
        for itp, tp in enumerate(['p', 'n']):
            cvd[tp]['k'] = bs.kpoints[cvd[tp]['kpoint_index'][0]].frac_coords
            cvd[tp]['bidx'], cvd[tp]['sidx'] = \
                self.get_bindex_bspin(cvd[tp], is_cbm=bool(itp))
            cvd[tp]['Es'] = np.array(bs.bands[cvd[tp]['sidx']][cvd[tp]['bidx']])
        band_gap = bs.get_band_gap()

        # featurize
        feat = OrderedDict()
        feat['band_gap'] = band_gap['energy']
        feat['is_gap_direct'] = band_gap['direct']
        feat['direct_gap'] = min(cvd['n']['Es'] - cvd['p']['Es'])
        for tp in ['p', 'n']:
            feat['{}_ex1_norm'.format(tp)] = norm(cvd[tp]['k'])
            if bs.structure:
                feat['{}_ex1_degen'.format(tp)] = bs.get_kpoint_degeneracy(cvd[tp]['k'])
            else:
                feat['{}_ex1_degen'.format(tp)] = float('NaN')

        if self.kpoints:
            obands = {'n': [], 'p': []}
            for spin in bs.bands:
                for band_idx in range(bs.nb_bands):
                    if max(bs.bands[spin][band_idx]) < bs.efermi:
                        obands['p'].append(bs.bands[spin][band_idx])
                    if min(bs.bands[spin][band_idx]) > bs.efermi:
                        obands['n'].append(bs.bands[spin][band_idx])
            bands = {tp: np.zeros((len(obands[tp]), len(self.kpoints))) for tp in ['p', 'n']}
            for tp in ['p', 'n']:
                for ib, ob in enumerate(obands[tp]):
                    bands[tp][ib, :] = griddata(points=np.array(bs_kpts),
                                   values=np.array(ob) - cvd[tp]['energy'],
                                   xi=self.kpoints, method=self.find_method)
                for ik, k in enumerate(self.kpoints):
                    sorted_band = np.sort(bands[tp][:, ik])
                    if tp == 'p':
                        sorted_band = sorted_band[::-1]
                    for ib in range(self.nbands):
                        k_name = '{}_{};{};{}_en{}'.format(tp, k[0], k[1], k[2], ib+1)
                        try:
                            feat[k_name] = sorted_band[ib]
                        except IndexError:
                            feat[k_name] = float('NaN')
        return list(feat.values())

    def feature_labels(self):
        labels = ['band_gap', 'is_gap_direct', 'direct_gap',
                  'p_ex1_norm', 'p_ex1_degen', 'n_ex1_norm', 'n_ex1_degen']
        if self.kpoints:
            for tp in ['p', 'n']:
                for k in self.kpoints:
                    for ib in range(self.nbands):
                        labels.append('{}_{};{};{}_en{}'.format(
                                tp, k[0], k[1], k[2], ib + 1))
        return labels

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

