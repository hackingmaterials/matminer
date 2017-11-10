from __future__ import division, unicode_literals, print_function

import numpy as np
from numpy.linalg import norm

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import OPSiteFingerprint, get_tet_bcc_motif
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure, \
    BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
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
                p_ex1_degen: degeneracy of VBM
                n_ex1_degen: degeneracy of CBM
        """

        if isinstance(bs, dict):
            bs = BandStructure.from_dict(bs)
        if bs.is_metal():
            raise ValueError("Cannot featurize a metallic band structure!")

        # preparation
        vbm = bs.get_vbm()
        vbm_k = bs.kpoints[vbm['kpoint_index'][0]].frac_coords
        vbm_bidx, vbm_bspin = self.get_bindex_bspin(vbm, is_cbm=False)
        vbm_ens = np.array(bs.bands[vbm_bspin][vbm_bidx])
        cbm = bs.get_cbm()
        cbm_k = bs.kpoints[cbm['kpoint_index'][0]].frac_coords
        cbm_bidx, cbm_bspin = self.get_bindex_bspin(cbm, is_cbm=True)
        cbm_ens = np.array(bs.bands[cbm_bspin][cbm_bidx])
        band_gap = bs.get_band_gap()
        # featurize
        self.feat = []
        self.feat.append(('band_gap', band_gap['energy']))
        self.feat.append(('is_gap_direct', band_gap['direct']))
        self.feat.append(('direct_gap', min(cbm_ens - vbm_ens)))
        self.feat.append(('p_ex1_norm', norm(vbm_k)))
        self.feat.append(('n_ex1_norm', norm(cbm_k)))
        if bs.structure:
            self.feat.append(('p_ex1_degen', bs.get_kpoint_degeneracy(vbm_k)))
            self.feat.append(('n_ex1_degen', bs.get_kpoint_degeneracy(cbm_k)))
        else:
            for prop in ['p_ex1_degen', 'n_ex1_degen']:
                self.feat.append((prop, float.NaN))
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

    def featurize(self, dos, contributors=1, significance_threshold=0.1,
                  coordination_features=True, energy_cutoff=0.5,
                  sampling_resolution=100, gaussian_smear=0.1):
        """
        Args:
            dos (pymatgen CompleteDos or their dict):
                The density of states to featurize. Must be a complete DOS,
                (i.e. contains PDOS and structure, in addition to total DOS)
            contributors (int):
                Sets the number of top contributors to the DOS that are
                returned as features. (i.e. contributors=1 will only return the
                main cb and main vb orbital)
            significance_threshold (float):
                Sets the significance threshold for orbitals in the DOS.
                Does not impact the number of contributors returned. Only
                determines the feature value xbm_significant_contributors.
                The threshold is a fractional value between 0 and 1.
            coordination_features (bool):
                If true, the coordination environment of the PDOS contributors
                will also be returned. Only limited environments are currently
                supported. If the environment is neither, "unrecognized" will
                be returned.
            energy_cutoff (float in eV):
                The extent (into the bands) to sample the DOS
            sampling_resolution (int):
                Number of points to sample DOS
            gaussian_smear (float in eV):
                Gaussian smearing (sigma) around each sampled point in the DOS
        Returns:
             ([float | string]): a list of band structure features.
                List of currently supported features:
                .. xbm_percents: [(float)] fractions that orbitals contribute
                .. xbm_locations: [[(float)]] cartesian locations of orbitals
                .. xbm_species: [(str)] elemental specie of orbitals (ex: 'Ti')
                .. xbm_characters: [(str)] orbital characters (s p d or f)
                .. xbm_coordinations: [(str)] the coordination geometry that
                        the orbitals reside in. (the coordination environment
                        of the site the orbital is associated with)
                .. xbm_significant_contributors: (int) the number of orbitals
                        with contributions above the significance_threshold
        """

        if isinstance(dos, dict):
            dos = CompleteDos.from_dict(dos)

        # preparation
        orbital_scores = DOSFeaturizer.get_cbm_vbm_scores(dos,
                                                          coordination_features,
                                                          energy_cutoff,
                                                          sampling_resolution,
                                                          gaussian_smear)

        orbital_scores.sort(key=lambda x: x['cbm_score'], reverse=True)
        cbm_contributors = orbital_scores[0:contributors]
        cbm_sig_cont = [orb for orb in orbital_scores
                        if orb['cbm_score'] > significance_threshold]
        orbital_scores.sort(key=lambda x: x['vbm_score'], reverse=True)
        vbm_contributors = orbital_scores[0:contributors]
        vbm_sig_cont = [orb for orb in orbital_scores
                        if orb['vbm_score'] > significance_threshold]

        # featurize
        self.feat = []
        self.feat.append(('cbm_percents',
                          [cbm_contributors[i]['cbm_score']
                           for i in range(0, contributors)]))
        self.feat.append(('cbm_locations',
                          [list(cbm_contributors[i]['location'])
                           for i in range(0, contributors)]))
        self.feat.append(('cbm_species',
                          [cbm_contributors[i]['specie'].symbol
                           for i in range(0, contributors)]))
        self.feat.append(('cbm_characters',
                          [str(cbm_contributors[i]['character'])
                           for i in range(0, contributors)]))
        if coordination_features:
            self.feat.append(('cbm_coordinations',
                              [cbm_contributors[i]['coordination']
                               for i in range(0, contributors)]))
        self.feat.append(('cbm_significant_contributors',
                          len(cbm_sig_cont)))
        self.feat.append(('vbm_percents',
                          [vbm_contributors[i]['vbm_score']
                           for i in range(0, contributors)]))
        self.feat.append(('vbm_locations',
                          [list(vbm_contributors[i]['location'])
                           for i in range(0, contributors)]))
        self.feat.append(('vbm_species',
                          [vbm_contributors[i]['specie'].symbol
                           for i in range(0, contributors)]))
        self.feat.append(('vbm_characters',
                          [str(vbm_contributors[i]['character'])
                           for i in range(0, contributors)]))
        if coordination_features:
            self.feat.append(('vbm_coordinations',
                              [vbm_contributors[i]['coordination']
                               for i in range(0, contributors)]))
        self.feat.append(('vbm_significant_contributors',
                          len(vbm_sig_cont)))

        return list(x[1] for x in self.feat)

    def feature_labels(self):
        return list(x[0] for x in self.feat)

    @staticmethod
    def get_cbm_vbm_scores(dos, coordination_features, energy_cutoff,
                           sampling_resolution, gaussian_smear):
        """
        Args:
            dos (pymatgen CompleteDos or their dict):
                The density of states to featurize. Must be a complete DOS,
                (i.e. contains PDOS and structure, in addition to total DOS)
            coordination_features (bool):
                if true, will also return the coordination enviornment of the
                PDOS features
            energy_cutoff (float in eV):
                The extent (into the bands) to sample the DOS
            sampling_resolution (int):
                Number of points to sample DOS
            gaussian_smear (float in eV):
                Gaussian smearing (sigma) around each sampled point in the DOS
        Returns:
            orbital_scores [(dict)]:
                A list of how much each orbital contributes to the partial
                density of states up to energy_cutoff. Dictionary items are:
                .. cbm_score: (float) fractional contribution to conduction band
                .. vbm_score: (float) fractional contribution to valence band
                .. species: (pymatgen Specie) the Specie of the orbital
                .. character: (str) is the orbital character s, p, d, or f
                .. location: [(float)] cartesian coordinates of the orbital
                .. coordination (str) optional-coordination enviornment from op
                                        site feature vector
        """

        cbm, vbm = dos.get_cbm_vbm(tol=0.01)

        structure = dos.structure
        sites = structure.sites

        orbital_scores = []
        for i in range(0, len(sites)):

            # if you desire coordination enviornment as feature
            if coordination_features:
                geometry = get_tet_bcc_motif(structure, i)

            site = sites[i]
            proj = dos.get_site_spd_dos(site)
            for orb in proj:
                # calculate contribution
                energies = [e for e in proj[orb].energies]
                smear_dos = proj[orb].get_smeared_densities(gaussian_smear)
                dos_up = smear_dos[Spin.up]
                dos_down = smear_dos[Spin.down] if Spin.down in smear_dos\
                    else smear_dos[Spin.up]
                dos_total = [sum(id) for id in zip(dos_up, dos_down)]

                vbm_score = 0
                vbm_space = np.linspace(vbm, vbm - energy_cutoff,
                                        num=sampling_resolution)
                for e in vbm_space:
                    vbm_score += np.interp(e, energies, dos_total)

                cbm_score = 0
                cbm_space = np.linspace(cbm, cbm + energy_cutoff,
                                        num=sampling_resolution)
                for e in cbm_space:
                    cbm_score += np.interp(e, energies, dos_total)

                # add orbital scores to list
                orbital_score = {
                    'cbm_score': cbm_score,
                    'vbm_score': vbm_score,
                    'specie': site.specie,
                    'character': orb,
                    'location': site.coords}
                if coordination_features:
                    orbital_score['coordination'] = geometry
                orbital_scores.append(orbital_score)

        # normalize by total contribution
        total_cbm = sum([orbital_scores[i]['cbm_score'] for i in
                         range(0, len(orbital_scores))])
        total_vbm = sum([orbital_scores[i]['vbm_score'] for i in
                         range(0, len(orbital_scores))])
        for orbital in orbital_scores:
            orbital['cbm_score'] = orbital['cbm_score'] / total_cbm
            orbital['vbm_score'] = orbital['vbm_score'] / total_vbm

        return orbital_scores

    def implementors(self):
        return ['Maxwell Dylla', 'Anubhav Jain']
