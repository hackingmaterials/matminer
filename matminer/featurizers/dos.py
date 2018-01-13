import numpy as np
from collections import OrderedDict
from matminer.featurizers.base import BaseFeaturizer
from pymatgen import Spin
from pymatgen.electronic_structure.dos import CompleteDos


class DOSFeaturizer(BaseFeaturizer):
    """
    Featurizes a pymatgen density of states, CompleteDos, object.
    """

    def __init__(self, contributors=1, significance_threshold=0.1,
                 energy_cutoff=0.5, sampling_resolution=100, gaussian_smear=0.1):
        """
        Args:
            contributors (int):
                Sets the number of top contributors to the DOS that are
                returned as features. (i.e. contributors=1 will only return the
                main cb and main vb orbital)
            significance_threshold (float):
                Sets the significance threshold for orbitals in the DOS.
                Does not impact the number of contributors returned. Only
                determines the feature value xbm_significant_contributors.
                The threshold is a fractional value between 0 and 1.
            energy_cutoff (float in eV):
                The extent (into the bands) to sample the DOS
            sampling_resolution (int):
                Number of points to sample DOS
            gaussian_smear (float in eV):
                Gaussian smearing (sigma) around each sampled point in the DOS
        """
        self.contributors = contributors
        self.significance_threshold = significance_threshold
        self.energy_cutoff = energy_cutoff
        self.sampling_resolution = sampling_resolution
        self.gaussian_smear = gaussian_smear

    def featurize(self, dos):
        """
        Args:
            dos (pymatgen CompleteDos or their dict):
                The density of states to featurize. Must be a complete DOS,
                (i.e. contains PDOS and structure, in addition to total DOS)
                and must contain the structure.

        Returns:
            xbm_score_i (float): fractions of ith contributor orbital
            xbm_location_i (str): fractional coordinate of ith contributor.
                For example, '0.0;0.0;0.0' if Gamma
            xbm_specie_i: (str) elemental specie of ith contributor (ex: 'Ti')
            xbm_character_i: (str) orbital character of ith contributor (s p d or f)
            xbm_nsignificant: (int) the number of orbitals with contributions
                above the significance_threshold
        """

        if isinstance(dos, dict):
            dos = CompleteDos.from_dict(dos)
        if dos.structure is None:
            raise ValueError('The input dos must contain the structure.')

        orbscores = get_cbm_vbm_scores(dos, self.energy_cutoff,
                                       self.sampling_resolution,
                                       self.gaussian_smear)

        feat = OrderedDict()
        for ex in ['cbm', 'vbm']:
            orbscores.sort(key=lambda x: x['{}_score'.format(ex)], reverse=True)
            scores = np.array([s['{}_score'.format(ex)] for s in orbscores])
            feat['{}_nsignificant'.format(ex)] = len(scores[scores > self.significance_threshold])

            i = 0
            while i < self.contributors:
                sd = orbscores[i]
                if i < len(orbscores):
                    for p in ['character', 'specie']:
                        feat['{}_{}_{}'.format(ex, p, i + 1)] = sd[p]
                    feat['{}_location_{}'.format(ex, i + 1)] = '{};{};{}'.format(
                        sd['location'][0], sd['location'][1], sd['location'][2])
                    feat['{}_score_{}'.format(ex, i + 1)] = float(sd['{}_score'.format(ex)])
                else:
                    for p in ['character', 'specie', 'location', 'score']:
                        feat['{}_{}_{}'.format(ex, p, i + 1)] = float('NaN')
                i += 1

        return list(feat.values())

    def feature_labels(self):
        labels = []
        for ex in ['cbm', 'vbm']:
            labels.append('{}_nsignificant'.format(ex))
            i = 0
            while i < self.contributors:
                for p in ['character', 'specie', 'location', 'score']:
                    labels.append('{}_{}_{}'.format(ex, p, i + 1))
                i += 1

        return labels

    def implementors(self):
        return ['Maxwell Dylla', 'Alireza Faghaninia', 'Anubhav Jain']


def get_cbm_vbm_scores(dos, energy_cutoff, sampling_resolution, gaussian_smear):
    """
    Quantifies the strength of the contribution of all orbitals of various
        species/sites to the conduction band minimum (CBM) and the valence band
        maximum (VBM) up to energy_cutoff inside the bands from the CBM/VBM.
        An example use of the output may be sorting it based on cbm_score
        or vbm_score.
    Args:
        dos (pymatgen CompleteDos or their dict):
            The density of states to featurize. Must be a complete DOS,
            (i.e. contains PDOS and structure, in addition to total DOS)
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
            .. location: [(float)] fractional coordinates of the orbital
    """

    cbm, vbm = dos.get_cbm_vbm(tol=0.01)
    structure = dos.structure
    sites = structure.sites

    orbital_scores = []
    for i in range(0, len(sites)):
        site = sites[i]
        proj = dos.get_site_spd_dos(site)
        for orb in proj:
            # calculate contribution
            energies = [e for e in proj[orb].energies]
            smear_dos = proj[orb].get_smeared_densities(gaussian_smear)
            dos_up = smear_dos[Spin.up]
            dos_down = smear_dos[Spin.down] if Spin.down in smear_dos \
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
                'specie': str(site.specie),
                'character': str(orb),
                'location': list(site.frac_coords)}
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
