import numpy as np
from collections import OrderedDict
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition import BandCenter
from pymatgen import Spin
from pymatgen.electronic_structure.dos import CompleteDos, FermiDos


class DOSFeaturizer(BaseFeaturizer):
    """
    Significant character and contribution of the density of state from a
    CompleteDos, object. Contributors are the atomic orbitals from each site
    within the structure. This underlines the importance of dos.structure.

    Args:
        contributors (int):
            Sets the number of top contributors to the DOS that are
            returned as features. (i.e. contributors=1 will only return the
            main cb and main vb orbital)
        decay_length (float in eV):
            The dos is sampled by an exponential decay function. this parameter
            sets the decay length of the exponential. Three times the decay
            length corresponds to 10% sampling strength. There is a hard cutoff
            at five times the decay length (1% sampling strength)
        sampling_resolution (int):
            Number of points to sample DOS
        gaussian_smear (float in eV):
            Gaussian smearing (sigma) around each sampled point in the DOS

    Returns (featurize returns [float] and featurize_labels returns [str]):
        xbm_score_i (float): fractions of ith contributor orbital
        xbm_location_i (str): fractional coordinate of ith contributor/site
        xbm_character_i (str): character of ith contributor (s, p, d, f)
        xbm_specie_i (str): elemental specie of ith contributor (ex: 'Ti')
        xbm_hybridization (int): the ammount of hybridization at the band edge
            characterized by an entropy score (x ln x). the hybridization score
            is larger for a greater number of significant contributors
    """
    def __init__(self, contributors=1, decay_length=0.1,
                 sampling_resolution=100, gaussian_smear=0.05):
        self.contributors = contributors
        self.decay_length = decay_length
        self.sampling_resolution = sampling_resolution
        self.gaussian_smear = gaussian_smear

    def featurize(self, dos):
        """
        Args:
            dos (pymatgen CompleteDos or their dict):
                The density of states to featurize. Must be a complete DOS,
                (i.e. contains PDOS and structure, in addition to total DOS)
                and must contain the structure.
        """
        if isinstance(dos, dict):
            dos = CompleteDos.from_dict(dos)
        if dos.structure is None:
            raise ValueError('The input dos must contain the structure.')

        orbscores = get_cbm_vbm_scores(dos, self.decay_length,
                                       self.sampling_resolution,
                                       self.gaussian_smear)

        feat = OrderedDict()
        for ex in ['cbm', 'vbm']:
            orbscores.sort(key=lambda x: x['{}_score'.format(ex)],
                           reverse=True)
            scores = np.array([s['{}_score'.format(ex)] for s in orbscores])
            feat['{}_hybridization'.format(ex)] = - np.sum(
                scores * np.log(scores + 1e-10))  # avoid log(0)

            i = 0
            while i < self.contributors:
                sd = orbscores[i]
                if i < len(orbscores):
                    for p in ['character', 'specie']:
                        feat['{}_{}_{}'.format(ex, p, i + 1)] = sd[p]
                    feat['{}_location_{}'.format(ex, i + 1)] =\
                        '{};{};{}'.format(sd['location'][0], sd['location'][1],
                                          sd['location'][2])
                    feat['{}_score_{}'.format(ex, i + 1)] =\
                        float(sd['{}_score'.format(ex)])
                else:
                    for p in ['character', 'specie', 'location', 'score']:
                        feat['{}_{}_{}'.format(ex, p, i + 1)] = float('NaN')
                i += 1

        return list(feat.values())

    def feature_labels(self):
        """
        Returns ([str]): list of names of the features. See the docs for the
            featurize method for more information.
        """
        labels = []
        for ex in ['cbm', 'vbm']:
            labels.append('{}_hybridization'.format(ex))
            i = 0
            while i < self.contributors:
                for p in ['character', 'specie', 'location', 'score']:
                    labels.append('{}_{}_{}'.format(ex, p, i + 1))
                i += 1

        return labels

    def implementors(self):
        return ['Maxwell Dylla', 'Alireza Faghaninia', 'Anubhav Jain']


class DopingFermi(BaseFeaturizer):
    """
    The fermi level (w.r.t. selected reference energy) associated with a
    specified carrier concentration (1/cm3) and temperature. This featurizar
    requires the total density of states and structure. The Structure
    as dos.structure (e.g. in CompleteDos) is required by FermiDos class.

    Args:
        dopings ([float]): list of doping concentrations 1/cm3. Note that a
            negative concentration is treated as electron majority carrier
            (n-type) and positive for holes (p-type)
        eref (str or int or float): energy alignment reference. Defaults
            to midgap (equilibrium fermi). A fixed number can also be used.
            str options: "midgap", "vbm", "cbm", "dos_fermi", "band_center"
        T (float): absolute temperature in Kelvin
        return_eref: if True, instead of aligning the fermi levels based
            on eref, it (eref) will be explicitly returned as a feature

    Returns (featurize returns [float] and featurize_labels returns [str]):
        examples:
            fermi_c-1e+20T300 (float): the fermi level for the electron
                concentration of 1e20 and the temperature of 300K.
            fermi_c1e+18T600 (float): fermi level for the hole concentration
                of 1e18 and the temperature of 600K.
            midgap eref (float): if return_eref==True then eref (midgap here)
                energy is returned. In this case, fermi levels are absolute as
                opposed to relative to eref (i.e. if not return_eref)
    """
    def __init__(self, dopings=None, eref="midgap", T=300, return_eref=False):
        self.dopings = dopings or [-1e20, 1e20]
        self.eref = eref
        self.T = T
        self.return_eref = return_eref
        self.BC = BandCenter()

    def featurize(self, dos, bandgap=None):
        """
        Args:
            dos (pymatgen Dos, CompleteDos or FermiDos):
            bandgap (float): for example the experimentally measured band gap
                or one that is calculated via more accurate methods than the
                one used to generate dos. dos will be scissored to have the
                same electronic band gap as bandgap.

        Returns ([float]): features are fermi levels in eV at the given
            concentrations and temperature + eref in eV if return_eref
        """
        dos = FermiDos(dos, bandgap=bandgap)
        feats = []
        eref = 0.0
        for c in self.dopings:
            fermi = dos.get_fermi(c=c, T=self.T, nstep=50)
            if isinstance(self.eref, str):
                if self.eref == "dos_fermi":
                    eref = dos.efermi
                elif self.eref in ["midgap", "vbm", "cbm"]:
                    ecbm, evbm = dos.get_cbm_vbm()
                    if self.eref == "midgap":
                        eref = (evbm + ecbm) / 2.0
                    elif self.eref == "vbm":
                        eref = evbm
                    elif self.eref == "cbm":
                        eref = ecbm
                elif self.eref == "band center":
                    eref = self.BC.featurize(dos.structure.composition)[0]
                else:
                    raise ValueError('Unsupported "eref": {}'.format(self.eref))
            else:
                eref = self.eref
            if not self.return_eref:
                fermi -= eref
            feats.append(fermi)
        if self.return_eref:
            feats.append(eref)
        return feats

    def feature_labels(self):
        """
        Returns ([str]): list of names of the features generated by featurize
            example: "fermi_c-1e+20T300" that is the fermi level for the
            electron concentration of 1e20 (c-1e+20) and temperature of 300K.
        """
        labels = []
        for c in self.dopings:
            labels.append("fermi_c{}T{}".format(c, self.T))
        if self.return_eref:
            labels.append("{} eref".format(self.eref))
        return labels

    def implementors(self):
        return ["Alireza Faghaninia"]

    def citations(self):
        return []


class Hybridization(BaseFeaturizer):
    """
    quantify s/p/d/f orbital character and their hybridizations at band edges

    Args:
        decay_length (float in eV):
            The dos is sampled by an exponential decay function. this parameter
            sets the decay length of the exponential. Three times the decay
            length corresponds to 10% sampling strength. There is a hard cutoff
            at five times the decay length (1% sampling strength)
        sampling_resolution (int):
            Number of points to sample DOS
        gaussian_smear (float in eV):
            Gaussian smearing (sigma) around each sampled point in the DOS
        species ([str]): the species for which orbital contributions are
            separately returned.

    Returns (featurize returns [float] and featurize_labels returns [str]):
        set of orbitals contributions and hybridizations. If species, then also
        individual contributions from given species. Examples:
            cbm_s (float): s-orbital character of the cbm up to energy_cutoff
            vbm_sp (float): sp-hybridization at the vbm edge. Minimum is 0
                or no hybridization (e.g. all s or vbm_s==1) and 1.0 is
                maximum hybridization (i.e. vbm_s==0.5, vbm_p==0.5)
            cbm_Si_p (float): p-orbital character of Si
    """
    def __init__(self, decay_length=0.1, sampling_resolution=100,
                 gaussian_smear=0.05, species=None):
        self.decay_length = decay_length
        self.sampling_resolution = sampling_resolution
        self.gaussian_smear = gaussian_smear
        self.species = species or []

    def featurize(self, dos, decay_length=None):
        """
        takes in the density of state and return the orbitals contributions
        and hybridizations.

        Args:
            dos (pymatgen CompleteDos): note that dos.structure is required
            decay_length (float or None): if set, it overrides the instance
                variable self.decay_length.

        Returns ([float]): features, see class doc for more info
        """
        decay_length = decay_length or self.decay_length
        if isinstance(dos, dict):
            dos = CompleteDos.from_dict(dos)
        if dos.structure is None:
            raise ValueError('The input dos must contain the structure.')

        orbscores = get_cbm_vbm_scores(dos,
                                       decay_length,
                                       self.sampling_resolution,
                                       self.gaussian_smear)
        feat = OrderedDict()
        for ex in ['cbm', 'vbm']:
            for orbital in ['s', 'p', 'd', 'f']:
                feat['{}_{}'.format(ex, orbital)] = 0.0
                for specie in self.species:
                    feat['{}_{}_{}'.format(ex, specie, orbital)] = 0.0
            for hybrid in ['sp', 'sd', 'sf', 'pd', 'pf', 'df']:
                feat['{}_{}'.format(ex, hybrid)] = 0.0

        for contrib in orbscores:
            character = contrib['character']
            feat['cbm_{}'.format(character)] += contrib['cbm_score']
            feat['vbm_{}'.format(character)] += contrib['vbm_score']
            for specie in self.species:
                if contrib['specie'] == specie:
                    feat['cbm_{}_{}'.format(specie, character)] += contrib[
                        'cbm_score']
                    feat['vbm_{}_{}'.format(specie, character)] += contrib[
                        'vbm_score']

        for ex in ['cbm', 'vbm']:
            for hybrid in ['sp', 'sd', 'sf', 'pd', 'pf', 'df']:
                orb1 = feat['{}_{}'.format(ex, hybrid[0])]
                orb2 = feat['{}_{}'.format(ex, hybrid[1])]
                feat['{}_{}'.format(ex, hybrid)] = (orb1 * orb2) * 4.0  # 4x so max=1.0
        return list(feat.values())

    def feature_labels(self):
        """
        Returns ([str]): feature names starting with the extremum (cbm or vbm)
        followed by either s,p,d,f orbital to show normalized contribution
        or a pair showing their hybridization or contribution of an element.
        See the class docs for examples.
        """
        labels = []
        for ex in ['cbm', 'vbm']:
            for orbital in ['s', 'p', 'd', 'f']:
                labels.append('{}_{}'.format(ex, orbital))
                for specie in self.species:
                    labels.append('{}_{}_{}'.format(ex, specie, orbital))
            for hybrid in ['sp', 'sd', 'sf', 'pd', 'pf', 'df']:
                labels.append('{}_{}'.format(ex, hybrid))
        return labels

    def implementors(self):
        return ['Alireza Faghaninia', 'Anubhav Jain', 'Maxwell Dylla']


def get_cbm_vbm_scores(dos, decay_length, sampling_resolution, gaussian_smear):
    """
    Quantifies the contribution of all atomic orbitals (s/p/d/f) from all
    crystal sites to the conduction band minimum (CBM) and the valence band
    maximum (VBM). An exponential decay function is used to sample the DOS.
    An example use may be sorting the output based on cbm_score or vbm_score.

    Args:
        dos (pymatgen CompleteDos or their dict):
            The density of states to featurize. Must be a complete DOS,
            (i.e. contains PDOS and structure, in addition to total DOS)
        decay_length (float in eV):
            The dos is sampled by an exponential decay function. this parameter
            sets the decay length of the exponential. Three times the decay
            length corresponds to 10% sampling strength. There is a hard cutoff
            at five times the decay length (1% sampling strength)
        sampling_resolution (int):
            Number of points to sample DOS
        gaussian_smear (float in eV):
            Gaussian smearing (sigma) around each sampled point in the DOS

    Returns:
        orbital_scores [(dict)]:
            A list of how much each orbital contributes to the partial
            density of states near the band edge. Dictionary items are:
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
            vbm_space = np.linspace(vbm, vbm - (5. * decay_length),
                                    num=sampling_resolution)
            for e in vbm_space:
                vbm_score += (np.interp(e, energies, dos_total) *
                              np.exp(-(vbm - e) * decay_length))
            cbm_score = 0
            cbm_space = np.linspace(cbm, cbm + (5. * decay_length),
                                    num=sampling_resolution)
            for e in cbm_space:
                cbm_score += (np.interp(e, energies, dos_total) *
                              np.exp(-(e - cbm) * decay_length))

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
        orbital['cbm_score'] /= total_cbm
        orbital['vbm_score'] /= total_vbm
    return orbital_scores
