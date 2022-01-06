from collections import OrderedDict

import numpy as np
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import CompleteDos, FermiDos

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition import BandCenter


class SiteDOS(BaseFeaturizer):
    """
    report the fractional s/p/d/f dos for a particular site. a CompleteDos
    object is required because knowledge of the structure is needed. this
    featurizer will work for metals as well as semiconductors. if the dos is a
    semiconductor, cbm and vbm will correspond to the two respective band
    edges. if the dos is a metal, then cbm and vbm correspond to above and
    below the fermi level, respectively.

    Args:
        decay_length (float in eV):
            the dos is sampled by an exponential decay function. this parameter
            sets the decay length of the exponential. three times the
            decay_length corresponds to 10% sampling strength. there is a hard
            cutoff at five times the decay length (1% sampling strength)
        sampling_resolution (int):
            number of points to sample dos
        gaussian_smear (float in eV):
            Gaussian smearing (sigma) around each sampled point in dos

    Returns (list of floats):
        cbm_score_i (float): fractional score for i in {s,p,d,f}
        cbm_score_total (float): the total sum of all the {s,p,d,f} scores
            this is useful information when comparing the relative
            contributions from multiples sites
        vbm_score_i (float): fractional score for i in {s,p,d,f}
        vbm_score_total (float): the total sum of all the {s,p,d,f} scores
            this is useful information when comparing the relative
            contributions from multiples sites
    """

    def __init__(self, decay_length=0.1, sampling_resolution=100, gaussian_smear=0.05):
        self.decay_length = decay_length
        self.sampling_resolution = sampling_resolution
        self.gaussian_smear = gaussian_smear

    def featurize(self, dos, idx):
        """
        get dos scores for given site index

        Args:
            dos (pymatgen CompleteDos or their dict):
                dos to featurize, must contain pdos and structure
            idx (int): index of target site in structure.
        """
        if isinstance(dos, dict):
            dos = CompleteDos.from_dict(dos)
        if dos.structure is None:
            raise ValueError("The input dos must contain the structure.")

        orbscores = get_site_dos_scores(dos, idx, self.decay_length, self.sampling_resolution, self.gaussian_smear)

        features = []
        for edge in ["cbm", "vbm"]:
            for score in ["s", "p", "d", "f", "total"]:
                features.append(orbscores[edge][score])
        return features

    def feature_labels(self):
        """
        Returns (list of str): list of names of the features. See the docs for
            the featurizer class for more information.
        """
        labels = []
        for edge in ["cbm", "vbm"]:
            for score in ["s", "p", "d", "f", "total"]:
                labels.append(f"{edge}_{score}")
        return labels

    def citations(self):
        return [
            "@article{dylla2020machine,"
            "title={Machine Learning Chemical Guidelines for Engineering Electronic Structures in Half-Heusler Thermoelectric Materials},"
            "author={Dylla, Maxwell T and Dunn, Alexander and Anand, Shashwat and Jain, Anubhav and Snyder, G Jeffrey and others},"
            "journal={Research}, volume={2020}, pages={6375171}, year={2020}, publisher={AAAS}}"
        ]

    def implementors(self):
        return ["Max Dylla"]


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
        xbm_hybridization (int): the amount of hybridization at the band edge
            characterized by an entropy score (x ln x). the hybridization score
            is larger for a greater number of significant contributors
    """

    def __init__(
        self,
        contributors=1,
        decay_length=0.1,
        sampling_resolution=100,
        gaussian_smear=0.05,
    ):
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
            raise ValueError("The input dos must contain the structure.")

        orbscores = get_cbm_vbm_scores(dos, self.decay_length, self.sampling_resolution, self.gaussian_smear)

        feat = OrderedDict()
        for ex in ["cbm", "vbm"]:
            orbscores.sort(key=lambda x: x[f"{ex}_score"], reverse=True)
            scores = np.array([s[f"{ex}_score"] for s in orbscores])
            feat[f"{ex}_hybridization"] = -np.sum(scores * np.log(scores + 1e-10))  # avoid log(0)

            i = 0
            while i < self.contributors:
                sd = orbscores[i]
                if i < len(orbscores):
                    for p in ["character", "specie"]:
                        feat[f"{ex}_{p}_{i + 1}"] = sd[p]
                    feat[f"{ex}_location_{i + 1}"] = "{};{};{}".format(
                        sd["location"][0], sd["location"][1], sd["location"][2]
                    )
                    feat[f"{ex}_score_{i + 1}"] = float(sd[f"{ex}_score"])
                else:
                    for p in ["character", "specie", "location", "score"]:
                        feat[f"{ex}_{p}_{i + 1}"] = float("NaN")
                i += 1

        return list(feat.values())

    def feature_labels(self):
        """
        Returns ([str]): list of names of the features. See the docs for the
            featurize method for more information.
        """
        labels = []
        for ex in ["cbm", "vbm"]:
            labels.append(f"{ex}_hybridization")
            i = 0
            while i < self.contributors:
                for p in ["character", "specie", "location", "score"]:
                    labels.append(f"{ex}_{p}_{i + 1}")
                i += 1

        return labels

    def citations(self):
        return [
            "@article{dylla2020machine,"
            "title={Machine Learning Chemical Guidelines for Engineering Electronic Structures in Half-Heusler Thermoelectric Materials},"
            "author={Dylla, Maxwell T and Dunn, Alexander and Anand, Shashwat and Jain, Anubhav and Snyder, G Jeffrey and others},"
            "journal={Research}, volume={2020}, pages={6375171}, year={2020}, publisher={AAAS}}"
        ]

    def implementors(self):
        return ["Maxwell Dylla", "Alireza Faghaninia", "Anubhav Jain"]


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
            fermi = dos.get_fermi(concentration=c, temperature=self.T, nstep=50)
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
                    raise ValueError(f'Unsupported "eref": {self.eref}')
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
            labels.append(f"fermi_c{c}T{self.T}")
        if self.return_eref:
            labels.append(f"{self.eref} eref")
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

    def __init__(
        self,
        decay_length=0.1,
        sampling_resolution=100,
        gaussian_smear=0.05,
        species=None,
    ):
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
            raise ValueError("The input dos must contain the structure.")

        orbscores = get_cbm_vbm_scores(dos, decay_length, self.sampling_resolution, self.gaussian_smear)
        feat = OrderedDict()
        for ex in ["cbm", "vbm"]:
            for orbital in ["s", "p", "d", "f"]:
                feat[f"{ex}_{orbital}"] = 0.0
                for specie in self.species:
                    feat[f"{ex}_{specie}_{orbital}"] = 0.0
            for hybrid in ["sp", "sd", "sf", "pd", "pf", "df"]:
                feat[f"{ex}_{hybrid}"] = 0.0

        for contrib in orbscores:
            character = contrib["character"]
            feat[f"cbm_{character}"] += contrib["cbm_score"]
            feat[f"vbm_{character}"] += contrib["vbm_score"]
            for specie in self.species:
                if contrib["specie"] == specie:
                    feat[f"cbm_{specie}_{character}"] += contrib["cbm_score"]
                    feat[f"vbm_{specie}_{character}"] += contrib["vbm_score"]

        for ex in ["cbm", "vbm"]:
            for hybrid in ["sp", "sd", "sf", "pd", "pf", "df"]:
                orb1 = feat[f"{ex}_{hybrid[0]}"]
                orb2 = feat[f"{ex}_{hybrid[1]}"]
                feat[f"{ex}_{hybrid}"] = (orb1 * orb2) * 4.0  # 4x so max=1.0
        return list(feat.values())

    def feature_labels(self):
        """
        Returns ([str]): feature names starting with the extrema (cbm or vbm)
        followed by either s,p,d,f orbital to show normalized contribution
        or a pair showing their hybridization or contribution of an element.
        See the class docs for examples.
        """
        labels = []
        for ex in ["cbm", "vbm"]:
            for orbital in ["s", "p", "d", "f"]:
                labels.append(f"{ex}_{orbital}")
                for specie in self.species:
                    labels.append(f"{ex}_{specie}_{orbital}")
            for hybrid in ["sp", "sd", "sf", "pd", "pf", "df"]:
                labels.append(f"{ex}_{hybrid}")
        return labels

    def citations(self):
        return [
            "@article{dylla2020machine,"
            "title={Machine Learning Chemical Guidelines for Engineering Electronic Structures in Half-Heusler Thermoelectric Materials},"
            "author={Dylla, Maxwell T and Dunn, Alexander and Anand, Shashwat and Jain, Anubhav and Snyder, G Jeffrey and others},"
            "journal={Research}, volume={2020}, pages={6375171}, year={2020}, publisher={AAAS}}"
        ]

    def implementors(self):
        return ["Alireza Faghaninia", "Anubhav Jain", "Maxwell Dylla"]


class DosAsymmetry(BaseFeaturizer):
    """
    Quantifies the asymmetry of the DOS near the Fermi level.

    The DOS asymmetry is defined the natural logarithm of the quotient of the
    total DOS above the Fermi level and the total DOS below the Fermi level. A
    positive number indicates that there are more states directly above the
    Fermi level than below the Fermi level. This featurizer is only meant for
    metals and semi-metals.

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
    """

    def __init__(self, decay_length=0.5, sampling_resolution=100, gaussian_smear=0.05):
        self.decay_length = decay_length
        self.sampling_resolution = sampling_resolution
        self.gaussian_smear = gaussian_smear

    def featurize(self, dos):
        """Calculates the DOS asymmetry.

        Args:
            dos (Dos): A pymatgen Dos object.

        Returns:
            A float describing the asymmetry of the DOS.
        """

        # smears dos for spin up and down
        smear_dos = dos.get_smeared_densities(self.gaussian_smear)
        dos_up = smear_dos[Spin.up]
        dos_down = smear_dos[Spin.down] if Spin.down in smear_dos else smear_dos[Spin.up]
        dos_total = [sum(id) for id in zip(dos_up, dos_down)]

        # determines energy range to sample
        energies = [e for e in dos.energies]
        vbm_space = np.linspace(
            dos.efermi,
            dos.efermi - (5.0 * self.decay_length),
            num=self.sampling_resolution,
        )
        cbm_space = np.linspace(
            dos.efermi,
            dos.efermi + (5.0 * self.decay_length),
            num=self.sampling_resolution,
        )

        # accumulates dos score over energy ranges
        vbm_score = 0
        for e in vbm_space:
            vbm_score += np.interp(e, energies, dos_total) * np.exp(-(dos.efermi - e) * self.decay_length)
        cbm_score = 0
        for e in cbm_space:
            cbm_score += np.interp(e, energies, dos_total) * np.exp(-(e - dos.efermi) * self.decay_length)

        return np.log(cbm_score / vbm_score)

    def feature_labels(self):
        """Returns the labels for each of the features."""
        return ["dos_asymmetry"]

    def citations(self):
        return []

    def implementors(self):
        return ["Maxwell Dylla"]


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
            dos_down = smear_dos[Spin.down] if Spin.down in smear_dos else smear_dos[Spin.up]
            dos_total = [sum(id) for id in zip(dos_up, dos_down)]
            vbm_score = 0
            vbm_space = np.linspace(vbm, vbm - (5.0 * decay_length), num=sampling_resolution)
            for e in vbm_space:
                vbm_score += np.interp(e, energies, dos_total) * np.exp(-(vbm - e) * decay_length)
            cbm_score = 0
            cbm_space = np.linspace(cbm, cbm + (5.0 * decay_length), num=sampling_resolution)
            for e in cbm_space:
                cbm_score += np.interp(e, energies, dos_total) * np.exp(-(e - cbm) * decay_length)

            # add orbital scores to list
            orbital_score = {
                "cbm_score": cbm_score,
                "vbm_score": vbm_score,
                "specie": str(site.specie),
                "character": str(orb),
                "location": list(site.frac_coords),
            }
            orbital_scores.append(orbital_score)

    # normalize by total contribution
    total_cbm = sum(orbital_scores[i]["cbm_score"] for i in range(0, len(orbital_scores)))
    total_vbm = sum(orbital_scores[i]["vbm_score"] for i in range(0, len(orbital_scores)))
    for orbital in orbital_scores:
        orbital["cbm_score"] /= total_cbm
        orbital["vbm_score"] /= total_vbm
    return orbital_scores


def get_site_dos_scores(dos, idx, decay_length, sampling_resolution, gaussian_smear):
    """
    Quantifies the contribution of all atomic orbitals (s/p/d/f) from a
    particular crystal site to the conduction band minimum (CBM) and the
    valence band maximum (VBM). An exponential decay function is used to sample
    the DOS. if the dos is a metal, then CBM and VBM indicate the orbital
    scores above and below the fermi energy, respectively.

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
        idx (int):
            site index for which to gather dos s/p/d/f scores

    Returns:
        orbital_scores (dict):
            a dictionary of the fractional s/p/d/f orbital scores from the
            total dos accumulated from that site. dictionary structure:
                {cbm: {s: (float), ..., f: (float), total: (float)},
                 vbm: {s: (float), ..., f: (float), total: (float)}}
    """
    cbm, vbm = dos.get_cbm_vbm(tol=0.01)
    structure = dos.structure
    site = structure.sites[idx]

    # calculate s/p/d/f dos for cbm and vbm
    orbital_scores = {}
    proj = dos.get_site_spd_dos(site)
    for orb in proj:

        # smear dos for spin up and down
        smear_dos = proj[orb].get_smeared_densities(gaussian_smear)
        dos_up = smear_dos[Spin.up]
        dos_down = smear_dos[Spin.down] if Spin.down in smear_dos else smear_dos[Spin.up]
        dos_total = [sum(id) for id in zip(dos_up, dos_down)]

        # determine energy range to sample
        energies = [e for e in proj[orb].energies]
        vbm_space = np.linspace(vbm, vbm - (5.0 * decay_length), num=sampling_resolution)
        cbm_space = np.linspace(cbm, cbm + (5.0 * decay_length), num=sampling_resolution)

        # accumulate dos score over energy range
        vbm_score = 0
        for e in vbm_space:
            vbm_score += np.interp(e, energies, dos_total) * np.exp(-(vbm - e) * decay_length)
        cbm_score = 0
        for e in cbm_space:
            cbm_score += np.interp(e, energies, dos_total) * np.exp(-(e - cbm) * decay_length)
        orbital_scores[str(orb)] = {"cbm": cbm_score, "vbm": vbm_score}

    # ensure that f-orbitals are represented as zero contribution if none
    if not ("f" in orbital_scores.keys()):
        orbital_scores["f"] = {"cbm": 0.0, "vbm": 0.0}

    # reorder scores so band edge is first followed by orbital
    reordered_scores = {}
    for band in ["cbm", "vbm"]:
        reordered_scores[band] = {}
        for orb in ["s", "p", "d", "f"]:
            reordered_scores[band][orb] = orbital_scores[orb][band]

    # normalize by total cbm/vbm edge contribution from site
    for edge in reordered_scores:
        total_score = sum(reordered_scores[edge].values())
        for orb in reordered_scores[edge].keys():
            reordered_scores[edge][orb] /= total_score
        reordered_scores[edge]["total"] = total_score
    return reordered_scores
