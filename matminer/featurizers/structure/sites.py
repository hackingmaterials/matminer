"""
Structure featurizers based on aggregating site features.
"""

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import (
    SOAP,
    AverageBondAngle,
    AverageBondLength,
    CoordinationNumber,
    CrystalNNFingerprint,
    LocalPropertyDifference,
    OPSiteFingerprint,
)
from matminer.featurizers.utils.stats import PropertyStats


class SiteStatsFingerprint(BaseFeaturizer):
    """
    Computes statistics of properties across all sites in a structure.

    This featurizer first uses a site featurizer class (see site.py for
    options) to compute features of each site in a structure, and then computes
    features of the entire structure by measuring statistics of each attribute.
    Can optionally compute the the statistics of only sites with certain ranges
    of oxidation states (e.g., only anions).

    Features:
        - Returns each statistic of each site feature
    """

    def __init__(
        self,
        site_featurizer,
        stats=("mean", "std_dev"),
        min_oxi=None,
        max_oxi=None,
        covariance=False,
    ):
        """
        Args:
            site_featurizer (BaseFeaturizer): a site-based featurizer
            stats ([str]): list of weighted statistics to compute for each feature.
                If stats is None, a list is returned for each features
                that contains the calculated feature for each site in the
                structure.
                *Note for nth mode, stat must be 'n*_mode'; e.g. stat='2nd_mode'
            min_oxi (int): minimum site oxidation state for inclusion (e.g.,
                zero means metals/cations only)
            max_oxi (int): maximum site oxidation state for inclusion
            covariance (bool): Whether to compute the covariance of site features
        """

        self.site_featurizer = site_featurizer
        self.stats = tuple([stats]) if type(stats) == str else stats
        if self.stats and "_mode" in "".join(self.stats):
            nmodes = 0
            for stat in self.stats:
                if "_mode" in stat and int(stat[0]) > nmodes:
                    nmodes = int(stat[0])
            self.nmodes = nmodes

        self.min_oxi = min_oxi
        self.max_oxi = max_oxi
        self.covariance = covariance

    @property
    def _site_labels(self):
        return self.site_featurizer.feature_labels()

    def fit(self, X, y=None, **fit_kwargs):
        """
        Fit the SiteStatsFeaturizer using the fitting function of the underlying
        site featurizer. Only applicable if the site featurizer is fittable.
        See the ".fit()" method of the site_featurizer used to construct the
        class for more information.
        Args:
            X (Iterable):
            y (optional, Iterable):
            **fit_kwargs: Keyword arguments used by the fit function of the
                site featurizer class.
        Returns:
            self (SiteStatsFeaturizer)
        """
        self.site_featurizer.fit(X, y, **fit_kwargs)
        return self

    def featurize(self, s):
        # Get each feature for each site
        vals = [[] for t in self._site_labels]
        for i, site in enumerate(s.sites):
            if (self.min_oxi is None or site.specie.oxi_state >= self.min_oxi) and (
                self.max_oxi is None or site.specie.oxi_state >= self.max_oxi
            ):
                opvalstmp = self.site_featurizer.featurize(s, i)
                for j, opval in enumerate(opvalstmp):
                    if opval is None:
                        vals[j].append(0.0)
                    else:
                        vals[j].append(opval)

        # If the user does not request statistics, return the site features now
        if self.stats is None:
            return vals

        # Compute the requested statistics
        stats = []
        for op in vals:
            for stat in self.stats:
                stats.append(PropertyStats().calc_stat(op, stat))

        # If desired, compute covariances
        if self.covariance:
            if len(s) == 1:
                stats.extend([0] * int(len(vals) * (len(vals) - 1) / 2))
            else:
                covar = np.cov(vals)
                tri_ind = np.triu_indices(len(vals), 1)
                stats.extend(covar[tri_ind].tolist())

        return stats

    def feature_labels(self):
        if self.stats:
            labels = []
            # Make labels associated with the statistics
            for attr in self._site_labels:
                for stat in self.stats:
                    labels.append(f"{stat} {attr}")

            # Make labels associated with the site labels
            if self.covariance:
                sl = self._site_labels
                for i, sa in enumerate(sl):
                    for sb in sl[(i + 1) :]:
                        labels.append(f"covariance {sa}-{sb}")
            return labels
        else:
            return self._site_labels

    def citations(self):
        return self.site_featurizer.citations()

    def implementors(self):
        return ["Nils E. R. Zimmermann", "Alireza Faghaninia", "Anubhav Jain", "Logan Ward", "Alex Dunn"]

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Create a SiteStatsFingerprint class according to a preset

        Args:
            preset (str) - Name of preset
            kwargs - Options for SiteStatsFingerprint
        """

        if preset == "SOAP_formation_energy":
            return SiteStatsFingerprint(SOAP.from_preset("formation_energy"), **kwargs)

        elif preset == "CrystalNNFingerprint_cn":
            return SiteStatsFingerprint(CrystalNNFingerprint.from_preset("cn", cation_anion=False), **kwargs)

        elif preset == "CrystalNNFingerprint_cn_cation_anion":
            return SiteStatsFingerprint(CrystalNNFingerprint.from_preset("cn", cation_anion=True), **kwargs)

        elif preset == "CrystalNNFingerprint_ops":
            return SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops", cation_anion=False), **kwargs)

        elif preset == "CrystalNNFingerprint_ops_cation_anion":
            return SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops", cation_anion=True), **kwargs)

        elif preset == "OPSiteFingerprint":
            return SiteStatsFingerprint(OPSiteFingerprint(), **kwargs)

        elif preset == "LocalPropertyDifference_ward-prb-2017":
            return SiteStatsFingerprint(
                LocalPropertyDifference.from_preset("ward-prb-2017"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"],
            )

        elif preset == "CoordinationNumber_ward-prb-2017":
            return SiteStatsFingerprint(
                CoordinationNumber(nn=VoronoiNN(weight="area"), use_weights="effective"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"],
            )

        elif preset == "Composition-dejong2016_AD":
            return SiteStatsFingerprint(
                LocalPropertyDifference(
                    properties=[
                        "Number",
                        "AtomicWeight",
                        "Column",
                        "Row",
                        "CovalentRadius",
                        "Electronegativity",
                    ],
                    signed=False,
                ),
                stats=["holder_mean::%d" % d for d in range(0, 4 + 1)] + ["std_dev"],
            )

        elif preset == "Composition-dejong2016_SD":
            return SiteStatsFingerprint(
                LocalPropertyDifference(
                    properties=[
                        "Number",
                        "AtomicWeight",
                        "Column",
                        "Row",
                        "CovalentRadius",
                        "Electronegativity",
                    ],
                    signed=True,
                ),
                stats=["holder_mean::%d" % d for d in [1, 2, 4]] + ["std_dev"],
            )

        elif preset == "BondLength-dejong2016":
            return SiteStatsFingerprint(
                AverageBondLength(VoronoiNN()),
                stats=["holder_mean::%d" % d for d in range(-4, 4 + 1)] + ["std_dev", "geom_std_dev"],
            )

        elif preset == "BondAngle-dejong2016":
            return SiteStatsFingerprint(
                AverageBondAngle(VoronoiNN()),
                stats=["holder_mean::%d" % d for d in range(-4, 4 + 1)] + ["std_dev", "geom_std_dev"],
            )

        else:
            # TODO: Why assume coordination number? Should this just raise an error? - lw
            # One of the various Coordination Number presets:
            # MinimumVIRENN, MinimumDistanceNN, JmolNN, VoronoiNN, etc.
            try:
                return SiteStatsFingerprint(CoordinationNumber.from_preset(preset), **kwargs)
            except Exception:
                pass

        raise ValueError("Unrecognized preset!")
