"""
Structure featurizers based on aggregating site features.
"""
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Element, Specie

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
    Can optionally compute the statistics of only sites with certain ranges
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
        self.stats = tuple([stats]) if isinstance(stats, str) else stats
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

    @classmethod
    def from_preset(cls, preset, **kwargs):
        """
        Create a SiteStatsFingerprint class according to a preset

        Args:
            preset (str) - Name of preset
            kwargs - Options for SiteStatsFingerprint
        """

        if preset == "SOAP_formation_energy":
            return cls(SOAP.from_preset("formation_energy"), **kwargs)

        elif preset == "CrystalNNFingerprint_cn":
            return cls(CrystalNNFingerprint.from_preset("cn", cation_anion=False), **kwargs)

        elif preset == "CrystalNNFingerprint_cn_cation_anion":
            return cls(CrystalNNFingerprint.from_preset("cn", cation_anion=True), **kwargs)

        elif preset == "CrystalNNFingerprint_ops":
            return cls(CrystalNNFingerprint.from_preset("ops", cation_anion=False), **kwargs)

        elif preset == "CrystalNNFingerprint_ops_cation_anion":
            return cls(CrystalNNFingerprint.from_preset("ops", cation_anion=True), **kwargs)

        elif preset == "OPSiteFingerprint":
            return cls(OPSiteFingerprint(), **kwargs)

        elif preset == "LocalPropertyDifference_ward-prb-2017":
            return cls(
                LocalPropertyDifference.from_preset("ward-prb-2017"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"],
            )

        elif preset == "CoordinationNumber_ward-prb-2017":
            return cls(
                CoordinationNumber(nn=VoronoiNN(weight="area"), use_weights="effective"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"],
            )

        elif preset == "Composition-dejong2016_AD":
            return cls(
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
            return cls(
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
            return cls(
                AverageBondLength(VoronoiNN()),
                stats=["holder_mean::%d" % d for d in range(-4, 4 + 1)] + ["std_dev", "geom_std_dev"],
            )

        elif preset == "BondAngle-dejong2016":
            return cls(
                AverageBondAngle(VoronoiNN()),
                stats=["holder_mean::%d" % d for d in range(-4, 4 + 1)] + ["std_dev", "geom_std_dev"],
            )

        else:
            # TODO: Why assume coordination number? Should this just raise an error? - lw
            # One of the various Coordination Number presets:
            # MinimumVIRENN, MinimumDistanceNN, JmolNN, VoronoiNN, etc.
            try:
                return cls(CoordinationNumber.from_preset(preset), **kwargs)
            except Exception:
                pass

        raise ValueError("Unrecognized preset!")


class PartialsSiteStatsFingerprint(SiteStatsFingerprint):
    """
    Computes statistics of properties across all sites in a structure, and
    breaks these down by element. This featurizer first uses a site featurizer
    class (see site.py for options) to compute features of each site of a
    specific element in a structure, and then computes features of the entire
    structure by measuring statistics of each attribute.
    Features:
        - Returns each statistic of each site feature, broken down by element
    """

    def __init__(
        self,
        site_featurizer,
        stats=("mean", "std_dev"),
        min_oxi=None,
        max_oxi=None,
        covariance=False,
        include_elems=(),
        exclude_elems=(),
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

        self.include_elems = list(include_elems)
        self.exclude_elems = list(exclude_elems)
        super().__init__(site_featurizer, stats, min_oxi, max_oxi, covariance)

    def fit(self, X, y=None):
        """Define the list of elements to be included in the PRDF. By default,
        the PRDF will include all of the elements in `X`
        Args:
            X: (numpy array nx1) structures used in the training set. Each entry
                must be Pymatgen Structure objects.
            y: *Not used*
            fit_kwargs: *not used*
        """

        # This method largely copies code from the partial-RDF fingerprint

        # Initialize list with included elements
        elements = [Element(e) for e in self.include_elems]

        # Get all of elements that appear
        for structure in X:
            for element in structure.composition.elements:
                if isinstance(element, Specie):
                    element = element.element  # converts from Specie to Element object
                if element not in elements and element.name not in self.exclude_elems:
                    elements.append(element)

        # Store the elements
        self.elements_ = [e.symbol for e in sorted(elements)]

    def featurize(self, s):
        """
        Get PSSF of the input structure.
        Args:
            s: Pymatgen Structure object.
        Returns:
            pssf: 1D array of each element's ssf
        """

        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")
        if not hasattr(self, "elements_") or self.elements_ is None:
            raise Exception("You must run 'fit' first!")

        output = []
        for e in self.elements_:
            pssf_stats = self.compute_pssf(s, e)
            output.append(pssf_stats)

        return np.hstack(output)

    def compute_pssf(self, s, e):
        # This code is extremely similar to super().featurize(). The key
        # difference is that only one specific element is analyzed.

        # Get each feature for each site
        vals = [[] for t in self._site_labels]
        for i, site in enumerate(s.sites):
            if site.specie.symbol == e:
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
        if not hasattr(self, "elements_") or self.elements_ is None:
            raise Exception("You must run 'fit' first!")

        labels = []
        for e in self.elements_:
            e_labels = [f"{e} {l}" for l in super().feature_labels()]
            for l in e_labels:
                labels.append(l)

        return labels

    def implementors(self):
        return ["Jack Sundberg"]
