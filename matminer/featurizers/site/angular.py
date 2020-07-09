from __future__ import division

import numpy as np
from functools import lru_cache
from scipy.special import sph_harm

from sympy.physics.wigner import wigner_3j
from pymatgen.analysis.local_env import VoronoiNN

from matminer.utils.caching import get_nearest_neighbors
from matminer.featurizers.base import BaseFeaturizer


class BondOrientationalParameter(BaseFeaturizer):
    """
    Averages of spherical harmonics of local neighbors

    Bond Orientational Parameters (BOPs) describe the local environment around an atom by
    considering the local symmetry of the bonds as computed using spherical harmonics.
    To create descriptors that are invariant to rotating the coordinate system, we use the
    average of all spherical harmonics of a certain degree - following the approach of
    `Steinhardt et al. <https://link.aps.org/doi/10.1103/PhysRevB.28.784>`_.
    We weigh the contributions of each neighbor with the solid angle of the Voronoi tessellation
    (see `Mickel et al. <https://aip.scitation.org/doi/abs/10.1063/1.4774084>_` for further
    discussion). The weighing scheme makes these descriptors vary smoothly with small distortions
    of a crystal structure.

    In addition to the average spherical harmonics, this class can also compute the :math:`W` and
    :math:`\hat{W}` parameters proposed by `Steinhardt et al. <https://link.aps.org/doi/10.1103/PhysRevB.28.784>`_.

    Attributes:
        BOOP Q l=<n> - Average spherical harmonic for a certain degree, n.
        BOOP W l=<n> - W parameter for a certain degree of spherical harmonic, n.
        BOOP What l=<n> - :math:`\hat{W}` parameter for a certain degree of spherical harmonic, n.

    References:
        `Steinhardt et al., _PRB_ (1983) <https://link.aps.org/doi/10.1103/PhysRevB.28.784>`_
        `Seko et al., _PRB_ (2017) <http://link.aps.org/doi/10.1103/PhysRevB.95.144110>`_
    """

    def __init__(self, max_l=10, compute_w=False, compute_w_hat=False):
        """
        Initialize the featurizer

        Args:
            max_l (int) - Maximum spherical harmonic to consider
            compute_w (bool) - Whether to compute Ws as well
            compute_w_hat (bool) - Whether to compute What
        """
        self._nn = VoronoiNN(weight='solid_angle')
        self.max_l = max_l
        self.compute_W = compute_w
        self.compute_What = compute_w_hat

    def featurize(self, strc, idx):
        # Get the nearest neighbors of the atom
        nns = get_nearest_neighbors(self._nn, strc, idx)

        # Get the polar and azimuthal angles of each face
        phi = np.arccos([x['poly_info']['normal'][-1] for x in nns])
        theta = np.arctan2([x['poly_info']['normal'][1] for x in nns],
                           [x['poly_info']['normal'][0] for x in nns])

        # Get the weights for each neighbor
        weights = np.array([x['weight'] for x in nns])
        weights /= weights.sum()

        # Compute the spherical harmonics for the desired `l`s
        Qs = []
        Ws = []
        for l in range(1, self.max_l + 1):
            # Average the spherical harmonic over each neighbor, weighted by solid angle
            qlm = dict((m, np.dot(weights, sph_harm(m, l, theta, phi)))
                       for m in range(-l, l + 1))

            # Compute the average over all m's
            Qs.append(np.sqrt(np.pi * 4 / (2 * l + 1) *
                              np.sum(np.abs(list(qlm.values())) ** 2)))

            # Compute the W, if desired
            if self.compute_W or self.compute_What:
                w = 0
                # Loop over all non-zero Wigner 3j coefficients
                for (m1, m2, m3), wcoeff in get_wigner_coeffs(l):
                    w += qlm[m1] * qlm[m2] * qlm[m3] * wcoeff
                Ws.append(w.real)

        # Compute Whats, if desired
        if self.compute_What:
            Whats = [w / (q / np.sqrt(np.pi * 4 / (2 * l + 1))) ** 3 if abs(q) > 1.0e-6 else 0.0
                     for l, q, w in zip(range(1, self.max_l + 1), Qs, Ws)]

        # Compile the results. Always returns Qs, and optionally the W/What
        if self.compute_W:
            Qs += Ws
        if self.compute_What:
            Qs += Whats
        return Qs

    def feature_labels(self):
        q_labels = ['BOOP Q l={}'.format(l) for l in range(1, self.max_l+1)]
        if self.compute_W:
            q_labels += ['BOOP W l={}'.format(l) for l in range(1, self.max_l+1)]
        if self.compute_What:
            q_labels += ['BOOP What l={}'.format(l) for l in range(1, self.max_l + 1)]
        return q_labels

    def citations(self):
        return ["@article{Seko2017,"
                "author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, "
                "Keita and Takahashi, Akira and Tanaka, Isao},"
                "doi = {10.1103/PhysRevB.95.144110},"
                "journal = {Physical Review B}, number = {14}, pages = {144110},"
                "title = {{Representation of compounds for machine-learning prediction of physical properties}},"
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.95.144110},"
                "volume = {95},year = {2017}}",
                "@article{Steinhardt1983,"
                "author = {Steinhardt, Paul J. and Nelson, David R. and Ronchetti, Marco},"
                "doi = {10.1103/PhysRevB.28.784}, journal = {Physical Review B},"
                "month = {jul}, number = {2}, pages = {784--805},"
                "title = {{Bond-orientational order in liquids and glasses}},"
                "url = {https://link.aps.org/doi/10.1103/PhysRevB.28.784}, "
                "volume = {28}, year = {1983}}"]

    def implementors(self):
        return ['Logan Ward', 'Aidan Thompson']


@lru_cache(maxsize=32)
def get_wigner_coeffs(l):
    """Get the list of non-zero Wigner 3j triplets

    Args:
        l (int): Desired l
    Returns:
        List of tuples that contain:
            - ((int)) m coordinates of the triplet
            - (float) Wigner coefficient
    """

    return [((m1, m2, m3), float(wigner_3j(l, l, l, m1, m2, m3)))
            for m1, m2, m3 in _iterate_wigner_3j(l)]


def _iterate_wigner_3j(l):
    """Iterator over all non-zero Wigner 3j triplets

    Args:
        l (int) - Desired l
    Generates:
        pairs of acceptable l's
    """

    for m1 in range(-l, l+1):
        for m2 in range(-l, l+1):
            m3 = -1 * (m1 + m2)
            if -l <= m3 <= l:
                yield m1, m2, m3