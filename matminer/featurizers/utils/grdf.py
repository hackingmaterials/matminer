"""Functions designed to work with General Radial Distribution Function"""

from scipy.special import erf
from scipy import integrate
from math import pi

import numpy as np


def initialize_pairwise_function(name, **options):
    """Create a new pairwise function object

    Args:
        name (string): Name of class to instantiate
    Keyword Arguments:
        Any options for the pairwise class (see each pairwise function for details)
    """

    # Get the desired class
    try:
        cls = globals()[name]
    except:
        raise ValueError('No such class: {}'.format(name))

    # Instantiate it
    output = cls(**options)

    # Check types
    if not isinstance(output, AbstractPairwise):
        raise ValueError('Not a pairwise measure: {}'.format(name))
    return output


class AbstractPairwise(object):
    """Abstract class for pairwise functions used in Generalized Radial Distribution Function"""

    def name(self):
        """Make a label for this pairwise function

        Returns:
            (string) Label for the function
        """
        params = sorted(self.__dict__.items(), key=lambda x: x[0])
        return '{} {}'.format(self.__class__.__name__,
                              ' '.join('{}={}'.format(k, v) for k, v in params))

    def __call__(self, r_ij):
        """Compute the pairwise sum for a series of radii

        Args:
            r_ij ([float]) - Pairwise distances
        Returns:
             (float) - Pairwise functions for each
        """

        raise NotImplementedError()

    def volume(self, cutoff):
        """Compute the volume of this pairwise function

        Args:
            cutoff (float): Cutoff distance for radial distribution function
        Returns:
            (float): Volume of bin
        """

        results = integrate.quad(lambda x: 4. * pi * self(x) * x ** 2., 0, cutoff)
        if results[1] > 1e-5:
            raise ValueError('Numerical integration fails for this function.'
                             ' Please implement analytic integral')
        return results[0]


class Histogram(AbstractPairwise):
    """Rectangular window function, used in conventional Radial Distribution Functions"""

    def __init__(self, start, width):
        """Initialize the window function

        Args:
            start (float): Beginning of window
            width (float): Size of window
        """
        self.start = start
        self.width = width

    def __call__(self, r_ij):
        return np.logical_and(np.greater_equal(r_ij, self.start),
                              np.less(r_ij, self.start + self.width), dtype=np.float)

    def volume(self, cutoff):
        return 4. / 3 * np.pi * (min(self.start + self.width, cutoff) ** 3 - self.start ** 3)


class Gaussian(AbstractPairwise):
    """Gaussian function, with specified width and center"""

    def __init__(self, width, center):
        """Initialize the gaussian function

        Args:
            width (float): Width of the gaussian
            center (float): Center of the gaussian
        """
        self.width = width
        self.center = center

    def __call__(self, r_ij):
        return np.exp(-1 * np.power(np.subtract(r_ij, self.center) / self.width, 2))

    def volume(self, cutoff):
        return pi * self.width * (
            np.sqrt(pi) * (2 * self.center ** 2 + self.width ** 2) * (
                erf((cutoff - self.center) / self.width) + erf(self.center / self.width)
            ) + 2 * self.width * (self.center * self(0) - (self.center + cutoff) * self(cutoff))
        )
