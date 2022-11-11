"""data mining materials properties"""

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("matminer").version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass
