"""data mining materials properties"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matminer")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
