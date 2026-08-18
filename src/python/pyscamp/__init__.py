"""pyscamp: Python bindings for SCAMP (SCAlable Matrix Profile).

The compute lives in the compiled extension ``pyscamp._core``; this package
re-exports its public functions and adds the pure-Python ``join()`` surface.
Prior to 5.0 pyscamp was a single compiled module imported directly; the
public names below are unchanged, so ``import pyscamp; pyscamp.selfjoin(...)``
continues to work.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from ._core import (
    abjoin,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    autotune,
    gpu_supported,
    selfjoin,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
)
from .join import JoinResult, KNNMatches, join

# Single source of truth for the version is the installed distribution
# metadata, which setuptools_scm derives from git tags: a tagged release
# reports that tag (e.g. "5.0.0"), an untagged dev build reports a
# "X.Y.Z.devN" string. Importing from a source tree that was never installed
# (e.g. a raw PYTHONPATH to a build dir) has no metadata -> fall back to "dev".
try:
    __version__ = _dist_version("pyscamp")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    # Unified entry point (new in 5.0).
    "join",
    "JoinResult",
    "KNNMatches",
    # Legacy per-type functions.
    "selfjoin",
    "abjoin",
    "selfjoin_sum",
    "abjoin_sum",
    "selfjoin_knn",
    "abjoin_knn",
    "selfjoin_matrix",
    "abjoin_matrix",
    # Utilities.
    "autotune",
    "gpu_supported",
    "__version__",
]
