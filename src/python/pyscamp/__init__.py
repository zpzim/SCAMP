"""pyscamp: Python bindings for SCAMP (SCAlable Matrix Profile).

The compute lives in the compiled extension ``pyscamp._core``; this package
re-exports its public functions and adds the pure-Python ``join()`` surface.
Prior to 5.0 pyscamp was a single compiled module imported directly; the
public names below are unchanged, so ``import pyscamp; pyscamp.selfjoin(...)``
continues to work.
"""

from ._core import (
    __version__,
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
