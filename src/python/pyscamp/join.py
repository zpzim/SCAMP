"""Unified pyscamp.join() API.

A single entry point covering self/ab joins x {1nn, sum, knn, matrix},
with optional nearest-neighbor index (index=) and optional both-direction
left/right profiles (left_right=). Results come back as a JoinResult whose
populated fields depend on the request. The heavy lifting is delegated to
the compiled core (pyscamp._core._do_join).

Left / right profiles
---------------------
left_right=True returns the join in both directions as `left_*` and `right_*`.
For a *self-join* these are the standard left/right matrix profiles:

  * left  = column direction: for position j, the nearest neighbor among
            positions i < j -- the best *preceding* match.
  * right = row direction: for position i, the nearest neighbor among
            positions j > i -- the best *subsequent* match.

The default (left_right=False) `profile` for a self-join is the combined full
matrix profile, i.e. the elementwise min(left, right). For an ab-join, left is
"each subsequence in A, its match in B" and right is the reverse (B in A);
there is no preceding/subsequent ordering between two different series, so the
left/right labels there are purely the two join directions.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

from . import _core as _mp

__all__ = ["join", "JoinResult", "KNNMatches"]

_METHODS = ("1nn", "sum", "knn", "matrix")

# left_right is only meaningful where there is a distinct per-row profile to
# return. The matrix summary already reduces over both axes into one grid.
_LEFT_RIGHT_METHODS = frozenset({"1nn", "sum", "knn"})

# Default output-grid size for method="matrix" when mheight/mwidth are unset.
_DEFAULT_MATRIX_DIM = 50


class KNNMatches(NamedTuple):
    """KNN result as parallel numpy arrays (one entry per emitted match).

    cols      : column index of the match (subsequence in A).
    rows      : row index of the match (the neighbor).
    distances : correlation or z-normalized Euclidean distance.
    """

    cols: np.ndarray
    rows: np.ndarray
    distances: np.ndarray


@dataclass
class JoinResult:
    """Structured result of a join. Only the fields relevant to the call are
    populated; everything else stays None. Inspect via repr().

    Single-direction fields (left_right=False)
        profile : 1D distances for 1nn/sum.
        index   : nearest-neighbor indices (1nn with index=True only).
        matrix  : 2D pooled distance-matrix summary (method="matrix").
        matches : KNNMatches namedtuple (method="knn").

    Left / right fields (left_right=True)
        left_profile / left_index    : column direction (self-join: preceding).
        right_profile / right_index   : row direction (self-join: subsequent).
        left_matches / right_matches : KNNMatches for each direction (knn).
    """

    method: str
    profile: Optional[np.ndarray] = None
    index: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    matches: Optional[KNNMatches] = None
    left_profile: Optional[np.ndarray] = None
    left_index: Optional[np.ndarray] = None
    right_profile: Optional[np.ndarray] = None
    right_index: Optional[np.ndarray] = None
    left_matches: Optional[KNNMatches] = None
    right_matches: Optional[KNNMatches] = None

    def __repr__(self) -> str:
        present = [
            name
            for name in (
                "profile",
                "index",
                "matrix",
                "matches",
                "left_profile",
                "left_index",
                "right_profile",
                "right_index",
                "left_matches",
                "right_matches",
            )
            if getattr(self, name) is not None
        ]
        return f"JoinResult(method={self.method!r}, populated={present})"


def _knn_from(d: dict, prefix: str) -> Optional[KNNMatches]:
    key = prefix + "match_cols"
    if key not in d:
        return None
    return KNNMatches(
        cols=d[prefix + "match_cols"],
        rows=d[prefix + "match_rows"],
        distances=d[prefix + "match_dist"],
    )


def _result_from_dict(method: str, left_right: bool, d: dict) -> JoinResult:
    """Assemble a JoinResult from the flat dict _do_join returns."""
    res = JoinResult(method=method)
    if method == "knn":
        if left_right:
            res.left_matches = _knn_from(d, "left_")
            res.right_matches = _knn_from(d, "right_")
        else:
            res.matches = _knn_from(d, "")
        return res
    if method == "matrix":
        res.matrix = d.get("matrix")
        return res
    # 1nn / sum
    if left_right:
        res.left_profile = d.get("left_profile")
        res.left_index = d.get("left_index")
        res.right_profile = d.get("right_profile")
        res.right_index = d.get("right_index")
    else:
        res.profile = d.get("profile")
        res.index = d.get("index")
    return res


def join(
    a,
    b=None,
    m: Optional[int] = None,
    *,
    method: str = "1nn",
    index: bool = True,
    left_right: bool = False,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    mheight: Optional[int] = None,
    mwidth: Optional[int] = None,
    precision: str = "double",
    pearson: bool = False,
    threads: Optional[int] = None,
    gpus: Optional[list] = None,
    max_tile_size: Optional[int] = None,
    allow_trivial_match: bool = True,
) -> JoinResult:
    """Compute a matrix-profile join through a single entry point.

    Parameters
    ----------
    a : 1D array
        Time series. Columns of the distance matrix.
    b : 1D array, optional
        Second time series. Omitted -> self-join of `a`; otherwise ab-join
        (rows of the distance matrix come from `b`).
    m : int
        Subsequence (window) length. Required.
    method : {"1nn", "sum", "knn", "matrix"}
        Which profile to compute.
    index : bool, default True
        (method="1nn") Also return the nearest-neighbor index. index=False
        selects the cheaper 1NN kernel that skips index tracking (GitHub #95).
    left_right : bool, default False
        Return both join directions as left_*/right_* instead of the single
        combined profile (GitHub #34/#130). See the module docstring for the
        preceding/subsequent semantics. Not valid for method="matrix".
    k : int
        (method="knn") Number of neighbors per subsequence. Required for knn.
    threshold : float, optional
        (sum / knn / matrix) Correlation threshold in [-1, 1].
    mheight, mwidth : int, optional
        (method="matrix") Output grid dimensions. Default 50 x 50.
    precision : {"double", "single", "ultra"}, default "double"
    pearson : bool, default False
        Return Pearson correlations instead of z-normalized Euclidean distance.
    threads : int, optional
        CPU worker threads. gpus : list[int], optional (empty list forces CPU).
        max_tile_size : int, optional.
    allow_trivial_match : bool, default True
        (ab-join only) When False, treats a and b as aligned and excludes
        trivial self-matches near the equivalent main diagonal.

    Returns
    -------
    JoinResult
        Only the fields relevant to the call are populated.
    """
    if m is None:
        raise TypeError("join() requires the subsequence length m, e.g. join(a, m=100)")
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(_METHODS)}")

    self_join = b is None

    # Reject method-inappropriate options with an actionable message.
    if method != "knn" and k is not None:
        raise ValueError(f"k is only valid for method='knn', not {method!r}")
    if method == "knn" and k is None:
        raise ValueError("method='knn' requires k=<number of neighbors>")
    if method != "matrix" and (mheight is not None or mwidth is not None):
        raise ValueError(
            f"mheight/mwidth are only valid for method='matrix', not {method!r}"
        )
    if method != "1nn" and index is not True:
        raise ValueError(f"index= is only valid for method='1nn', not {method!r}")
    if threshold is not None and method == "1nn":
        raise ValueError("threshold is not used by method='1nn'")
    if left_right and method not in _LEFT_RIGHT_METHODS:
        raise ValueError(
            f"left_right is not supported for method={method!r} "
            f"(the matrix summary already reduces over both axes)"
        )
    if self_join and not allow_trivial_match:
        raise ValueError(
            "allow_trivial_match is an ab-join option; self-joins always "
            "exclude trivial matches"
        )

    # Assemble the kwargs the compiled layer understands (only what applies).
    kw = {"pearson": pearson, "precision": precision}
    if threads is not None:
        kw["threads"] = threads
    if gpus is not None:
        kw["gpus"] = gpus
    if max_tile_size is not None:
        kw["max_tile_size"] = max_tile_size
    if not self_join:
        kw["allow_trivial_match"] = allow_trivial_match
    if method in ("sum", "knn", "matrix") and threshold is not None:
        kw["threshold"] = threshold
    if method == "matrix":
        kw["mheight"] = _DEFAULT_MATRIX_DIM if mheight is None else mheight
        kw["mwidth"] = _DEFAULT_MATRIX_DIM if mwidth is None else mwidth

    d = _mp._do_join(
        a,
        b,
        m,
        method,
        bool(index),
        bool(left_right),
        int(k) if k is not None else 0,
        **kw,
    )
    return _result_from_dict(method, left_right, d)
