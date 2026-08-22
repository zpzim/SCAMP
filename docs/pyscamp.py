"""
pyscamp: Python bindings for SCAMP
----------------------------------

.. currentmodule:: pyscamp

.. autosummary::
   :toctree: _generate

   join
   JoinResult
   KNNMatches
   selfjoin
   abjoin
   selfjoin_sum
   abjoin_sum
   selfjoin_knn
   abjoin_knn
   selfjoin_matrix
   abjoin_matrix
   autotune
   gpu_supported
"""


def gpu_supported():
    """Returns true if both 1) The module was compiled with GPU support and 2) GPUs are available."""
    ...


def autotune(devices=None, cache_path=""):
    """
    Run the SCAMP GPU kernel autotuner for the selected device(s) and persist
    the chosen kernel configurations to disk. Future pyscamp calls on the same
    machine will read these configurations from the cache and use them when
    launching GPU kernels.

    A full sweep takes a few minutes on a recent GPU. The output is verbose
    so you can follow progress; pass ``cache_path`` to redirect the write
    elsewhere (e.g. for a sandboxed run).

    :param devices: List of CUDA device IDs to tune. If empty (default),
                    only device 0 is tuned -- a full sweep takes O(minutes)
                    and most multi-GPU boxes hold identical devices, so
                    sweeping them all wastes wall time on identical configs.
                    Pass ``devices=[0, 1, ...]`` explicitly to tune more
                    than one (e.g. if you have two different GPU models).
    :type devices: list[int], optional
    :param cache_path: Filesystem path to read/write the cache from. Empty
                       (default) resolves in this order: ``$SCAMP_AUTOTUNE_CACHE``
                       (if set, used verbatim), then ``$XDG_CACHE_HOME/scamp/autotune.txt``
                       (if set), then a platform-specific user dir
                       (``$HOME/.cache/scamp/autotune.txt`` on Linux/macOS;
                       ``%LOCALAPPDATA%\\scamp\\autotune.txt`` on Windows).
                       Parent directories are created automatically.
    :type cache_path: str, optional
    :return: Number of devices that were tuned.
    :rtype: int
    :raises RuntimeError: If pyscamp was built without CUDA support.
    :raises ValueError: If no CUDA devices are available.
    """
    ...


def selfjoin(a, m, **kwargs):
    """
    Computes the matrix profile for time series A.

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :return: A tuple containing the matrix profile as the first element and the indices as the second element.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    """
    ...


def abjoin(a, b, m, **kwargs):
    """
    For each subsequence in time series A, finds the nearest neighbor in time series B.

    :param a: Time series, b will be queried for subsequences in a.
    :type a: 1D array
    :param b: Time series in which to search for matches for subsequences in a.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :return: A tuple. First element: The nearest neighbor distance of subsequences in a to time series b. Second element: The index (in b) of each nearest neighbor.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    """
    ...


def selfjoin_sum(a, m, **kwargs):
    """
    Returns the sum of the correlations above specified threshold (default 0) for each subsequence in a time series.

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: For each subsequence in A, returns the sum of correlations above the specified threshold to other subsequences in A.
    :rtype: np.ndarray[float64]
    """
    ...


def abjoin_sum(a, b, m, **kwargs):
    """
    For each subsequence in time series a, returns the sum of the correlations to subsequences in time series b above specified threshold (default 0).

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param b: Time series to search for matches.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: For each subsequence in A, returns the sum of correlations above the specified threshold in B.
    :rtype: np.ndarray[float64]
    """
    ...


def selfjoin_knn(a, m, k, **kwargs):
    """
    [GPU ONLY, EXPERIMENTAL] Returns the approximate k nearest neighbors for each subsequence in a time series.

    :param a: Time series to compute the KNN matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param k: Number of neighbors to return for each subsequence.
    :type k: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: List of tuples (col, row, distance) containing the matches (up to K) for each column of the distance matrix, row is the index of the match, and d is the distance between the two subsequences.
    :rtype: List of tuple[int, int, float]
    """
    ...


def abjoin_knn(a, b, m, k, **kwargs):
    """
    [GPU ONLY, EXPERIMENTAL] For each subsequence in time series A, returns its approximate K nearest neighbors in time series B.

    :param a: Time series to compute the KNN matrix profile for.
    :type a: 1D array
    :param b: Time series in which to search for matches.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param k: Number of neighbors to return for each subsequence.
    :type k: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: List of tuples (col, row, distance) containing the matches (up to K) for each column of the distance matrix, col is the index in A, row is the index in B of the match, and d is the distance between the two subsequences.
    :rtype: List of tuple[int, int, float]
    """
    ...


def selfjoin_matrix(a, m, **kwargs):
    """
    [EXPERIMENTAL] Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidean Distance.

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param mheight: Height of the pooled distance matrix to output. Default 50.
    :type mheight: int, optional
    :param mwidth: Width of the pooled distance matrix to output. Default 50.
    :type mwidth: int, optional
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: A 2D array of height mheight and width mwidth. This is a pooled version of the full distance matrix.
    :rtype: 2D array
    """
    ...


def abjoin_matrix(a, b, m, **kwargs):
    """
    [EXPERIMENTAL] Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidean Distance.

    :param a: Time series corresponding to the columns of the distance matrix.
    :type a: 1D array
    :param b: Time series corresponding to the rows of the distance matrix.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param mheight: Height of the pooled distance matrix to output. Default 50.
    :type mheight: int, optional
    :param mwidth: Width of the pooled distance matrix to output. Default 50.
    :type mwidth: int, optional
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: A 2D array of height mheight and width mwidth. This is a pooled version of the full distance matrix.
    :rtype: 2D array
    """
    ...


def join(a, b=None, m=None, *, method="1nn", index=True, left_right=False,
         k=None, threshold=None, mheight=None, mwidth=None,
         precision="double", pearson=False, threads=None, gpus=None,
         max_tile_size=None, allow_trivial_match=True):
    """
    Unified entry point for every SCAMP join. Computes a self-join of ``a``
    (when ``b`` is omitted) or an ab-join of ``a`` against ``b``, in one of
    several forms selected by ``method``, and returns a :class:`JoinResult`
    whose populated fields depend on the arguments.

    This is the recommended interface as of pyscamp 5.0. The per-type
    functions (:func:`selfjoin`, :func:`abjoin`, :func:`selfjoin_sum`, etc.)
    remain available and are equivalent to the corresponding ``join`` call.

    :param a: Time series. Columns of the distance matrix.
    :type a: 1D array
    :param b: Second time series. If omitted, a self-join of ``a`` is computed;
              otherwise an ab-join whose rows come from ``b``.
    :type b: 1D array, optional
    :param m: Subsequence (window) length. Required.
    :type m: int
    :param method: Which profile to compute: ``"1nn"`` (nearest-neighbor
                   matrix profile), ``"sum"`` (sum of correlations above a
                   threshold), ``"knn"`` (approximate k-nearest-neighbors), or
                   ``"matrix"`` (pooled distance-matrix summary). Default
                   ``"1nn"``.
    :type method: str, optional
    :param index: (``method="1nn"`` only) When True (default) the result also
                  carries the nearest-neighbor index. Pass ``index=False`` to
                  compute only the distance profile, which uses a cheaper
                  kernel that does not track indices.
    :type index: bool, optional
    :param left_right: Return both join directions as ``left_*`` / ``right_*``
                       instead of a single combined profile. For a self-join
                       these are the left (nearest *preceding* neighbor) and
                       right (nearest *subsequent* neighbor) matrix profiles;
                       for an ab-join they are the A-in-B and B-in-A
                       directions. Not valid for ``method="matrix"``.
    :type left_right: bool, optional
    :param k: (``method="knn"`` only, required) Number of neighbors to return
              per subsequence.
    :type k: int, optional
    :param threshold: (sum / knn / matrix) Correlation threshold in [-1, 1].
    :type threshold: float, optional
    :param mheight: (``method="matrix"`` only) Output grid height. Default 50.
    :type mheight: int, optional
    :param mwidth: (``method="matrix"`` only) Output grid width. Default 50.
    :type mwidth: int, optional
    :param precision: ``"double"`` (default), ``"single"``, or ``"ultra"``.
    :type precision: str, optional
    :param pearson: Return Pearson correlations instead of z-normalized
                    Euclidean distance. Default False.
    :type pearson: bool, optional
    :param threads: Number of CPU worker threads.
    :type threads: int, optional
    :param gpus: CUDA device ids to use; an empty list forces CPU execution.
    :type gpus: list[int], optional
    :param max_tile_size: Tile size override for performance tuning.
    :type max_tile_size: int, optional
    :param allow_trivial_match: (ab-join only) When False, treats ``a`` and
                                ``b`` as aligned and excludes trivial
                                self-matches near the equivalent main diagonal.
    :type allow_trivial_match: bool, optional
    :return: A result object whose populated fields depend on the request.
    :rtype: JoinResult
    """
    ...


class JoinResult:
    """
    Structured result returned by :func:`join`. Only the fields relevant to
    the request are populated; the rest are ``None``.

    :ivar method: The join method that was run.
    :ivar profile: Primary distance profile (``method`` ``"1nn"`` / ``"sum"``,
                   ``left_right=False``). 1D array.
    :ivar index: Nearest-neighbor indices (``method="1nn"``, ``index=True``,
                 ``left_right=False``). 1D array.
    :ivar matrix: Pooled 2D distance-matrix summary (``method="matrix"``).
    :ivar matches: :class:`KNNMatches` for ``method="knn"``,
                   ``left_right=False``.
    :ivar left_profile: Column-direction profile (``left_right=True``); for a
                        self-join, the nearest *preceding* neighbor.
    :ivar left_index: Indices for ``left_profile`` (1nn).
    :ivar right_profile: Row-direction profile (``left_right=True``); for a
                         self-join, the nearest *subsequent* neighbor.
    :ivar right_index: Indices for ``right_profile`` (1nn).
    :ivar left_matches: :class:`KNNMatches` for the left direction
                        (``method="knn"``, ``left_right=True``).
    :ivar right_matches: :class:`KNNMatches` for the right direction.
    """
    ...


class KNNMatches:
    """
    KNN matches as three parallel numpy arrays (one entry per emitted match).
    A ``namedtuple``, so it also unpacks as ``cols, rows, distances``.

    :ivar cols: Column index of each match (subsequence in A).
    :ivar rows: Row index of each match (the neighbor).
    :ivar distances: Correlation or z-normalized Euclidean distance per match.
    """
    ...
