"""
pyscamp: Python bindings for SCAMP
----------------------------------

.. currentmodule:: pyscamp

.. autosummary::
   :toctree: _generate

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
    Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidean Distance.

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
    Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidean Distance.

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
