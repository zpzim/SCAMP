Keyword Arguments for pyscamp Methods
=====================================
pyscamp methods support several different keyword arguments. These apply to
both the unified :func:`~pyscamp.join` entry point and the per-type functions
(``selfjoin``, ``abjoin``, etc.).

:func:`~pyscamp.join` additionally takes the structural arguments ``method``
(which profile to compute), ``index`` (1NN with or without its index),
``left_right`` (both directions as left/right profiles), and ``k`` (number of
neighbors for ``method="knn"``). Those are documented on :func:`~pyscamp.join`
itself rather than here, since they select the computation rather than tune it.

threshold=[float]:
  Distance threshold used for various profile types, correlations found below this threshold will be ignored
pearson=[bool]:
  Output Pearson Correlation rather than Z-normalized euclidean distance
threads=[int]:
  Number of CPU threads to use with SCAMP (if using gpus it is recommended to not use this flag). If you want to prevent GPUs from being used, pass gpus=[] as a kwarg.
gpus=[list of integers]:
  Cuda device ids of gpus to run on, by default we run on all gpus if you have any. To opt out of gpu execution, specify an empty list here.
precision=[string]:
  One of ['single', 'double', 'ultra'], default is double precision. Double and ultra precision are supported on CPU and GPU; single precision is GPU only.
mwidth=[int]:
  For matrix summaries, the width of the output matrix (default 50)
mheight=[int]:
  For matrix summaries, the height of the output matrix (default 50)
max_tile_size=[int]:
  Size of the tiles SCAMP splits the computation into. You normally do not need to set this; it can be used for performance tuning or to reduce peak memory use on very large inputs.
verbose=[bool]:
  Enable verbose output. This will log to stdout. (default False)
allow_trivial_match=[bool]:
  ab-join only. When True (default), all subsequence pairs are considered. When False, treats ``a`` and ``b`` as aligned (e.g. overlapping segments of the same series) and applies the self-join exclusion zone, filtering trivial near-diagonal matches. Equivalent to the ``--aligned`` flag in the CLI. Passing this kwarg to a self-join raises ``ValueError``.

