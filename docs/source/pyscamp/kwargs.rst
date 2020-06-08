Keyword Arguments for pyscamp Methods
=====================================
pyscamp methods support several different keyword arguments.

threshold=[float]:
  Distance threshold used for various profile types, correlations found below this threshold will be ignored
pearson=[bool]:
  Output Pearson Correlation rather than Z-normalized euclidean distance
threads=[int]:
  Number of CPU threads to use with SCAMP (if using gpus it is recommended to not use this flag)
gpus=[list of integers]:
  Cuda device ids of gpus to run on, by default we run on all gpus if you have any.
precision=[string]:
  One of [single, mixed, double] default is double precision, other precision types only supported on GPU
mwidth=[int]:
  For matrix summaries, the width of the output matrix (default 50)
mheight=[int]:
  For matrix summaries, the height of the output matrix (default 50)
verbose=[bool]:
  Enable verbose output. This will log to stdout. (default False)

