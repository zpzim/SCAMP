#This file includes tests to check the correcntess of the pyscamp module.
#It does not do extenstive testing of the SCAMP framework that is handled by other integration tests.

import pyscamp as mp
import numpy as np
from distance_matrix_fast import *
from test_common import *
import random 
import sys

failed = False
arr = []
arr2 = []

arr = np.random.random(size=(8000,))
arr2 = np.random.random(size=(8000,))

dm_self = distance_matrix(arr, None, 1024)
dm_ab = distance_matrix(arr, arr2, 1024)

dist, index = mp.selfjoin(arr, 1024, pearson=True)
vdist, vindex = reduce_1nn_index(dm_self)
if compare_vectors(vdist, dist) and compare_index(vindex, vdist, index, dist):
  print("1NN INDEX Self join pass")
else:
  failed = True
  print("1NN INDEX Self join fail")


dist, index = mp.abjoin(arr, arr2, 1024, pearson=True)
vdist, vindex = reduce_1nn_index(dm_ab)


if compare_vectors(vdist, dist) and compare_index(vindex, vdist, index, dist):
  print("1NN INDEX AB join pass")
else:
  failed = True
  print("1NN INDEX AB join fail")

dist = mp.selfjoin_sum(arr, 1024, threshold=0.90, pearson=True)
vdist = reduce_sum_thresh(dm_self, 0.90)

if compare_vectors_sum(vdist, dist, 0.90):
  print("SUM Self join pass")
else:
  failed = True
  print("SUM Self join fail")


dist = mp.abjoin_sum(arr, arr2, 1024, threshold=0.90, pearson=True)
vdist = reduce_sum_thresh(dm_ab, 0.90)

if compare_vectors_sum(vdist, dist, 0.90):
  print("SUM AB join pass")
else:
  failed = True
  print("SUM AB join fail")

dist = mp.selfjoin(arr, 1024, threads=1)
dist = mp.selfjoin(arr, 1024, threads=2)

matrix_summary = reduce_matrix(dm_self, 5, 10, True)
matrix_out = mp.selfjoin_matrix(arr, 1024, threshold=0.05, mwidth=10, mheight=5, pearson=True)
if compare_matrix(matrix_summary, matrix_out, 0.05):
  print ("Matrix Summary self join pass")
else:
  failed = True
  print ("Matrix Summary self join fail")

matrix_summary = reduce_matrix(dm_ab, 5, 10, False)
matrix_out = mp.abjoin_matrix(arr, arr2, 1024, threshold=0.05, mwidth=10, mheight=5, pearson=True)
if compare_matrix(matrix_summary, matrix_out, 0.05):
  print ("Matrix Summary AB join pass")
else:
  failed = True
  print ("Matrix Summary AB join fail")

# Small window size tests (issue #135): with m < 4 the exclusion zone formula
# ceil(m/4) differs from floor(m/4). Specifically m=3 gives ceil=1 vs floor=0;
# floor=0 caused the main diagonal (self-match, Euclidean distance = 0) to win
# for every subsequence, producing an all-zero distance vector.
small_m_arr = np.random.random(size=(200,))
small_m_arr2 = np.random.random(size=(200,))
for test_m in [3, 4]:
  dm_small = distance_matrix(small_m_arr, None, test_m)
  dist_small, index_small = mp.selfjoin(small_m_arr, test_m, pearson=True)
  vdist_small, vindex_small = reduce_1nn_index(dm_small)
  if compare_vectors(vdist_small, dist_small) and compare_index(vindex_small, vdist_small, index_small, dist_small):
    print("1NN INDEX Self join m={} pass".format(test_m))
  else:
    failed = True
    print("1NN INDEX Self join m={} fail".format(test_m))

if mp.gpu_supported():
  print('GPUs Supported')
  thresh = 0.12
  vdist, vindex = reduce_1nn_index(dm_self)
  x = mp.selfjoin_knn(arr, 1024, 5, threshold=thresh, pearson=True)
  if compare_all_neighbors(dm_self, vdist, vindex, x, thresh):
    print("KNN Self join pass")
  else:
    failed = True
    print("KNN Self join fail")
  vdist, vindex = reduce_1nn_index(dm_ab)
  x = mp.abjoin_knn(arr,arr2, 1024, 5, threshold=thresh, pearson=True)
  if compare_all_neighbors(dm_ab, vdist, vindex, x, thresh):
    print("KNN AB join pass")
  else:
    failed = True
    print("KNN AB join fail")

if failed:
  exit(1)


