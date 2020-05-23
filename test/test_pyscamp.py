#This file includes tests to check the correcntess of the pyscamp module.
#It does not do extenstive testing of the SCAMP framework that is handled by other integration tests.

import pyscamp as mp
import numpy as np
from distance_matrix_fast import *
import random 
import sys

def compare_index(valid, check):
  if np.any(valid.shape != check.shape):
    print('Output Shapes do not match')
    return False
  
  index_match_ratio = 0.001
  incorrect = np.count_nonzero(valid != check)
  ratio = incorrect / len(valid) 
  is_valid = ratio < index_match_ratio
  #if not is_valid:
  #  print(valid)
  #  print(check)
  #  print('\n')
  return is_valid

def compare_vectors(valid, check, eps):
  if np.any(valid.shape != check.shape):
    print('Output Shapes do not match')
    return False
    
  diff = np.abs(valid - check)
  is_valid = np.max(diff) < eps
  
  if not is_valid:
    print(diff[diff > eps])
    print(valid[diff > eps])
    print(check[diff > eps])
    print('\n')
  return is_valid

failed = False
arr = []
arr2 = []
num = 0
num2 = 0
vector_match_epsilon = 0.001


arr = np.random.random(size=(8000,))
arr2 = np.random.random(size=(8000,))

dm_self = distance_matrix(arr, None, 1024)
dm_ab = distance_matrix(arr, arr2, 1024)

dist, index = mp.selfjoin(arr, 1024, pearson=True)
vdist, vindex = reduce_1nn_index_unshifted(dm_self)

if compare_vectors(vdist, np.array(dist), vector_match_epsilon) and compare_index(vindex, np.array(index)):
  print("1NN INDEX Self join pass")
else:
  failed = True
  print("1NN INDEX Self join fail")


dist, index = mp.abjoin(arr, arr2, 1024, pearson=True)
vdist, vindex = reduce_1nn_index_unshifted(dm_ab)


if compare_vectors(vdist, dist, vector_match_epsilon) and compare_index(vindex, index):
  print("1NN INDEX AB join pass")
else:
  failed = True
  print("1NN INDEX AB join fail")

dist = mp.selfjoin_sum(arr, 1024, threshold=0.90, pearson=True)
dist = dist.reshape((len(dist), 1))
vdist = reduce_sum_thresh(dm_self, 0.90)

if compare_vectors(vdist, dist, vector_match_epsilon):
  print("SUM Self join pass")
else:
  failed = True
  print("SUM Self join fail")


dist = mp.abjoin_sum(arr, arr2, 1024, threshold=0.90, pearson=True)
dist = dist.reshape((len(dist), 1))
vdist = reduce_sum_thresh(dm_ab, 0.90)

if compare_vectors(vdist, dist, vector_match_epsilon):
  print("SUM AB join pass")
else:
  failed = True
  print("SUM AB join fail")

dist = mp.selfjoin(arr, 1024, threads=1)
dist = mp.selfjoin(arr, 1024, threads=2)


if mp.gpu_supported():
  print('GPUs Supported')
  # TODO(zpzim): add a correctness check here once we have a test for that
  x = mp.selfjoin_knn(arr, 1024, 5, threshold=0.95, pearson=True)
  # TODO(zpzim): add a correctness check here once we have a test for that
  matrix = mp.abjoin_matrix(arr, arr2, 1024, threshold=0.125, mwidth=10, mheight=5, pearson=True)

if failed:
  exit(1)


