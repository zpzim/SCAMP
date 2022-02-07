import sys, os
from distance_matrix_fast import *
from test_common import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import subprocess


extra_opts = ''
outfile = '/dev/null'
static_test_cases = ['SampleInput/randomwalk1K_nan.txt', 'SampleInput/poorly_conditioned_test.txt']
window_sizes_to_test = [5, 7, 13, 25, 33, 78, 92, 100, 189, 339, 500]
tile_sizes_to_test = [1024, 2048, 4096]
input_sizes_to_test = [20, 100, 250, 500, 1500, 8000]
matrix_sizes_to_test = [3, 10, 101]
thresholds_to_test = [0.0,0.125,0.5]

np.set_printoptions(edgeitems=30, linewidth=100000)
np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument('--executable', help='SCAMP executable to test, input \'pyscamp\' to use pyscamp', required=True)
parser.add_argument('--force_gpu', type=bool, help='Forces GPU-specific tests to run')
parser.add_argument('--output_file', help='File for test std output')
parser.add_argument('--extra_args', help='Extra arguments to be passed to each test invocation')
parser.add_argument('--window_sizes', type=int, nargs='*')
parser.add_argument('--tile_sizes', type=int, nargs='*')
parser.add_argument('--input_sizes', type=int, nargs='*')
parser.add_argument('--matrix_sizes', type=int, nargs='*')
parser.add_argument('--thresholds', type=float, nargs='*')
args = parser.parse_args()


executable = args.executable

if args.output_file is not None:
  outfile = args.output_file

if args.extra_args is not None:
  extra_opts = args.extra_args

if args.window_sizes is not None:
  window_sizes_to_test = args.window_sizes

if args.tile_sizes is not None:
  tile_sizes_to_test = args.tile_sizes

if args.input_sizes is not None:
  input_sizes_to_test = args.input_sizes

if args.matrix_sizes is not None:
  matrix_sizes_to_test = args.matrix_sizes

if args.thresholds is not None:
  thresholds_to_test = args.thresholds

if executable == 'pyscamp':
  import pyscamp as mp
  if args.tile_sizes:
    print('Warning: ignoring tile_sizes during pyscamp execution')
  tile_sizes_to_test = [0]

gpu_enabled = False
if args.force_gpu or (executable == 'pyscamp' and mp.gpu_supported()):
  gpu_enabled = True

index_match_ratio = 0.001
matrix_match_ratio = 0.001

vector_match_epsilon_SUM = 0.001
vector_match_epsilon_1NN = 0.001

matrix_match_epsilon = 0.001

def generate_input_arrays(input_sizes):
  arrs = []
  for s in input_sizes:
    arrs.append(np.random.randn(s))
  return arrs

def read_file_inputs(files):
  arrs = []
  for fpath in files:
    x = np.array(pd.read_csv(fpath, sep=' ', header=None))
    arrs.append(x.reshape((len(x),)))
  return arrs

def generate_tests(input_sizes, windows):
  tests = set()
  for a in range(len(input_sizes)):
    for b in range(len(input_sizes)):
        exclusion = 0
        if a == b: 
          exclusion = input_sizes[a] // 4
        for window in windows:
          if window <= input_sizes[a] - exclusion and window <= input_sizes[b] - exclusion:
            tests.add((a, b, window))
  return tests


def evaluate_result(dm_reductions, scamp_results, subtestargs):
  ptype = subtestargs['ptype']
  if ptype == "1NN_INDEX":
    valid_nn, valid_idx = dm_reductions[("1NN_INDEX",None,None,None)]
    return compare_index(valid_idx, valid_nn, scamp_results[1], scamp_results[0]) and compare_vectors(valid_nn, scamp_results[0])
  
  if ptype == "1NN":
    valid_data = dm_reductions[("1NN",None,None,None)]
    return compare_vectors(valid_data, scamp_results[0])
    
  if ptype == "SUM_THRESH":
    valid_data = dm_reductions[("SUM_THRESH",subtestargs['threshold'],None,None)]
    return compare_vectors_sum(valid_data, scamp_results[0], subtestargs['threshold'])
  
  if ptype == "MATRIX_SUMMARY":
    valid_data = dm_reductions[("MATRIX_SUMMARY", None, subtestargs['rrow'], subtestargs['rcol'])] 
    return compare_matrix(valid_data, scamp_results[0], 0.0)
  
  if ptype == "ALL_NEIGHBORS":
    valid_dm = dm_reductions[("ALL_NEIGHBORS", None, None, None)]
    valid_nn, valid_idx = dm_reductions[("1NN_INDEX",None,None,None)]
    return compare_all_neighbors(valid_dm, valid_nn, valid_idx, scamp_results[0], subtestargs['threshold'])
  
  return None

def run_pyscamp(inputs, a, b, window, max_matches, thresh, ptype, rrows, rcols):  
  args = {}
  args['pearson'] = True
  if thresh:
    args['threshold'] = thresh
  if rrows:
    args['mheight'] = rrows
  if rcols:
    args['mwidth'] = rcols
  if '--no_gpu' in extra_opts:
    args['gpus'] = []

  if not max_matches:
    max_matches = 5
  
  mp_columns_out = None
  mp_columns_out_index = None
  mp_rows_out = None
  mp_rows_out_index = None
    
  if ptype == "1NN_INDEX":
    if a == b:
      mp_columns_out, mp_columns_out_index = mp.selfjoin(inputs[a], window, **args)
    else:
      mp_columns_out, mp_columns_out_index = mp.abjoin(inputs[a], inputs[b], window, **args)
  elif ptype == "SUM_THRESH":
    if a == b:
      mp_columns_out = mp.selfjoin_sum(inputs[a], window, **args)
    else:
      mp_columns_out = mp.abjoin_sum(inputs[a], inputs[b], window, **args)
  elif ptype == "ALL_NEIGHBORS":
    if a == b:
      mp_columns_out = mp.selfjoin_knn(inputs[a], window, max_matches, **args)
    else:
      mp_columns_out = mp.abjoin_knn(inputs[a], inputs[b], window, max_matches, **args)
  elif ptype == "MATRIX_SUMMARY":
    if a == b:
      mp_columns_out = mp.selfjoin_matrix(inputs[a], window, **args)
    else:
      mp_columns_out = mp.abjoin_matrix(inputs[a], inputs[b], window, **args)
  else:
    raise ValueError('pyscamp does not support profile type {}'.format(ptype))

  if mp_columns_out is not None:
    mp_columns_out = mp_columns_out.squeeze()
  if mp_columns_out_index is not None:
    mp_columns_out_index = mp_columns_out_index.squeeze()
  if mp_rows_out is not None:
    mp_rows_out = mp_rows_out.squeeze()
  if mp_rows_out_index is not None:
    mp_rows_out_index = mp_rows_out_index.squeeze()
  
  return mp_columns_out, mp_columns_out_index, mp_rows_out, mp_rows_out_index

def read_file_to_array(filename):
  if not os.path.exists(filename):
    return None
  try:
    return np.array(pd.read_csv(filename, sep=' ', header=None, na_values='-nan(ind)')).squeeze()
  except pd.errors.EmptyDataError:
    print('Note: file {} was empty. Returning empty array.'.format(filename))
    return np.array([])

def run_scamp(inputs, a, b, window, tilesz, max_matches, thresh, ptype, rrows, rcols, keep_rows, aligned):
  if executable == 'pyscamp':
    if keep_rows:
      raise ValueError('keep_rows not supported with pyscamp.')
    if aligned:
      raise ValueError('aligned not supported with pyscamp.')
    return run_pyscamp(inputs, a, b, window, max_matches, thresh, ptype, rrows, rcols)

  args = f'--output_pearson --window={window} --input_a_file_name=a.txt {extra_opts}'

  if a != b:
    args += f' --input_b_file_name=b.txt'
  args += f' --max_tile_size={tilesz} --profile_type={ptype}'

  if keep_rows:
    args += ' --keep_rows_separate'

  if aligned:
    args += ' --aligned'

  if thresh is not None:
    args += f' --threshold={thresh}'
  
  if rrows is not None and rcols is not None:
    args += f' --reduced_height={rrows} --reduced_width={rcols}'

  if max_matches is not None:
    args += f' --max_matches_per_column={max_matches}'

  print(args)
  
  ret = subprocess.call(os.path.abspath(executable) + ' ' + args, shell=True)
  
  mp_columns_out = read_file_to_array('mp_columns_out')
  mp_columns_out_index = read_file_to_array('mp_columns_out_index')  
  mp_rows_out = read_file_to_array('mp_rows_out')
  mp_rows_out_index = read_file_to_array('mp_rows_out_index')

  return mp_columns_out, mp_columns_out_index, mp_rows_out, mp_rows_out_index


def run_test(test, inputs):
  a = test[0]
  a_data = inputs[a]
  b = test[1]
  b_data = inputs[b]
  window = test[2]
  if a != b:
    dm = distance_matrix_np(a_data,b_data,window)
  else:
    dm = distance_matrix_np(a_data,None,window)

  dm_reductions = {}
  dm_reductions[('1NN_INDEX', None, None, None)] = reduce_1nn_index(dm)
  dm_reductions[('1NN', None, None, None)] = reduce_1nn(dm)
  dm_reductions[('ALL_NEIGHBORS', None, None, None)] = dm
  
  result_sum = []
  for thresh in thresholds_to_test:
    dm_reductions[('SUM_THRESH', thresh, None, None)] = reduce_sum_thresh(dm, thresh)
  for rows in matrix_sizes_to_test:
    if dm.shape[0] < rows:
      continue
    for cols in matrix_sizes_to_test:
      if dm.shape[1] < cols:
        continue
      dm_reductions[('MATRIX_SUMMARY', None, rows, cols)] = reduce_matrix(dm, rows,cols, a == b)
  
  # We only need local files for the CLI
  if executable != 'pyscamp':
    np.savetxt('a.txt', a_data)
    np.savetxt('b.txt', b_data)
  
  subtests = {}
  prev_tile_size = None
  for tile_sz in tile_sizes_to_test:
    if prev_tile_size is not None and prev_tile_size > len(a_data) and prev_tile_size > len(b_data):
      break
    subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "1NN_INDEX", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
    subtest_args = tuple(subtest_dict.values())
    scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "1NN_INDEX", None, None,False, False);
    valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
    subtests[subtest_args] = valid
 
    # Pyscamp does not support 1NN profiles
    if executable != 'pyscamp':
      subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "1NN", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
      subtest_args = tuple(subtest_dict.values())
      scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "1NN", None, None,False, False);
      valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
      subtests[subtest_args] = valid
   

    for thresh in thresholds_to_test:
      subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': thresh, 'ptype': "SUM_THRESH", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
      subtest_args = tuple(subtest_dict.values())
      scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, thresh, "SUM_THRESH", None, None,False, False);
      valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
      subtests[subtest_args] = valid

      # KNN MPs are only supported when cuda devices are available and SCAMP is built with CUDA.
      # KNN not supported on both CPU/GPU yet
      '''
      if gpu_enabled: 
        subtest_dict = {'tilesz': tile_sz, 'matchpercol': 5, 'threshold': thresh, 'ptype': 'ALL_NEIGHBORS', 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
        subtest_args = tuple(subtest_dict.values())
        scamp_results = run_scamp(inputs, a, b, window, tile_sz, 5, thresh, "ALL_NEIGHBORS", None, None,False, False);
        valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
        subtests[subtest_args] = valid
      '''

    for rrow in matrix_sizes_to_test:
      if rrow >= len(inputs[b]) - window + 1:
        continue
      for rcol in matrix_sizes_to_test:
        if rcol >= len(inputs[a]) - window + 1:
          continue
        # GPUs do not currently output the exact matrix summary, it is not currently possible to use the current verification method on GPU output.
        if not gpu_enabled:
          subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "MATRIX_SUMMARY", 'rrow': rrow, 'rcol': rcol, 'keeprows': False, 'aligned': False}
          subtest_args = tuple(subtest_dict.values())
          scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "MATRIX_SUMMARY", rrow, rcol, False, False);
          valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
          subtests[subtest_args] = valid

    prev_tile_size = tile_sz

  if os.path.exists('a.txt'):
    os.remove('a.txt')
  if os.path.exists('b.txt'):
    os.remove('b.txt')
        
  return subtests

def all_tests_passed(results):
  test_count = 0
  correct_tests = 0
  for t, subtests in results:
    for key, test in subtests.items():
      test_count += 1
      if test:
        correct_tests += 1
      else:
         print(t)
         print(key)
  print(f'{correct_tests} of {test_count} tests passed')
  return test_count == correct_tests

inputs = generate_input_arrays(input_sizes_to_test)
file_inputs = read_file_inputs(static_test_cases)
inputs += [x for x in file_inputs]

input_sizes_to_test += [len(x) for x in file_inputs]

tests = generate_tests(input_sizes_to_test, window_sizes_to_test)

all_results = []
for test in tqdm(tests):
  alen = len(inputs[test[0]])
  blen = len(inputs[test[1]])
  results = run_test(test, inputs)
  for key, result in results.items():
    if not result:
      print(f'A = {alen}, B = {blen}, m = {test[2]}, tile size = {key[0]}, matchpercol = {key[1]}, threshold = {key[2]}, profile type = {key[3]}, reduced_rows = {key[4]}, reduced cols = {key[5]}, keep_rows = {key[6]}, aligned = {key[7]}')
      print('FAIL!')
  all_results.append([test,results])


if all_tests_passed(all_results):
  exit(0)

exit(1)

