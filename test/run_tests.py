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
matrix_sizes_to_test = []
thresholds_to_test = [0.0,0.125,0.5]

np.set_printoptions(edgeitems=30, linewidth=100000)
np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument('--executable', help='SCAMP executable to test', required=True)
parser.add_argument('--output_file', help='File for test std output')
parser.add_argument('--extra_args', help='Extra arguments to be passed to each test invocation')
parser.add_argument('--window_sizes', type=int, nargs='+')
parser.add_argument('--tile_sizes', type=int, nargs='+')
parser.add_argument('--input_sizes', type=int, nargs='+')
parser.add_argument('--matrix_sizes', type=int, nargs='+')
parser.add_argument('--thresholds', type=float, nargs='+')
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



index_match_ratio = 0.001
matrix_match_ratio = 0.001

vector_match_epsilon_SUM = 0.001
vector_match_epsilon_1NN = 0.001

matrix_match_epsilon = 0.001

min_reduce_ratio = 1024

matrix_check_ratio = 1 * min_reduce_ratio

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
    valid_data = dm_reductions[("1NN_INDEX",None,None,None)]
    return compare_index(valid_data[1], valid_data[0], scamp_results[1], scamp_results[0]) and compare_vectors(valid_data[0], scamp_results[0])
  
  if ptype == "1NN":
    valid_data = dm_reductions[("1NN",None,None,None)]
    return compare_vectors(valid_data, scamp_results[0])
    
  if ptype == "SUM_THRESH":
    valid_data = dm_reductions[("SUM_THRESH",subtestargs['threshold'],None,None)]
    return compare_vectors_sum(valid_data, scamp_results[0], subtestargs['threshold'])
  
  if ptype == "ALL_NEIGHBORS_MATRIX":
    valid_data = dm_reductions[("ALL_NEIGHBORS_MATRIX", None, subtestargs['rrow'], subtestargs['rcol'])] 
    valid_data[valid_data < subtestargs['threshold']] = -1.0
    return compare_matrix(valid_data, scamp_results[0][:,:-1])
  
  if ptype == "ALL_NEIGHBORS":
    valid_data = dm_reductions[("ALL_NEIGHBORS", subtestargs['threshold'], None, None)]
    return compare_all_neighbors(valid_data, scamp_results[0])
  
  return None

def run_scamp(inputs, a, b, window, tilesz, max_matches, thresh, ptype, rrows, rcols, keep_rows, aligned):
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
    args += f' --reduce_all_neighbors --reduced_height={rrows} --reduced_width={rcols}'

  if max_matches is not None:
    args += f' --max_matches_per_column={max_matches}'

  #print(args)

  ret = subprocess.call(os.path.abspath(executable) + ' ' + args, shell=True)
  
  mp_columns_out = None
  mp_columns_out_index = None
  mp_rows_out = None
  mp_rows_out_index = None
  if os.path.exists('mp_columns_out'):
    mp_columns_out = np.array(pd.read_csv('mp_columns_out', sep=' ', header=None))
    os.remove('mp_columns_out')
  if os.path.exists('mp_columns_out_index'):
    mp_columns_out_index = np.array(pd.read_csv('mp_columns_out_index', sep=' ',  header=None))
    os.remove('mp_columns_out_index')
  if os.path.exists('mp_rows_out'):
    mp_rows_out = np.array(pd.read_csv('mp_rows_out', sep=' ', header=None))
    os.remove('mp_rows_out')
  if os.path.exists('mp_rows_out_index'):
    mp_rows_out_index = np.array(pd.read_csv('mp_rows_out_index', sep=' ', header=None))
    os.remove('mp_rows_out_index')
  
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
  dm_reductions[('ALL_NEIGHBORS', 0.0, None, None)] = dm
  
  result_sum = []
  for thresh in thresholds_to_test:
    dm_reductions[('SUM_THRESH', thresh, None, None)] = reduce_sum_thresh(dm, thresh)
  for rows in matrix_sizes_to_test:
    for cols in matrix_sizes_to_test:
      dm_reductions[('ALL_NEIGHBORS_MATRIX', None, rows, cols)] = reduce_matrix(dm, rows,cols)
  
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
 
    subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "1NN", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
    subtest_args = tuple(subtest_dict.values())
    scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "1NN", None, None,False, False);
    valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
    subtests[subtest_args] = valid
   
    # All_neighbors not supported on both CPU/GPU yet 
    ''' 
    subtest_dict = {'tilesz': tile_sz, 'matchpercol': None, 'threshold': 0.0, 'ptype': 'ALL_NEIGHBORS', 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
    subtest_args = tuple(subtest_dict.values())
    scamp_results = run_scamp(inputs, a, b, window, tile_sz, 99999999, 0.0, "ALL_NEIGHBORS", None, None,False, False);
    valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
    subtests[subtest_args] = valid
    '''

    for thresh in thresholds_to_test:
      subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': thresh, 'ptype': "SUM_THRESH", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
      subtest_args = tuple(subtest_dict.values())
      scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, thresh, "SUM_THRESH", None, None,False, False);
      valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
      subtests[subtest_args] = valid
  
    

    for rrow in matrix_sizes_to_test:
      if rrow * min_reduce_ratio > len(b_data):
        continue
      for rcol in matrix_sizes_to_test:
        if rcol * min_reduce_ratio > len(a_data):
          continue
        for thresh in thresholds_to_test:
          subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': thresh, 'ptype': "ALL_NEIGHBORS_MATRIX", 'rrow': rrow, 'rcol': rcol, 'keeprows': False, 'aligned': False}
          subtest_args = tuple(subtest_dict.values())
          #subtest_args = (tile_sz, 999999999, thresh, "ALL_NEIGHBORS_MATRIX", rrow, rcol, False, False)
          scamp_results = run_scamp(inputs, a, b, window, tile_sz, 999999999, thresh, "ALL_NEIGHBORS", rrow, rcol, False, False);
          valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
          subtests[subtest_args] = valid

    prev_tile_size = tile_sz

  os.remove('a.txt')
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

print('Path to SCAMP executable is: ' + os.path.abspath(executable))  
  
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

