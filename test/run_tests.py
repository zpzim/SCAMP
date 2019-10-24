import sys, os
from distance_matrix_fast import *
import numpy as np

np.random.seed(13)

executable = sys.argv[1]
outfile = sys.argv[2]
extra_opts = sys.argv[3]

window_sizes_to_test = [3, 7, 11, 100, 500]
#tile_sizes_to_test = [1024, 4987, 6255, 9200, 16000, 32000, 640000]
tile_sizes_to_test = [1024, 2048, 4096]
#input_sizes_to_test = [100, 250, 500, 1000, 4000, 8000, 160000, 32000]
input_sizes_to_test = [1000, 2000, 4000, 16000]
#matrix_sizes_to_test = [1, 16, 32, 64, 128, 256, 512]
matrix_sizes_to_test = [1, 16]
#thresholds_to_test = [-1.0, 0.0, 0.5, 0.7, 0.9, 0.99, 1]
#thresholds_to_test = [0.0, 0.5, 0.7, 0.9, 0.99, 1]
#thresholds_to_test = [0.0, 0.5, 0.9]
thresholds_to_test = [0.0]

index_match_ratio = 0.001
matrix_match_ratio = 0.001

vector_match_epsilon_SUM = 0.1
vector_match_epsilon_1NN = 0.001

matrix_match_epsilon = 0.001

def generate_input_arrays(input_sizes):
  arrs = []
  for s in input_sizes:
    arrs.append(np.random.randn(s))
  return arrs

def generate_tests(input_sizes, windows):
  tests = set()
  for a in range(len(input_sizes)):
    for b in range(len(input_sizes)):
        for window in windows:
          if window <= input_sizes[a] and window <= input_sizes[b]:
            tests.add((a, b, window))
  return tests

def compare_index(valid, check):
  if np.any(valid.shape != check.shape):
    #print(valid.shape, ' ', check.shape)
    print('Output Shapes do not match')
    return False
  
  incorrect = np.count_nonzero(valid - check + 1) 
  ratio = incorrect / len(valid) 
  #print(ratio)
  return ratio < index_match_ratio

def compare_vectors(valid, check, eps):
  if np.any(valid.shape != check.shape):
    #print(valid.shape, ' ', check.shape)
    print('Output Shapes do not match')
    return False
    
  diff = np.abs(valid - check)
  if np.max(diff) > eps:
    print(np.max(diff))
    #print(np.concatenate((valid, check, diff), axis=1))

  #incorrect = np.count_nonzero(diff > vector_match_epsilon)
  #print(diff / valid)
  #print(np.max(diff / valid))
  return np.max(diff) < eps

  ''' 
  for val, chk in zip(valid, check):
    if abs(val - chk
  '''
  
  #return True

def compare_matrix(valid, check):
  if np.any(valid.shape != check.shape):
    #print(valid.shape, ' ', check.shape)
    print('Output Shapes do not match')
    return False
  
  #print(valid)
  #print(check)
  diff = np.abs(valid - check)
  #print(diff)
  incorrect = np.count_nonzero(diff > matrix_match_epsilon)
  #print(incorrect)

  #print(valid.shape[0] * valid.shape[1])
  ratio = incorrect / (valid.shape[0] * valid.shape[1])
  #print(ratio)

  return ratio < matrix_match_ratio


def evaluate_result(dm_reductions, scamp_results, subtestargs):
  ptype = subtestargs['ptype']
  if ptype == "1NN_INDEX":
    valid_data = dm_reductions[("1NN_INDEX",None,None,None)]
    return compare_index(valid_data[1], scamp_results[1]) and compare_vectors(valid_data[0], scamp_results[0], vector_match_epsilon_1NN)
  
  if ptype == "1NN":
    valid_data = dm_reductions[("1NN",None,None,None)]
    return compare_vectors(valid_data, scamp_results[0], vector_match_epsilon_1NN)
    
  if ptype == "SUM_THRESH":
    valid_data = dm_reductions[("SUM_THRESH",subtestargs['threshold'],None,None)]
    return compare_vectors(valid_data, scamp_results[0], vector_match_epsilon_SUM)
  
  if ptype == "ALL_NEIGHBORS_MATRIX":
    valid_data = dm_reductions[("ALL_NEIGHBORS_MATRIX", None, subtestargs['rrow'], subtestargs['rcol'])] 
    valid_data[valid_data < subtestargs['threshold']] = -1.0
    #print(scamp_results[0].shape)
    return compare_matrix(valid_data, scamp_results[0][:,:-1])

  return None
  
    
    
  

  

def run_scamp(inputs, a, b, window, tilesz, max_matches, thresh, ptype, rrows, rcols, keep_rows, aligned):
  command = executable + f' --output_pearson --window={window} --input_a_file_name=a.txt'
  #print(a)
  if a != b:
    command += f' --input_b_file_name=b.txt'
  command += f' --max_tile_size={tilesz} --profile_type={ptype}'

  if keep_rows:
    command += ' --keep_rows_separate'

  if aligned:
    command += ' --aligned'

  if thresh is not None:
    command += f' --threshold={thresh}'
  
  if rrows is not None and rcols is not None:
    command += f' --reduce_all_neighbors --reduced_height={rrows} --reduced_width={rcols}'

  if max_matches is not None:
    command += f' --max_matches_per_column={max_matches}'

  print(command)

  ret = os.system(command)
  
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
  #print(f'a = {len(a_data)}, b = {len(b_data)}, window = {window}')
  if a != b:
    dm = distance_matrix(a_data,b_data,window)
  else:
    dm = distance_matrix(a_data,None,window)
  
  dm_reductions = {}
  dm_reductions[('1NN_INDEX', None, None, None)] = reduce_1nn_index(dm)
  dm_reductions[('1NN', None, None, None)] = reduce_1nn(dm)
  
  result_sum = []
  for thresh in thresholds_to_test:
    dm_reductions[('SUM_THRESH', thresh, None, None)] = reduce_sum_thresh(dm, thresh)
  for rows in matrix_sizes_to_test:
    for cols in matrix_sizes_to_test:
      dm_reductions[('ALL_NEIGHBORS_MATRIX', None, rows, cols)] = reduce_matrix(dm, rows,cols)
  
  
  np.savetxt('a.txt', a_data)
  np.savetxt('b.txt', b_data)
  
  subtests = {}
  for tile_sz in tile_sizes_to_test:
    subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "1NN_INDEX", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
    subtest_args = tuple(subtest_dict.values())
    #subtest_args = (tile_sz, None, None, "1NN_INDEX", None, None, False, False)
    scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "1NN_INDEX", None, None,False, False);
    valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
    subtests[subtest_args] = valid
 
    subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': None, 'ptype': "1NN", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
    subtest_args = tuple(subtest_dict.values())
    #subtest_args = (tile_sz, None, None, "1NN", None, None, False, False)
    scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, None, "1NN", None, None,False, False);
    valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
    subtests[subtest_args] = valid
  
    for thresh in thresholds_to_test:
      subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': thresh, 'ptype': "SUM_THRESH", 'rrow': None, 'rcol': None, 'keeprows': False, 'aligned': False}
      subtest_args = tuple(subtest_dict.values())
      #subtest_args = (tile_sz, None, thresh, "SUM_THRESH", None, None, False, False)
      scamp_results = run_scamp(inputs, a, b, window, tile_sz, None, thresh, "SUM_THRESH", None, None,False, False);
      valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
      subtests[subtest_args] = valid

    for rrow in matrix_sizes_to_test:
      for rcol in matrix_sizes_to_test:
        for thresh in thresholds_to_test:
          subtest_dict = {'tilesz' : tile_sz, 'matchpercol' : None, 'threshold': thresh, 'ptype': "ALL_NEIGHBORS_MATRIX", 'rrow': rrow, 'rcol': rcol, 'keeprows': False, 'aligned': False}
          subtest_args = tuple(subtest_dict.values())
          #subtest_args = (tile_sz, 999999999, thresh, "ALL_NEIGHBORS_MATRIX", rrow, rcol, False, False)
          scamp_results = run_scamp(inputs, a, b, window, tile_sz, 999999999, thresh, "ALL_NEIGHBORS", rrow, rcol, False, False);
          valid = evaluate_result(dm_reductions,scamp_results,subtest_dict)
          subtests[subtest_args] = valid
  
  os.remove('a.txt')
  os.remove('b.txt')
        
  return subtests
    
tests = generate_tests(input_sizes_to_test, window_sizes_to_test)
inputs = generate_input_arrays(input_sizes_to_test)

all_results = []
for test in tests:
  alen = len(inputs[test[0]])
  blen = len(inputs[test[1]])
  print("Running Test: ", test)
  results = run_test(test, inputs)
  for key, result in results.items():
    print(f'A = {alen}, B = {blen}, m = {test[2]}, tile size = {key[0]}, matchpercol = {key[1]}, threshold = {key[2]}, profile type = {key[3]}, reduced_rows = {key[4]}, reduced cols = {key[5]}, keep_rows = {key[6]}, aligned = {key[7]}')
    if result:
      print('PASS!')
    else:
      print('FAIL!')
  all_results.append(results)


if all_tests_passed(all_results):
  exit(0)

exit(1)

