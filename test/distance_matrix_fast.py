import numpy as np
import pandas as pd
import math
import sys
import numba

def get_precomputes(T, m, nanvalues):
  prefix_sum = np.zeros((len(T),))
  prefix_sum_sq = np.zeros((len(T),))
  n = len(T) - m + 1;
  norms = np.zeros((n,))
  means = np.zeros((n,))
  df = np.zeros((n,))
  dg = np.zeros((n,))

  prefix_sum[0] = T[0]
  prefix_sum_sq[0] = T[0] * T[0]
  for i in range(1,len(T)):
    prefix_sum[i] = T[i] + prefix_sum[i - 1]
    prefix_sum_sq[i] = T[i] * T[i] + prefix_sum_sq[i - 1]

  means[0] = prefix_sum[m - 1] / m
  for i in range(1,n):
    means[i] = (prefix_sum[i + m - 1] - prefix_sum[i - 1]) / m

  s = 0;
  for i in range(m):
    val = T[i] - means[0]
    s += val * val
  norms[0] = s

  for i in range(1,n):
    norms[i] = norms[i - 1] + ((T[i - 1] - means[i - 1]) + (T[i + m - 1] - means[i])) * (T[i + m - 1] - T[i - 1])

  for i in range(n):
    if nanvalues[i]:
      norms[i] = NaN
    else:
      norms[i] = 1.0 / math.sqrt(norms[i])

  for i in range(n-1):
    df[i] = (T[i + m] - T[i]) / 2.0;
    dg[i] = (T[i + m] - means[i + 1]) + (T[i] - means[i]);

  return means, norms, df, dg
    

'''
// Converts any NaN or inf values in the input to 0, returns the cleaned
// timeseries in timeseries_clean, and returns the subsequences which contained
// NaN in nanvals
void convert_non_finite_to_zero(const std::vector<double> &T, const int m,
                                std::vector<double> *timeseries_clean,
                                std::vector<bool> *nanvals) {
  timeseries_clean->resize(T.size());
  nanvals->resize(T.size() - m + 1);
  size_t steps_since_last_nan = m;
  for (int i = 0; i < T.size(); ++i) {
    if (std::isfinite(T[i])) {
      timeseries_clean->at(i) = T[i];
    } else {
      steps_since_last_nan = 0;
      timeseries_clean->at(i) = 0;
    }
    if (i >= m - 1) {
      nanvals->at(i - m + 1) = steps_since_last_nan < m;
    }
    steps_since_last_nan++;
  }
}
'''

def convert_non_finite_to_zero(T, m):
  timeseries_clean = np.zeros((len(T),))
  nanvals = np.zeros((len(T) - m + 1,))
  steps_since_last_nan = m
  for i in range(len(T)):
    if np.isfinite(T[i]):
      timeseries_clean[i] = T[i]
    else:
      steps_since_last_nan = 0
      timeseries_clean[i] = 0
    if i >= m - 1:
      nanvals[i - m + 1] = steps_since_last_nan < m
    steps_since_last_nan += 1
  return timeseries_clean, nanvals

'''
def get_precomputes(T, m):
  prefix_sum = np.zeros((len(T),))
  prefix_sum_sq = np.zeros((len(T),))
  n = len(T) - m + 1;
  norms = np.zeros((n,))
  means = np.zeros((n,))
  df = np.zeros((n,))
  dg = np.zeros((n,))

  prefix_sum[0] = T[0]
  prefix_sum_sq[0] = T[0] * T[0]
  for i in range(1,len(T)):
    prefix_sum[i] = T[i] + prefix_sum[i - 1]
    prefix_sum_sq[i] = T[i] * T[i] + prefix_sum_sq[i - 1]

  means[0] = prefix_sum[m - 1] / m
  for i in range(1,n):
    means[i] = (prefix_sum[i + m - 1] - prefix_sum[i - 1]) / m

  s = 0;
  for i in range(m):
    val = T[i] - means[0]
    s += val * val
  norms[0] = s

  for i in range(1,n):
    norms[i] = norms[i - 1] + ((T[i - 1] - means[i - 1]) + (T[i + m - 1] - means[i])) * (T[i + m - 1] - T[i - 1])

  for i in range(n):
    if nanvalues[i]:
      norms[i] = NaN
    else:
    norms[i] = 1.0 / math.sqrt(norms[i])

  for i in range(n-1):
    df[i] = (T[i + m] - T[i]) / 2.0;
    dg[i] = (T[i + m] - means[i + 1]) + (T[i] - means[i]);

  return means, norms, df, dg
'''

# Fast mp code for python (only computes self joins)
@numba.jit(nopython=True)
def do_mp(a,w,mua,siga,dfa,dga,mub,sigb,dfb,dgb):

    b = a
    na = len(a) - w + 1
    nb = na

    # Mark the last diagonal
    diagmax = na

    # This is the exclusion zone
    minlag = w // 4 + 1
    
    #Result MP
    mp = np.zeros((na,))    

    #Initialize starting covariance for the first row of the distance matrix
    c = np.zeros((diagmax - minlag,)) 
    for diag in range(minlag,diagmax):
      c[diag-minlag] = np.sum((a[diag:diag+w]-mua[diag]) * (b[:w]-mub[0]))
    

    #Compute the first row of correlations, this could be done inside the loop, but because python can't slice arrays like arr[:-0] we need to do the first iteration outside
    result = c*(sigb[0]*siga[minlag:])
    # Reduce along the rows
    mp[minlag:] = np.maximum(mp[minlag:], result)
    # Reduce along the column
    mp[0] = max(mp[0], np.amax(result))
    #The number of diagonals we compute gets smaller each iteration
    #Update the covariance for the next iteration
    c[:-1] = c[:-1] + dfb[0] * dga[minlag:-1] + dfa[minlag:-1]*dgb[0]

    # Repeat the process for each row
    for offset in range(1, nb-minlag):
      # Compute the correlations
      result = c[:-offset]*(sigb[offset]*siga[minlag+offset:])
      # Reduce along the row
      mp[minlag+offset:] = np.maximum(mp[minlag+offset:], result )
      # Reduce along the columns
      mp[offset] = max(mp[offset], np.amax(result))
      # Update the covatiance values for the next iteration
      c[:-offset-1] = c[:-offset-1] + dfb[offset] * dga[minlag+offset:-1] + dfa[minlag+offset:-1]*dgb[offset]

    return mp


def matrix_profile(a, w):
    mua, siga, dfa, dga = get_precomputes(a,w)
    mub = mua
    sigb = siga
    dfb = dfa
    dgb = dga

    return do_mp(a,b,w,mua,siga,dfa,dga,mub,sigb,dfb,dgb)


def distance_matrix(a,b,w):
    has_b = True
    if b is None:
        has_b = False
        b = a
    na = len(a) - w + 1
    if not has_b:
        nb = na
    else:
        nb = len(b) - w + 1
    out = np.zeros((nb,na))    


    a, nan_a = convert_non_finite_to_zero(a,w)
  
    if has_b:
      b, nan_b = convert_non_finite_to_zero(b,w)
    else:
      b = a;

    mua, siga, dfa, dga = get_precomputes(a,w,nan_a)
    if not has_b:
        mub = mua
        sigb = siga
        dfb = dfa
        dgb = dga
    else:
        mub, sigb, dfb, dgb = get_precomputes(b,w, nan_b)

    diagmax = na
    if not has_b:
        minlag = w // 4 + 1
    else:
        minlag = 0

    
    c = np.zeros((diagmax - minlag,)) 
    for diag in range(minlag,diagmax):
      c[diag-minlag] = np.sum((a[diag:diag+w]-mua[diag]) * (b[:w]-mub[0]))
    
    for offset in range(nb-minlag):
      result = c*(sigb[offset]*siga[minlag+offset:])
      out[offset, minlag+offset:] = result
      if not has_b:
        out[minlag+offset:, offset] = result
      x = c + dfb[offset] * dga[minlag+offset:] + dfa[minlag+offset:]*dgb[offset]
      c = x[:-1]
    if has_b:
      diagmax = nb
      c = np.zeros((diagmax - minlag,)) 
      for diag in range(minlag,diagmax):
        c[diag-minlag] = np.sum((b[diag:diag+w]-mub[diag]) * (a[:w]-mua[0]))
      for offset in range(na-minlag):
        result = c*(siga[offset]*sigb[minlag+offset:])
        out[minlag+offset:, offset] = result
        x = c + dfa[offset] * dgb[minlag+offset:] + dfb[minlag+offset:]*dga[offset]
        c = x[:-1]

    return out

def reduce_1nn_index(dm):
    idxs = np.argmax(dm, axis=0)
    corrs = np.zeros((len(idxs),))
    for i, idx in enumerate(idxs):
        corrs[i] = dm[idx, i]
    return corrs.reshape((corrs.shape[0],1)), idxs.reshape((idxs.shape[0],1))

def reduce_1nn(dm):
    corrs = np.amax(dm, axis=0)
    return corrs.reshape((corrs.shape[0],1))

def reduce_sum_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 < thresh] = 0
    result = np.sum(dm2, axis=0)
    return result.reshape((result.shape[0],1))

def reduce_matrix(dm, rows, cols):
    out = np.ones((rows,cols)) * -1
    reduced_rows = math.ceil(dm.shape[0] / rows)
    reduced_cols = math.ceil(dm.shape[1] / cols)
    for row in range(dm.shape[0]):
      rrow = row // reduced_rows
      for col in range(dm.shape[1]):
        rcol = col // reduced_cols
        if dm[row,col] > out[rrow, rcol]:
          out[rrow,rcol] = dm[row,col]
    return out

def test(filenamea, filenameb, window, expected_output):
    a = np.array(pd.read_csv(filenamea, header=None)).flatten()
    if filenameb is None:
        b = None
    else:
        b = np.array(pd.read_csv(filenameb, header=None)).flatten()
   
    out = np.array(pd.read_csv(expected_output, header=None)).flatten()
    out = 1 - (out*out / (2 * window))
    #dm = distance_matrix(a,b,window)
    #mp = reduce_1nn(dm)
    mp = matrix_profile(a,b,window)
    diff = np.abs(mp - out)
    epsilon = 1e-3
    return not np.any(diff > epsilon)

def run_test():
    if sys.argv[1] == sys.argv[2]:
        return test(sys.argv[1], None, int(sys.argv[3]), sys.argv[4])
    else:
        return test(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

def run():
    result = run_test()
    if result:
        print('Test Succeeded!')
    else:
        print('Test Failed!')
