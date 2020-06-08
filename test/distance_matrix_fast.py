import numpy as np
import math

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
    out = np.ones((nb,na)) * -2


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
        minlag = w // 4
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

# Simple (slow) distance matrix generation for reference
'''
def distance_matrix_simple(a,b, m):
  has_b = True
  if b is None:
    has_b = False
    b = a
  na = len(a) - m  + 1
  nb = len(b) - m  + 1
  out = np.zeros((nb,na))
  for i in range(na):
    for j in range(nb):
      out[j,i] = pearsonr(a[i:i+m], b[j:j+m])[0]
  if not has_b:
    minlag = m // 4 - 1
    for i in range(nb):
      x = max(0, i - minlag)
      y = min(na, i + minlag + 1)
      out[i,x:y] = 0
  return out
'''
    
def reduce_1nn_index(dm):
    corr = np.amax(dm, axis=0)
    idxs = np.argmax(dm, axis=0)
    idxs[corr == -2] = -1
    corr[corr == -2] = np.nan
    return corr.reshape((corr.shape[0],1)), idxs.reshape((idxs.shape[0],1))

def reduce_1nn(dm):
    corrs = np.amax(dm, axis=0)
    corrs[corrs == -2] = np.nan
    return corrs.reshape((corrs.shape[0],1))

def reduce_sum_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 <= thresh] = 0
     
    result = np.sum(dm2, dtype='float64', axis=0)
    return result.reshape((result.shape[0],1))

def reduce_frequency_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 > thresh] = 1
    dm2[dm2 <= thresh] = 0
  
    result = np.sum(dm2, dtype='float64', axis=0)
    return result.reshape((result.shape[0],1))
  

def reduce_matrix(dm, rows, cols):
    out = np.ones((rows,cols)) * -1
    reduced_rows = math.ceil(dm.shape[0] / rows)
    reduced_cols = math.ceil(dm.shape[1] / cols)
    for r in range(rows):
      st_r = r * reduced_rows
      ed_r = min(dm.shape[0], (r +1)* reduced_rows)
      for c in range(cols):
        st_c = c * reduced_cols
        ed_c = min(dm.shape[1], (c+1)*reduced_cols)
        out[r, c] = np.amax(dm[st_r:ed_r, st_c:ed_c])
    out[out == -2] = np.nan
    return out

