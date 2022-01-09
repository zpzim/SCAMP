import numpy as np
import math

def moving_mean(a, w):
  result = np.zeros((len(a) - w + 1,))
  p = a[0]
  s = 0
  for i in range(1, w):
    x = p + a[i]
    z = x - p
    s += (p - (x - z)) + (a[i] - z)
    p = x
  
  result[0] = (p + s) / w

  for i in range(w, len(a)):
    x = p - a[i - w]
    z = x - p
    s += (p - (x - z)) - (a[i - w] + z)
    p = x

    x = p + a[i]
    z = x - p
    s += (p - (x - z)) + (a[i] - z)
    p = x
    result[i - w + 1] = (p + s) / w

  return result;

def sum_of_squared_differences(a, means, w):
  result = np.zeros((len(a) - w + 1,))
  for i in range(len(a) - w + 1):
    vals = a[i:i+w] - means[i]
    vals = vals * vals
    result[i] = np.sum(vals)
  return result
  
def get_precomputes(T, m, nanvalues):
  flatness_epsilon = 1e-13
  n = len(T) - m + 1;
  df = np.zeros((n,))
  dg = np.zeros((n,))

  means = moving_mean(T,m)

  norms = sum_of_squared_differences(T, means, m)

  for i in range(n):
    if nanvalues[i]:
      norms[i] = np.nan
    elif norms[i] <= flatness_epsilon:
      norms[i] = np.nan
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

# Computes the distance matrix using the diagonal update method used in SCAMP
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

    out[np.isnan(out)] = -2;
    return out


# Computes the distance matrix using np.corrcoef
def distance_matrix_np(a,b,m):
  has_b = True
  if b is None:
    has_b = False
    b = a

  na = len(a) - m  + 1
  nb = len(b) - m  + 1

  a, nan_a = convert_non_finite_to_zero(a,m)

  if has_b:
    b, nan_b = convert_non_finite_to_zero(b,m)
  else:
    b = a;

  mua, siga, _, _ = get_precomputes(a,m,nan_a)
  if not has_b:
      mub = mua
      sigb = siga
  else:
      mub, sigb, _, _ = get_precomputes(b,m, nan_b)

  
  x = np.zeros((na,m))
  y = np.zeros((nb,m))
  for i in range(na):
    if (np.isnan(siga[i])):
      x[i,:] = np.nan
    else:
      x[i,:] = a[i:i+m]

  if has_b:
    for i in range(nb):
      if (np.isnan(sigb[i])):
        y[i,:] = np.nan
      else:
        y[i,:] = b[i:i+m]

  if has_b:
    out = np.corrcoef(x,y=y)
    # Take the AB-join portion of the correlation matrix
    out = out[na:, :na]
  else:
    out = np.corrcoef(x)

  # Mark exclusion zone
  if not has_b:
    minlag = m // 4 - 1
    for i in range(nb):
      x = max(0, i - minlag)
      y = min(na, i + minlag + 1)
      out[i,x:y] = -2 

  out[np.isnan(out)] = -2;
  return out

# Simple (slow) distance matrix generation using scipy.stats.pearsonr for reference
'''
def distance_matrix_simple(a,b, m):
  has_b = True
  if b is None:
    has_b = False
    b = a
  na = len(a) - m  + 1
  nb = len(b) - m  + 1
  out = np.ones((nb,na)) * -2
  for i in range(na):
    x = a[i:i+m]
    for j in range(nb):
      out[j,i] = pearsonr(x, b[j:j+m])[0]
  if not has_b:
    minlag = m // 4 - 1
    for i in range(nb):
      x = max(0, i - minlag)
      y = min(na, i + minlag + 1)
      out[i,x:y] = -2 
  out[np.isnan(out)] = -2;
  return out
'''
    
def reduce_1nn_index(dm):
    corr = np.amax(dm, axis=0)
    idxs = np.argmax(dm, axis=0)
    idxs[corr == -2] = -1
    corr[corr == -2] = np.nan
    return corr.squeeze(), idxs.squeeze()

def reduce_1nn(dm):
    corrs = np.amax(dm, axis=0)
    corrs[corrs == -2] = np.nan
    return corrs.squeeze()

def reduce_sum_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 <= thresh] = 0

    result = np.sum(dm2, dtype='float64', axis=0)
    return result.squeeze()

def reduce_frequency_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 > thresh] = 1
    dm2[dm2 <= thresh] = 0
  
    result = np.sum(dm2, dtype='float64', axis=0)
    return result.squeeze()
  

def reduce_matrix(dm_orig, rows, cols, self_join):
    dm = np.copy(dm_orig)
    if self_join:
      # In a self join. SCAMP only computes the upper triagnular
      # portion of the distance matrix. We need to erase the bottom
      # half to not get a different reduction in these cases. 
      for col in range(dm.shape[1]):
        if col + 1 >= dm.shape[0]:
          break
        dm[col + 1:, col] = np.nan

    reduced_rows = dm.shape[0] / rows
    reduced_cols = dm.shape[1] / cols
    out = np.ones((rows,cols)) * -2
    for r in range(rows):
      st_r = math.ceil(r * reduced_rows)
      ed_r = min(dm.shape[0], math.ceil((r+1)* reduced_rows))
      for c in range(cols):
        st_c = math.ceil(c * reduced_cols)
        ed_c = min(dm.shape[1], math.ceil((c+1)*reduced_cols))
        out[r, c] = np.nanmax(dm[st_r:ed_r, st_c:ed_c])
    out[out == -2] = np.nan
    return out

