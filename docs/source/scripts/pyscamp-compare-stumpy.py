if __name__ == '__main__':
  import pyscamp as mp
  import stumpy as mp_stump
  import numpy as np
  import pandas as pd
  import random 
  import sys
  import time
  import psutil
  import GPUtil as gputil

  thread_count = psutil.cpu_count(logical=True)
  print("="*40, "CPU Info", "="*40)
  print("Physical cores:", psutil.cpu_count(logical=False))
  print("Total cores:", psutil.cpu_count(logical=True))
  cpufreq = psutil.cpu_freq()
  print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
  print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
  print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
  for gpu in gputil.getGPUs():
    print('GPU: {} | {}'.format(gpu.id,gpu.name))

  input_sizes = [2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18, 2 ** 19, 2 ** 20, 2 ** 21]
  pyscamp_cpu = []
  stumpy_cpu = []
  pyscamp_gpu_sp = []
  pyscamp_gpu_dp = []
  pyscamp_cpu_gpu_dp = []
  stumpy_gpu = []

  print('Running stumpy one time before performing comparisons as it has some initialization cost the first run.')
  start = time.time()
  mp_stump.gpu_stump(np.array([1,2,3,4,5,6]).astype(float), 4, device_id=[0,1])
  end = time.time()
  print('Stumpy took {} seconds to initialize'.format(end-start))

  print('Starting perfomance comparison.')

  for sz in input_sizes:
    print(sz)
    arr = np.random.random(size=(sz,))

    start = time.time()
    mp.selfjoin(arr, 100, pearson=False, gpus=[])
    end = time.time()
    pyscamp_cpu.append(end-start)
    
    start = time.time()
    mp.selfjoin(arr, 100, pearson=False, gpus=[0, 1], threads=0, precision='single')
    end = time.time()
    pyscamp_gpu_sp.append(end-start)

    start = time.time()
    mp.selfjoin(arr, 100, pearson=False, gpus=[0, 1], threads=0, precision='double')
    end = time.time()
    pyscamp_gpu_dp.append(end-start)


    start = time.time()
    mp.selfjoin(arr, 100, pearson=False, gpus=[0, 1], threads=thread_count, precision='double')
    end = time.time()
    pyscamp_cpu_gpu_dp.append(end-start)
    
    if sz > 2 ** 19:
      # Stumpy CPU is extremely slow for larger inputs. Don't bother comparing here.
      stumpy_cpu.append(np.nan)
    else:
      start = time.time()
      mp_stump.stump(arr, 100)
      end = time.time()
      stumpy_cpu.append(end-start)

    start = time.time()
    mp_stump.gpu_stump(arr, 100, device_id=[0,1])
    end = time.time()
    stumpy_gpu.append(end-start)

  pyscamp_cpu_advantage = np.array(stumpy_cpu) / np.array(pyscamp_cpu)
  pyscamp_gpu_dp_adv = np.array(stumpy_gpu) / np.array(pyscamp_gpu_dp)


  result_data = {'input_size': input_sizes, 'stumpy CPU ({} threads)'.format(thread_count) : stumpy_cpu, 'pyscamp CPU ({} threads)'.format(thread_count) : pyscamp_cpu, 'pyscamp CPU Advantage': pyscamp_cpu_advantage, 'stumpy GPU (fp64)': stumpy_gpu, 'pyscamp GPU fp64': pyscamp_gpu_dp,  'pyscamp GPU fp64 Advantage': pyscamp_gpu_dp_adv, 'pyscamp GPU fp32' : pyscamp_gpu_sp, 'pyscamp CPU ({} threads) + GPU fp64'.format(thread_count) : pyscamp_cpu_gpu_dp}

  result = pd.DataFrame(result_data)

  result.to_csv('pyscamp-stumpy-perf-comparison.csv')

  print(result.to_markdown())
