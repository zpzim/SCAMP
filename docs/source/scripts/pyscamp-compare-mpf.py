import pyscamp as mp
import matrixprofile as mpf
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

input_sizes = [2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18, 2 ** 19]
pyscamp_cpu = []
mpx_cpu = []
scrimppp_cpu = []

print('Starting perfomance comparison.')

for sz in input_sizes:
  print(sz)
  arr = np.random.random(size=(sz,))

  start = time.time()
  mp.selfjoin(arr, 100, pearson=False, gpus=[])
  end = time.time()
  pyscamp_cpu.append(end-start)

  start = time.time()
  mpf.algorithms.mpx(arr, 100, n_jobs=thread_count)
  end = time.time()
  mpx_cpu.append(end-start)
    
  start = time.time()
  mpf.algorithms.scrimp_plus_plus(arr, 100, step_size=0.25, sample_pct=0.1, n_jobs=thread_count)  
  end = time.time()
  scrimppp_cpu.append(end-start)


pyscamp_cpu_advantage_mpx = np.array(mpx_cpu) / np.array(pyscamp_cpu)
pyscamp_cpu_advantage_scrimppp = np.array(scrimppp_cpu) / np.array(pyscamp_cpu)


result_data = {'input_size': input_sizes, 'mpx CPU ({} threads)'.format(thread_count): mpx_cpu, 'scrimp++ CPU (10% sampling, 25% step) ({} threads)'.format(thread_count): scrimppp_cpu, 'pyscamp CPU ({} threads)'.format(thread_count) : pyscamp_cpu, 'pyscamp CPU Advantage vs mpx': pyscamp_cpu_advantage_mpx, 'pyscamp CPU Advantage vs scrimp++': pyscamp_cpu_advantage_scrimppp }

result = pd.DataFrame(result_data)

result.to_csv('pyscamp-mpf-perf-comparison.csv')

print(result.to_markdown())
