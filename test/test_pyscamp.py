import pyscamp as mp
import random 
import sys

arr = []
num = 0
for i in range(1,32000):
  num += random.uniform(-0.1, 0.1)
  arr.append(num)

mp.scamp(arr, 1024)
print("CPU PASS")

if mp.gpu_supported():
  mp.scamp_knn(arr, 1024, 5)
  print("GPU PASS")

