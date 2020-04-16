import pySCAMP as mp
import pySCAMPcpu as mp_cpu
import random 
import sys

test_cuda = True

if len(sys.argv) > 1:
  test_cuda = False

arr = []
num = 0
for i in range(1,32000):
  num += random.uniform(-0.1, 0.1)
  arr.append(num)

print("START PYSCAMP")
profile, index = mp.scamp(arr, 1024)
profile, index = mp.scamp(arr, arr, 1024)

matches = mp.scamp_knn(arr, 1024, 3);
print(matches)
print("DONE PYSCAMP")

print("START PYSCAMP_CPU")
mp, index = mp_cpu.scamp(arr, 1024)
mp, index = mp_cpu.scamp(arr, arr, 1024)
print("DONE PYSCAMP_CPU")
