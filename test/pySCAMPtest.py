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

print("CPU START")
mp_cpu.SCAMP_SELF(arr, 1024)
print("CPU_DONE")

if test_cuda:
  import pySCAMP as mp
  print("GPU START")
  mp.SCAMP_SELF(arr, 1024)
  print("GPU DONE")

