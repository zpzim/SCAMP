#!/usr/bin/python
import os
import sys
import math

fd1 = open(sys.argv[1], 'r')
fd2 = open(sys.argv[2], 'r')
fd3 = open(sys.argv[3], 'w')

maxdiff = 0
count = 1
for line1, line2 in zip(fd1,fd2):
    if 'nan' in line1:
      line1 = 'nan' 
    if 'nan' in line2:
      line2 = 'nan'
    x = float(line1)
    x2 = float(line2)
    if math.isnan(x) != math.isnan(x2) or math.isinf(x) != math.isinf(x2):
      print('Failure: values do not agree') 
      print('Line: ', count, x, x2)
      exit(1)
    diff = x - x2;
    fd3.write(str(x - x2) + "\n")
    if(abs(diff) > maxdiff):
        maxdiff = abs(diff)
    if abs(diff) > 0.01:
        print('Failure: matrix profile differs from the ground truth by more than the threshold')  
        print('Line: ', count, x, x2)
        exit(1)
    count += 1

print("Max matrix profile difference was " + str(maxdiff))



