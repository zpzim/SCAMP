import os
import sys
import random


#Generates a random list of floating point values between 0 and 10
fd = open(sys.argv[3], "w")
upper_lower_bound = float(sys.argv[2]);
num = 0
for i in range(0, int(sys.argv[1])):
    num += random.uniform(-upper_lower_bound,upper_lower_bound);
    fd.write(str(num) + "\n")

fd.close()
