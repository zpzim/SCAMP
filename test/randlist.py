import os
import sys
import random


#Generates a random list of floating point values between 0 and 10
fd = open(sys.argv[2], "w")

for i in range(0, int(sys.argv[1])):
    fd.write(str(random.randint(0, 1000000000) / float(100000000)) + "\n")

fd.close()
