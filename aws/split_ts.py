import sys
from itertools import islice

full_ts = sys.argv[1]
lines_per_file = int(sys.argv[2])
window_size = int(sys.argv[3])
output_dir = sys.argv[4]

def write_lines(lines,file_count):
    small_filename = 'segment_{}'.format(file_count)
    smallfile = open(output_dir+"/"+small_filename, "w")
    for line in lines:
        smallfile.write(line)

def process_lines(f,t,w):
    c = 0
    prevlines = list(islice(bigfile,t))
    if len(prevlines) != t:
        write_lines(prevlines,c)
        c += 1
        return c
    while True:
        lines = list(islice(bigfile,t - w + 1))
        if len(lines) is 0:
            return c
        if len(lines) < 0.5 * t:
            prevlines = prevlines + lines
            write_lines(prevlines,c)
            c += 1
            return c
        if len(lines) != t - w + 1:
            write_lines(prevlines,c)
            c += 1
            prevlines = prevlines[-w+1:]
            prevlines = prevlines + lines
            write_lines(prevlines,c)
            c += 1
            return c
        write_lines(prevlines,c)
        c += 1
        prevlines = prevlines[-w+1:]
        prevlines = prevlines + lines



if lines_per_file <= window_size:
    print("Error: window size larger than split size")
    exit(1)


bigfile = open(full_ts,"r");
files = process_lines(bigfile,lines_per_file,window_size)
print("Split time series into " + str(files) + " files")
bigfile.close()

