import sys

full_ts = sys.argv[1]
lines_per_file = int(sys.argv[2])
window_size = int(sys.argv[3])
output_dir = sys.argv[4]

if lines_per_file <= window_size:
    print "Error: window size larger than split size"
    exit(0)


smallfile = None
bigfile = open(full_ts,"r");
bigfilearr = bigfile.readlines();
global_lineno = 0
file_count = 0
lines_at_a_time = lines_per_file
while global_lineno < len(bigfilearr) - 1.5 * lines_per_file:
    small_filename = 'segment_{}'.format(file_count)
    smallfile = open(output_dir+"/"+small_filename, "w")
    for lineno in range(lines_at_a_time):
        line = bigfilearr[lineno+global_lineno]
        smallfile.write(line)
    smallfile.close()
    global_lineno += lines_per_file - window_size + 1
    file_count += 1

small_filename = 'segment_{}'.format(file_count)
smallfile = open(output_dir+"/"+small_filename, "w")
for lineno in range(len(bigfilearr) - global_lineno):
    line = bigfilearr[lineno+global_lineno]
    smallfile.write(line)
smallfile.close()
