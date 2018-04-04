#Make sure that this points to the cuda install you want to use
CUDA_DIRECTORY=/usr/local/cuda-9.1
CC=$(CUDA_DIRECTORY)/bin/nvcc
ARCH=-gencode=arch=compute_60,code=sm_60
CFLAGS=-c -dc -use_fast_math -lineinfo -O2 -std=c++11 $(ARCH) -I$(CUDA_DIRECTORY)/include

LDFLAGS=-L$(CUDA_DIRECTORY)/lib64 -lcufft
SOURCES=fft_helper.cu tile.cu SCRIMP.cu
HEADERS=fft_helper.h tile.h SCRIMP.h
OBJECTS=SCRIMP.o
EXECUTABLE=SCRIMP-GPU

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)  $(SOURCES) $(HEADERS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
SCRIMP.o: SCRIMP.cu SCRIMP.h
	$(CC) $(CFLAGS) SCRIMP.cu -o $@

#tile.o: tile.h tile.cu
#	$(CC) $(CFLAGS) tile.cu -o $@

fft_helper.o: fft_helper.h fft_helper.cu
	$(CC) $(CFLAGS) fft_helper.cu $@

clean:
	rm -f *.o SCRIMP-GPU
