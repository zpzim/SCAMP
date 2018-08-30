#Make sure that this points to the cuda install you want to use
CUDA_DIRECTORY=/usr/local/cuda-9.1
CC=$(CUDA_DIRECTORY)/bin/nvcc
ARCH=-gencode=arch=compute_60,code=sm_70 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_35,code=sm_35
CFLAGS=-c -use_fast_math -lineinfo -O2 -std=c++11 $(ARCH) -I$(CUDA_DIRECTORY)/include

LDFLAGS=-L$(CUDA_DIRECTORY)/lib64 -lcufft
SOURCES=fft_helper.cu tile.cu SCAMP.cu
HEADERS=fft_helper.h tile.h SCAMP.h
OBJECTS=SCAMP.o fft_helper.o tile.o kernels.o
EXECUTABLE=SCAMP

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)  $(SOURCES) $(HEADERS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
SCAMP.o: SCAMP.cu SCAMP.h
	$(CC) $(CFLAGS) SCAMP.cu -o $@

tile.o: tile.h tile.cu
	$(CC) $(CFLAGS) tile.cu -o $@

fft_helper.o: fft_helper.h fft_helper.cu
	$(CC) $(CFLAGS) fft_helper.cu $@

kernels.o: kernels.h kernels.cu
	$(CC) $(CFLAGS) kernels.cu $@


clean:
	rm -f *.o SCAMP
