CUDA_DIR = /usr/local/cuda

CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_ARCH_FLAGS := -arch=sm_75
#CC_FLAGS += $(CUDA_ARCH_FLAGS) -I. -g -G -Xptxas -dlcm=cg
CC_FLAGS += $(CUDA_ARCH_FLAGS) -I. -O3
#CC_FLAGS += $(CUDA_ARCH_FLAGS) -I.

CC := $(CUDA_DIR)/bin/nvcc

OBJ = snappy_cuda.o snappy_compress.o snappy_decompress.o

all: snappy_cuda

snappy_cuda : $(OBJ)
	$(CC) $(OBJ) $(CC_FLAGS) -o $@
	./gen_cscope.sh
	
snappy_cuda.o: snappy_cuda.cu snappy_cuda.h
	$(CC) -c  $< $(CC_FLAGS)
snappy_compress.o: snappy_compress.cu snappy_compress.h
	$(CC) -c  $< $(CC_FLAGS)
snappy_decompress.o: snappy_decompress.cu snappy_decompress.h
	$(CC) -c  $< $(CC_FLAGS)

clean: 
	rm -f snappy_cuda *.o
