SRC_DIR=src
HEADER_DIR=./include
OBJ_DIR=obj
MPI_INSTALL=/users/profs/2017/francois.trahay/soft/install/openmpi
MPI_INCS=$(MPI_INSTALL)/include
MPI_LIB=$(MPI_INSTALL)/lib64
CUDA_LIB=/usr/local/cuda/lib64

CC=mpicc
NCC=nvcc
CFLAGS=-O3 -I$(HEADER_DIR) -fopenmp
NCCFLAGS=-I$(MPI_INCS) -I$(HEADER_DIR)
LDFLAGS=-lm -lcudart -L$(CUDA_LIB)


SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	main.c \
	openbsd-reallocarray.c \
	quantize.c \
	grey_filter.c \
	sobel_filter.cu \
	load_pixels.c \
	blur_filter.c \
	store_pixels.c \
	helpers.c \
	mpi_mode_1.c \
	mpi_mode_2.c \
	mpi_mode_3.c \
	mpi_mode_0.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o \
	$(OBJ_DIR)/grey_filter.o \
	$(OBJ_DIR)/sobel_filter.o \
	$(OBJ_DIR)/load_pixels.o \
	$(OBJ_DIR)/blur_filter.o \
	$(OBJ_DIR)/store_pixels.o \
	$(OBJ_DIR)/helpers.o \
	$(OBJ_DIR)/mpi_mode_1.o \
	$(OBJ_DIR)/mpi_mode_2.o \
	$(OBJ_DIR)/mpi_mode_3.o \
	$(OBJ_DIR)/mpi_mode_0.o \

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NCC) $(NCCFLAGS) $(LDFLAGS) -c -o $@ $^

sobelf:$(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f sobelf $(OBJ)
