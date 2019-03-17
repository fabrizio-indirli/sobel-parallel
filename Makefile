SRC_DIR=src
HEADER_DIR=include
CUDA_DIR=/usr/local/cuda-10.0/include,/usr/local/cuda-10.0/lib64,/usr/local/cuda-10.0
OBJ_DIR=obj

CC=gcc
NCC=nvcc
CFLAGS=-O3 -I$(HEADER_DIR)
NCCFLAGS=$(CFLAGS) -I$(CUDA_DIR)
LDFLAGS=-lm

SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	main.cu \
	openbsd-reallocarray.c \
	quantize.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(SRC_DIR)/main.cu \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

sobelf:$(OBJ)
	$(NCC) $(NCCFLAGS) $(LDFLAGS) -o $@ $^

clean:
	
