SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS=-O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS=-lm

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
	sobel_filter.c \
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

sobelf:$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ)
