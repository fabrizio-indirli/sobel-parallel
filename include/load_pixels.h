#include <gif_lib.h>
#include "datastr.h"
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

animated_gif * load_pixels( char * filename );

int 
output_modified_read_gif( char * filename, GifFileType * g );