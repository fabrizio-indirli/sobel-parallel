#include <gif_lib.h>
#include "datastr.h"
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"{
#endif 
    animated_gif * load_pixels( char * filename );
#ifdef __cplusplus
}
#endif