#include "helpers.h"
#include <stdio.h>

FILE *fOut;

void printVector(int * v, int n){
    int i;
    printf("[");
    for(i=0; i<n; i++){
        printf(" %d ", v[i]);
    }
    printf("]\n");
}

void printHexVector(pixel ** v, int n){
    int i;
    printf("[");
    for(i=0; i<n; i++){
        printf(" %#x ", v[i]);
    }
    printf("]\n");
}

// returns the average size (in pixels) of the subimages
long avgSize(int * ws, int * hs, int num_imgs){
    long totSize = 0;
    int i;

    for(i=0; i<num_imgs; i++){
        totSize += ws[i] * hs[i];
    }
    return totSize / (long)num_imgs;
}


// appends num to the current line and then goes to new line
void writeNumToLog(double n){
    fprintf(fOut, "%lf\n", n);
}

// appends num to the current line
void appendNumToRow(double n){
    fprintf(fOut, "%lf,",n);
}

// goes to new line in the log
void newRow(){
    fprintf(fOut, "\n");
}

// goes to new line in the log and appends the name of the GIF file
void newRowWithFilename(char * s){
    fprintf(fOut, "\n%s,", s);
}

// opens log file to write into it
void openLogFile(char * filename){
    fOut = fopen(filename,"a");
    if(ftell(fOut)==0) //file is empty
        fprintf(fOut, "filename,n_subimgs,width,height,import_time,filters_time,export_time,");
}

// close log file
void closeLogFile(){
    fclose(fOut);
}

// chooses how to use the MPI ranks to process the input.
// the output is: 0: no MPI; 1: MPI on subimgs;  2: MPI on pixels;  3: hybrid
int selectMPImode(int num_nodes, int num_imgs, long avg_size, int imgs_threshold, int pixels_threshold){

    // DECISION TREE FOR MPI MODE //
        if(num_nodes > 1){
            // use MPI

            if(num_imgs > 1){
                // it's possible to parallelize on images

                if(avg_size > imgs_threshold)
                {
                    // use MPI on different images

                    if(num_nodes > 2*num_imgs && avg_size > pixels_threshold){
                        // TODO: use MPI also on pixels
                        return 1;
                    }
                    else return 1;

                } else {
                    // no images parallelization

                    if(avg_size > pixels_threshold){
                        // use MPI on pixels
                        return 2;

                    } else return 0; // no MPI (images too small)

                }

            } else {
                // only 1 image, no images parallelization

                    if(avg_size > pixels_threshold){
                        // use MPI on pixels
                        return 2;

                    } else return 0; // no MPI (images too small)
            }

        } else {
            // no MPI because there is only 1 rank
            return 0;
        }
}