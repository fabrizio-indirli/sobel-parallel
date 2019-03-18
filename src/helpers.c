#include "helpers.h"
#include <stdio.h>

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
int avgSize(int * ws, int * hs, int num_imgs){
    long totSize = 0;
    for(i=0; i<num_imgs; i++){
        totSize += ws[i] * hs[i]
    }
    return totSize / num_imgs;
}



// appends num to the current line and then goes to new line
void writeNumToLog(double n, FILE *fOut){
    fprintf(fOut, "%lf\n", n);
}

// appends num to the current line
void appendNumToRow(double n, FILE *fOut){
    fprintf(fOut, "%lf,",n);
}

// goes to new line in the log
void newRow(FILE *fOut){
    fprintf(fOut, "\n");
}

// goes to new line in the log and appends the name of the GIF file
void newRowWithFilename(char * s, FILE *fOut){
    fprintf(fOut, "\n%s,", s);
}