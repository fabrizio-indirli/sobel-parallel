#include <stdio.h>
#include "datastr.h"

void printVector(int * v, int n);

void printHexVector(pixel ** v, int n);

long avgSize(int * ws, int * hs, int num_imgs);

void writeNumToLog(double n, FILE *fOut);

void appendNumToRow(double n, FILE *fOut);

void newRow(FILE *fOut);

void newRowWithFilename(char * s, FILE *fOut);

int selectMPImode(int num_nodes, int num_imgs, long avg_size, int imgs_threshold, int pixels_threshold);