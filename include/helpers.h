#include <stdio.h>

void printVector(int * v, int n);

void printHexVector(pixel ** v, int n);

int avgSize(int * ws, int * hs, int num_imgs);

void writeNumToLog(double n, FILE *fOut);

void appendNumToRow(double n, FILE *fOut);

void newRow(FILE *fOut);

void newRowWithFilename(char * s, FILE *fOut);