#include <stdio.h>
#include "datastr.h"

void printVector(int * v, int n);

void printHexVector(pixel ** v, int n);

long avgSize(int * ws, int * hs, int num_imgs);

void writeNumToLog(double n);

void appendNumToRow(double n);

void newRow();

void newRowWithFilename(char * s);

void openLogFile(char * filename);

void closeLogFile();

int selectMPImode(int num_nodes, int num_imgs, long avg_size, int imgs_threshold, int pixels_threshold);