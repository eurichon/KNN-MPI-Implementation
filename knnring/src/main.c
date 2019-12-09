#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../inc/knnring.h"

const double max = 1000.0;
const double min = -1000.0;

void printArray(double *Arr,int m,int n);
double getRandom(double min, double max);

int main(int argc,char *argv[]){	
    int n = 20; //data points
    int m = 20; //query points
    int d = 10; //dimensions
    int k = 2; //neighbors
	
	if (argc != 5)
		exit(-1);

	n = atoi(argv[1]);
	m = atoi(argv[2]);
	d = atoi(argv[3]);
	k = atoi(argv[4]);

    struct timespec start, finish;
	double elapsed;
    //srand(time(NULL));
	
    //create a dataset of n points of d dimensions
    double *X = (double *)malloc(n * d * sizeof(double));
    for(int i = 0; i < (n * d); ++i){
        X[i] =  ( (double) (rand()) ) / (double) RAND_MAX;
    }
	
    //select the m first of the dataset as query points
    double *Y = (double *)malloc(m * d * sizeof(double));
    for(int i = 0; i < (m * d); ++i){
        Y[i] =  X[i];//( (double) (rand()) ) / (double) RAND_MAX;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
	
    knnresult result = kNN(X,Y,n,m,d,k);
	printResult(result);

    clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Time is %lf\n", elapsed);

    return 0;
}

void printArray(double *Arr,int m,int n){
	printf("=======Array is===========\n");
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			printf("%lf ,",Arr[i*n + j]);
		}
		printf("\n");
	}	
}

double getRandom(double min, double max){
    double random = ((float)rand()) / (float)RAND_MAX;
	double diff = max - min;
	double r = random * diff;
	return (min + r);
}
