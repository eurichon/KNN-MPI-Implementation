#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const double max = 1000.0;
const double min = 0.0;

double getRandom(double min, double max);

int main(int argc,char *argv[])
{
	
	srand(time(NULL));
	if(argc != 4){
		printf("Incorrect number of arguments! Recieved (%i) while expected (3)\n",(argc-1));
		exit(-1);
	}
	
	int n,m,d;
	
	n = atoi(argv[1]);
	m = atoi(argv[2]);
	d = atoi(argv[3]);

	double *X = (double *)malloc(n * d * sizeof(double));
	double *Y = (double *)malloc(m * d * sizeof(double));
	double *Xsum = (double *)malloc(n * sizeof(double));
	double *Ysum = (double *)malloc(m * sizeof(double));
	double *D = (double *)malloc(n * m * sizeof(double));
	
	printf("Sizes %i %i %i\n",n,m,d);
	
	//fill with data
	if(n >= m){
		for(int i = 0;i < (n*d);++i){
			if(i < (m*d)){
				X[i] = getRandom(min,max);
				//X[i] = i+1;
				Y[i] = getRandom(min,max);
				//Y[i] = i+1;
			}else{
				X[i] = getRandom(min,max);
				//X[i] = i+1;
			}
		}
	}else{
		for(int i = 0;i < (m*d);++i){
			if(i < (n*d)){
				X[i] = getRandom(min,max);
				Y[i] = getRandom(min,max);
			}else{
				Y[i] = getRandom(min,max);
			}
		}
	}
	
	
	//calculate distances between (X,Y) using blas
	struct timespec start, finish;
	double elapsed,elapsed2;
	
	
	clock_gettime(CLOCK_MONOTONIC, &start);

	
	for(int i = 0; i < m ; ++i){
		Ysum[i] = cblas_ddot(d,&Y[i*d],1,&Y[i*d],1);
	}
	//printf("previous D\n");
	for(int i = 0; i < n; i++){
		Xsum[i] = cblas_ddot(d,&X[i*d],1,&X[i*d],1);
		for(int j = 0;j < m ; ++j){
			D[i*m + j] = Xsum[i] + Ysum[j];
			//printf("%lf ,",D[i*m + j]);
		}
		//printf("\n");
	}
	//printf("\n");
	
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,n,m,d,-2,X, d, Y, d,1,D,m);
	
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Blas Time is %lf\n", elapsed);
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	
	int correct = 1;
	
	//printf("D\n");
	for(int i = 0; i < n; ++i){
		for(int j = 0;j < m;++j){
			double sum = 0;
			for(int l = 0; l < d; ++l){
				sum = sum + pow(X[i*d + l] - Y[j * d + l],2);
			}
			//printf("%lf %lf,  ",D[i*m + j],sum);
			
			if(fabs((D[i*m + j]) - (sum)) > 1e-6){
				correct = 0;
			}
		}
		//printf("\n");
	}

	
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed2 = (finish.tv_sec - start.tv_sec);
	elapsed2 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Itearative time is %lf with result %i\n", elapsed2, correct);
	
	
	printf("Total speed up: %lf\n", elapsed2/elapsed);
	printf("========================================\n");
	
	return 0;
}


double getRandom(double min, double max){
    double random = ((float)rand()) / (float)RAND_MAX;
	double diff = max - min;
	double r = random * diff;
	return (min + r);
}