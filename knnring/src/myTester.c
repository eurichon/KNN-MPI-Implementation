#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "cblas.h"
#include <time.h>
#include "../inc/knnring_syn_asyn.h"

#define MASTER 0

const double max = 1000.0;
const double min = -1000.0;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
int kthSmallest(data *a, int left, int right, int k);
int partition(data *arr, int low, int high);
void swap(data *a,data *b);
int comparator(const void *p, const void *q);
double getRandom(double min, double max);

int main(int argc,char *argv[]){
	int id, p; 
	double begin, end;
	double *X, *Data;
	
	
	int n = 500;
	int d = 20;
	int k = 50;
	
	if(argc != 4)
		exit(-1);
	n = atoi(argv[1]);
	d = atoi(argv[2]);
	k = atoi(argv[3]);
	
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	//printf("Process %i form %i with total data %i\n",id,p,(n*d));
	
	/* =============================== Spread the data to all the processes ===========================*/
	if(id == MASTER){
		Data = (double *)malloc(p * (n * d) * sizeof(double));
		X = (double *)malloc((n * d) * sizeof(double));
		for(int i =0;i < p * (n * d); i++){
			Data[i] = ( (double) (rand()) ) / (double) RAND_MAX;//getRandom(min,max);
			if(i < (n * d)){
				X[i] = Data[i];
			}
		}
		
		for(int dest = 1; dest < p; dest++){
			//printf("Sending %i package\n",dest);
			MPI_Send(&Data[dest * (n * d)], (n * d), MPI_DOUBLE, dest, 2 , MPI_COMM_WORLD);
		}
		begin = MPI_Wtime();
	}else{
		X = (double *)malloc((n * d) * sizeof(double));
		MPI_Recv(X, (n * d), MPI_DOUBLE, MASTER, 2 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	
	knnresult mpi_res = distrAllkNN(X, n, d, k);


	/* =============================== Validate result ===========================*/
	
	if(id == MASTER){
		double end = MPI_Wtime();
		MPI_Request request[3];
		knnresult total_res;
		total_res.m = n * p;
		total_res.k = k;
		total_res.ndist = (double *)malloc(n * p * k * sizeof(double *));
		total_res.nidx = (int *)malloc(n * p * k * sizeof(int *));
		
		MPI_Gather( mpi_res.ndist, n * k, MPI_DOUBLE, total_res.ndist , n * k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Gather( mpi_res.nidx, n * k, MPI_INT, total_res.nidx , n * k, MPI_INT, MASTER, MPI_COMM_WORLD);
		

		knnresult serial_res = kNN(Data, Data, p*n, p*n, d, k);
		
		int dist_success = 1;
		int index_success = 1;
		
		for(int i = 0; i < p*n*k ; ++i){
			if(fabs(serial_res.ndist[i] - total_res.ndist[i]) > 1e-6){
				dist_success = 0;
			}
		}
		
		for(int i = 0; i < p*n*k ; ++i){
			if(serial_res.nidx[i] != total_res.nidx[i]){
				index_success = 0;
			}
		}
		
		if(dist_success){
			printf("DISTANCES SUCCEEDED\n");
		}else{
			printf("DISTANCES FAILED\n");
		}
		
		if(index_success){
			printf("INDEX SUCCEEDED\n");
		}else{
			printf("INDEX FAILED\n");
		}
		
		free(serial_res.ndist);
		free(serial_res.nidx);
		free(total_res.ndist);
		free(total_res.nidx);
		
		free(Data);
	}else{
		MPI_Gather( mpi_res.ndist, n * k, MPI_DOUBLE, NULL, n * k, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Gather( mpi_res.nidx, n * k, MPI_INT, NULL, n * k, MPI_INT, MASTER, MPI_COMM_WORLD);
		
	}
	
	MPI_Finalize();
	free(mpi_res.ndist);
	free(mpi_res.nidx);
	free(X);
	return 0;
}



double getRandom(double min, double max){
    double random = ((float)rand()) / (float)RAND_MAX;
	double diff = max - min;
	double r = random * diff;
	return (min + r);
}




knnresult kNN(double * X, double * Y, int n, int m, int d, int k){
	struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
	
    knnresult result;
	
    //allocate the needed memory for the result buffers
    result.nidx = (int *)malloc(m * k * sizeof(int));
    result.ndist = (double *)malloc(m * k * sizeof(double));

    // D, Xsum, Ysum are auxiliary tables which help for the fast calculation of the distances
	// The distances are stored then in the table distances alongside their indexes so we cant lose track of them
    data *distances = (data *)malloc(m * n * sizeof(data));
	double *D = (double *)malloc(m * n * sizeof(double));
	double *Xsum = (double *)malloc(n * sizeof(double));
	double *Ysum = (double *)malloc(m * sizeof(double));
	
	
	// calculate distances
	calcDistances(Y ,X, Ysum, Xsum, D, m, n, d);
	// copy distances and add the right indexes accordingly
	for(int i = 0;i < m; ++i){
		for(int j = 0; j < n; ++j){
			distances[i * n + j].dist = D[i * n + j];
            distances[i * n + j].index = j;
		}
	}
	
	// release auxiliary tables
	free(D);
	free(Xsum);
	free(Ysum);


	// For every query point calculate the k nearest neighbor 
	// and perform quick-sort in the first k positions of each row
	// then copy store the results
    for(int i = 0; i < m; i++){	
        if(!kthSmallest(&distances[i*n],0,n-1,k)){
                printf("Something went wrong!Exiting..\n");
            exit(-5);
        }
		
        qsort((void*)&distances[i*n],  k, sizeof(distances[0]), comparator);

        for(int j = 0; j < k ; ++j){
            result.ndist[i*k + j] = sqrt(fabs(distances[i*n + j].dist));
            result.nidx[i*k + j] = distances[i*n + j].index;
        }
    }
	
	free(distances);
    result.k = k;
    result.m = m;
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Blas Time is %lf\n", elapsed);
	
    return result;
}


/*
	Performs vectorized X*Y' with ROWMAJOR at least x70 times faster
	Y stores the sum(Y.^2) and X stores the sum(X.^2) as "col" and "row" respectivly so they are calculated only once
	and then it adds them with the mixed produxt x*y	
	It returns the <<squared>> values of the actual distances 

void calcDistances(double *X ,double *Y, double *Xsum, double *Ysum,double *D,int size_x, int size_y, int d){
	for(int i = 0; i < size_y ; ++i){
		Ysum[i] = cblas_ddot(d,&Y[i*d],1,&Y[i*d],1);
	}
	
	for(int i = 0; i < size_x; i++){
		Xsum[i] = cblas_ddot(d,&X[i*d],1,&X[i*d],1);
		for(int j = 0;j < size_y; j++){
			D[i*size_y + j] = Xsum[i] + Ysum[j];
		}
	}
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, size_x, size_y, d, -2, X, d, Y, d, 1, D, size_y);
}
*/

/*
	classic iterative implementation of quickselect
*/
int kthSmallest(data *a, int left, int right, int k)
{
    while (left <= right) {
        int pivotIndex = partition(a, left, right);

        if (pivotIndex == k - 1)
            return 1;
        else if (pivotIndex > k - 1)
            right = pivotIndex - 1;
        else
            left = pivotIndex + 1;
    }
    return 0;
}


/*
	auxilary function which performs partition in the the row arr with respect to the .dist of the struct element
*/
int partition(data *arr, int low, int high)
{
    double pivot = arr[high].dist;
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j].dist <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}


/*
	swaps two struct of type data
*/
void swap(data *a,data *b){
    data temp = *a;
    *a = *b;
    *b = temp;
}


/*
	performs comparison between two structs of type data with respect to the .dist value
	which is essential so we keep track of both distance and its index as we performs 
	quckselect and quicksort in the while struct
*/
int comparator(const void *p, const void *q)
{
    data l = *(const data *)p;
    data r = *(const data *)q;
    if(l.dist > r.dist){
        return 1;
    }else if(l.dist < r.dist){
        return -1;
    }else{
        return 0;;
    }
}

