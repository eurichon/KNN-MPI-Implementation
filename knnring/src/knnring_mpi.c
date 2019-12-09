#include "../inc/knnring_syn_asyn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define MASTER 0


knnresult distrAllkNN(double *X,int n,int d,int k){
	struct timespec start, finish;
	double elapsed;
	knnresult result;
	data *Data;
	double *Y, *Z, *Dist, *Xsum, *Ysum;
	int id, p;
	int *Indexes;
	
	
	
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	
	//printf("Hello from process %i\n",id);
	
	Y = (double *)malloc(n * d * sizeof(double));		// The rotated corpus set from i process
	Z = (double *)malloc(n * d * sizeof(double));		// Temp corpus set so we can send and receice at the same time
	Xsum = (double *)malloc(n * sizeof(double));		// helping array for a fast calculation of the distances where sum(X.^2) are stored
	Ysum = (double *)malloc(n * sizeof(double)); 		// helping array for a fast calculation of the distances where sum(Y.^2) are stored
	Dist = (double *)malloc(n * n * sizeof(double));	// helping array to keep the calculated distances
	Data = (data *)malloc(n * (n + k) * sizeof(data));  // Array which holds for each point the last knn's + the new incoming corpus Y
	
	
	result.nidx = (int *)malloc(n * k * sizeof(int));
	result.ndist = (double *)malloc(n * k * sizeof(double));
	result.m = n;
	result.k = k;
	
	
	// fill the first k columns with huge numbers for initialization purposes
	// these values will be rejects by quickselect in the first call of calcKnn
	// but there are essential otherwise all the result values will be 0
	for(int i = 0; i < n; i++){
		for(int j = 0; j < k; j++){
			Data[i * (n + k) + j].dist = 100000.0;
			Data[i * (n + k) + j].index = -1;
		}
	}
	
	
	int previous = (id - 1 + p) % p;	// previous neighboor
	int next = (id + 1) % p;			// next neighboor
	int curr = id;						// process from which we currently receive data
	int tag = 3;
	MPI_Status status;
	
	// Iterative rotation of data and update of knn 

	memcpy(Y, X, (n * d) * sizeof(double));			// copies X to Y so we can first calculate X*X'
	calcDistances(X, Y, Xsum, Ysum, Dist, n, n, d); // performs the distance calculation and puts the result to Dist table
	updateData(Data, Dist, curr, n, k);				// copies the elements of Dist in the rights position to Data and adds for each an id
	calcKnn(&result,Data,n, k);						// performs quickselect and qsort for each point and sotres it to result
	elapsed = 0.0;
	
	for(int i = 0; i < (p - 1); i++){
		curr = (curr - 1 + p) % p; //finds from which actual process we currently receive data
		clock_gettime(CLOCK_MONOTONIC, &start);
		MPI_Sendrecv(Y,(n * d), MPI_DOUBLE, next, tag, Z, (n * d), MPI_DOUBLE,previous, tag, MPI_COMM_WORLD, &status);	// Blocking method for simu; send/rec
		clock_gettime(CLOCK_MONOTONIC, &finish);
		elapsed += (finish.tv_sec - start.tv_sec);
		elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
		
		memcpy(Y, Z, (n * d) * sizeof(double));
		calcDistances(X, Y, Xsum, Ysum, Dist, n, n, d);
		updateData(Data, Dist, curr, n, k);
		calcKnn(&result, Data, n, k);
	}
	
	
	if(id == MASTER)
		printf("Communication time is: %lf\n",elapsed);
	
	//if(id == MASTER)
	//	printResult(result);
	
	return result;
}


/*
	for each element of X we store a row which contains the k last nearest neighboor followed by the n distances of the current set Y
	the first time because we initialize the k elements of each row with huge numbers quickselect will reject those values and will actually 
	put the knn's of the X with itself. On later calls it will perform quickselect between the last knn's and the new neighboor's that have arrived.
	it also combined the neighboor's with their indexes which are the id of the process we are receiving from multiplied by the n-number of elements 
	in each set plus the j current index of this set(shifted so the indexes will match)
*/
void updateData(data *Data,double *Dist,int curr,int n, int k){
	for(int i = 0; i < n; i++){
		for(int j = k; j < (k + n); j++){
			Data[i * (n + k) + j].dist = Dist[i * n + (j - k)];
			Data[i * (n + k) + j].index = (curr * n + (j - k));
		}
	}
}


/*
	Performs vectorized X*Y' with ROWMAJOR at least x70 times faster
	Y stores the sum(Y.^2) and X stores the sum(X.^2) as "col" and "row" respectivly so they are calculated only once
	and then it adds them with the mixed produxt x*y	
	It returns the <<squared>> values of the actual distances 
*/
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


/*
	Performs quickselect between the previous stored Knn's and the new set pushing the new knn's in the first k places of each row
	Soon after it performs quicksort in the k first places of each row so as to sort these new knn's
	Lastly it copies the values to the result and applying sqrt and fabs 
*/
void calcKnn(knnresult *r,data *D, int n ,int k){
	for(int i = 0; i < n; i++){
		quickSelect(&D[i * (n + k)], 0, (n + k - 1), k);
		qsort((void*)&D[i * (n + k)], k, sizeof(D[0]), dataComparator);
		for(int j = 0; j < k; j++){
			r->ndist[i * k + j] = sqrt(fabs(D[i * (n + k) + j].dist));
			r->nidx[i * k + j] = D[i * (n + k) + j].index;
		}
	}
}


/*
	auxilary function which performs partition in the the row arr with respect to the .dist of the struct element
*/
int partitionArray(data *arr, int low, int high) 
{ 
    double pivot = arr[high].dist; 
    int i = (low - 1); 
    for (int j = low; j <= high - 1; j++) { 
        if (arr[j].dist <= pivot) { 
            i++; 
            swapData(&arr[i], &arr[j]); 
        } 
    } 
    swapData(&arr[i + 1], &arr[high]); 
    return (i + 1); 
} 

  
/*
	classic iterative implementation of quickselect
*/
int quickSelect(data *a , int left, int right, int k) 
{ 
    while (left <= right) {  
        int pivotIndex = partitionArray(a, left, right); 

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
	swaps two struct of type data
*/
void swapData(data *a,data *b){
    data temp = *a;
    *a = *b;
    *b = temp;
}


/*
	performs comparison between two structs of type data with respect to the .dist value
	which is essential so we keep track of both distance and its index as we performs 
	quckselect and quicksort in the while struct
*/
int dataComparator(const void *p, const void *q)
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


/*
	auxilary function which prints a rowmajor array of m rows and n cols
*/
void printArray(double *Arr,int m,int n){
	printf("=======Array is===========\n");
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			printf("%lf ,",Arr[i*n + j]);
		}
		printf("\n");
	}	
}


/*
	auxilary function prints the result table for debugging purposes
*/
void printResult(knnresult result){
	printf("Result is:\n");
	for(int i = 0; i < result.m; ++i){
		for(int j = 0; j < result.k; ++j){
			printf("%lf ,",result.ndist[i*result.k + j]);
		}
		printf("\n");
	}
}