#include "../inc/knnring_syn_asyn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define MASTER 0
	
/*
	This struct contains all the necessary variables so we can call MPI_Isend & MPI_Irecv
*/
typedef struct{
	double *p1;
	double *p2;
	int size;
	int next;
	int previous;
	int tag;
	MPI_Request reqs;
	MPI_Status status;	
}Package;

void calcKnnAsyn(knnresult *r,data *D, Limits *l,Comm *comm, int *flag, Package *pack, int n ,int k);
void rotateData(Comm *comm, int *flag, Package *pack);
void myProd( Limits *in, Limits *inout, int *len, MPI_Datatype *dptr );
void calcMax(Limits *l,double *Dist, int n);

knnresult distrAllkNN(double *X,int n,int d,int k){
	knnresult result;
	struct timespec start, finish;
	double elapsed;
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
	int isOdd = id % 2;					// if is odd then (id % 2) = 1~True else if is even (id % 2) = 0~False
	int flag = 0;
	int tag = 3;	
		
	// Contains data needed for the MPI_Isend and MPI_Irecv routines
	Package pack;
	pack.p1 = Y;
	pack.p2 = Z;
	pack.size = (n * d);
	pack.tag = 4;
	pack.next = next;
	pack.previous = previous;
	
	// Contains the needed flags to control and coordinate the data exchange during claculations
	Comm comm;
	comm.type = (id % 2);
	comm.isFinished = 0;
	comm.stage = 0;
	
	
	Limits l, res_l;
	l.max = -10000.0;
	l.min = 10000.0;
	
	memcpy(Y, X, (n * d) * sizeof(double));			// copies X to Y so we can first calculate X*X'
	MPI_Request req;
	MPI_Status status;
	
	elapsed = 0.0;
	
	/* we can call it also in iterations of updateData and calcDistances if the size of the set becomes big enough*/
	rotateData(&comm, &flag ,&pack);
		
	calcDistances(X, Y, Xsum, Ysum, Dist, n, n, d); // performs the distance calculation and puts the result to Dist table
	calcMax(&l,Dist, n);
	updateData(Data, Dist, curr, n, k);				// copies the elements of Dist in the rights position to Data and adds for each an id
	calcKnnAsyn(&result,Data,&l,&comm, &flag ,&pack,n, k);						// performs quickselect and qsort for each point and sotres it to result
	
	/*  If the communication takes enough time it will surpass the calculation time above and the look below loop will wait untill
		it completes. The time it waits is the actual communication time in the asynchronous mpi where all the calculations have completed
		and the process waits for the data transfer. If the communication has been finished before we reached the loop then that means
		it was completly hidden by the calculations! The comm isFinished will be true and the loop will instantly break
	*/
	clock_gettime(CLOCK_MONOTONIC, &start);
	while(!comm.isFinished){
		rotateData(&comm, &flag ,&pack);
	}
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = elapsed + (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
	for(int i = 0; i < (p - 1); i++){ //(p - 1)
		curr = (curr - 1 + p) % p; //finds from which actual process we currently receive data
		
		memcpy(Y, Z, (n * d) * sizeof(double));
		
		comm.isFinished = 0;
		rotateData(&comm, &flag,  &pack);
		
		calcDistances(X, Y, Xsum, Ysum, Dist, n, n, d);
		calcMax(&l,Dist, n);
		updateData(Data, Dist, curr, n, k);
		calcKnnAsyn(&result, Data, &l, &comm, &flag ,&pack, n, k);
		
		clock_gettime(CLOCK_MONOTONIC, &start);
		while(!comm.isFinished){
			rotateData(&comm, &flag ,&pack);
		}
		clock_gettime(CLOCK_MONOTONIC, &finish);
		elapsed = elapsed + (finish.tv_sec - start.tv_sec);
		elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
		
	}
	if(id == MASTER)
			printf("Communication Time is %lf\n", elapsed);
	MPI_Op myOp; 
	MPI_Datatype ctype; 
	MPI_Type_contiguous( 2, MPI_DOUBLE, &ctype ); 
    MPI_Type_commit( &ctype ); 
	
	MPI_Op_create((void *)myProd, 1, &myOp );
	//printf("Before max %lf min %lf\n",l.max, l.min);	
    MPI_Allreduce( &l, &res_l, 1, ctype, myOp, MPI_COMM_WORLD ); 

	//printf("Process %i received max=%lf, min=%lf\n",id, res_l.max, res_l.min);
	
	result.max = res_l.max;
	result.min = res_l.min;
	
	if(id == MASTER) 
		printf("Global max=%lf & min=%lf\n",res_l.max, res_l.min);
	
	
	
	//if(id == MASTER)
	//	printResult(result);
	
	return result;
}


/*
	This is a custom function that will called inside the reduction process by mpi
	and will calculate the max and min contained in the struct Limits simultaneously
*/
void myProd(Limits *in, Limits *inout, int *len, MPI_Datatype *dptr ) 
{ 
    for (int i=0; i< *len; ++i) { 
		if((in->max) > (inout->max)){
			inout->max = in->max;
		}
		if((in->min) < (inout->min)){
			inout->min = in->min;
		}
    } 
} 


/*
	If an exchange of data should start this function checks in which stage we are currently
	and either makes the Odd processes to trasmite to the Even or vice versa or waits for either of them to finish
	to the ext stage by performing an MPI_Test
	Upon finishing it resets all the flags and stages and waits for the flag isFinished to be nulled outside of its body
	The var type contains the info if the process is even of odd
*/
void rotateData(Comm *comm, int *flag ,Package *pack){
	if(comm->isFinished){
		return;
	}else{
		if(comm->stage == 0){
			if(comm->type){
				MPI_Isend(pack->p1, pack->size, MPI_DOUBLE, pack->next, pack->tag, MPI_COMM_WORLD, &(pack->reqs));
			}else{
				MPI_Irecv(pack->p2, pack->size, MPI_DOUBLE, pack->previous, pack->tag, MPI_COMM_WORLD, &(pack->reqs));
			}
			comm->stage++;
		}else if(comm->stage == 1){
			MPI_Test(&(pack->reqs), flag, &(pack->status));
			if(*flag){
				comm->stage++;
				*flag = 0;
			}else{
				return;
			}
		}else if(comm->stage == 2){
			if(!comm->type){
				MPI_Isend(pack->p1, pack->size, MPI_DOUBLE, pack->next, pack->tag, MPI_COMM_WORLD, &(pack->reqs));
			}else{
				MPI_Irecv(pack->p2, pack->size, MPI_DOUBLE, pack->previous, pack->tag, MPI_COMM_WORLD, &(pack->reqs));
			}
			comm->stage++;
		}else if(comm->stage == 3){
			MPI_Test(&(pack->reqs), flag, &(pack->status));
			if(*flag){
				comm->stage++;
				*flag = 0;
			}else{
				return;
			}
		}else{
			comm->stage = 0;
			*flag = 0;
			comm->isFinished = 1;
		}
	}
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



void calcMax(Limits *l,double *Dist, int n){
	for(int i = 0;i < n;i++){
		for(int j = 0;j < n;j++){
			l->max = (Dist[i * n + j] > l->max)?(Dist[i * n + j]):(l->max);
		}		
	}
	
}



/*
	Performs quickselect between the previous stored Knn's and the new set pushing the new knn's in the first k places of each row
	Soon after it performs quicksort in the k first places of each row so as to sort these new knn's
	Lastly it copies the values to the result and applying sqrt and fabs and calculate global min & max
*/
void calcKnnAsyn(knnresult *r,data *D, Limits *l,Comm *comm, int *flag, Package *pack, int n ,int k){
	for(int i = 0; i < n; i++){		
		quickSelect(&D[i * (n + k)], 0, (n + k - 1), k);
		qsort((void*)&D[i * (n + k)], k, sizeof(D[0]), dataComparator);
		for(int j = 0; j < k; j++){
			r->ndist[i * k + j] = sqrt(fabs(D[i * (n + k) + j].dist));
			r->nidx[i * k + j] = D[i * (n + k) + j].index;
			if(j != 0 && l->min > r->ndist[i * k + j]){
				l->min = r->ndist[i * k + j];
			}
		}
		
		
		rotateData(comm, flag ,pack);
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
	auxilary function prints the time interval between start and finish
*/
void printTime(struct timespec start,struct timespec finish){
	double elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Blas Time is %lf\n", elapsed);
	clock_gettime(CLOCK_MONOTONIC, &start);
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