#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
#include "knn.h"
#include "mpi.h"

#define MASTER 0

const double max = 10000.0;
const double min = -10000.0;

void printArray(double *Arr,int m,int n);
double getRandom(double min, double max);
void calcDistances(double *X ,double *Y, double *Xsum, double *Ysum,double *D,int size_x, int size_y, int d);

int main(int argc,char *argv[]){
	int n,d,k,id,numtasks,chunk_size,leftover,size_x,size_y,size_z,tag_split;
	double start,end;
	double *X,*Y,*Z;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Status status;
	
	if(argc != 4){
		printf("In procees %i -> incorrect number of arguments! Recieved (%i) while expected (4)\n",id,(argc-1));
		exit(-1);
	}else{
		n = atoi(argv[1]);
		d = atoi(argv[2]);
		k = atoi(argv[3]);
	}
	
	tag_split = 1;
	chunk_size = (n / numtasks)*d;
	leftover = (n % numtasks)*d;
	
	if(id == MASTER){
		//srand(time(NULL));
		double *Data = (double *)malloc(n * d * sizeof(double));
		for(int i = 0; i < (n*d); ++i){
			Data[i] = ( (double) (rand()) ) / (double) RAND_MAX;//getRandom(min,max);
		}
		MPI_Datatype *indextype = (MPI_Datatype *)malloc((numtasks-1)*sizeof(MPI_Datatype));
		int offset = (chunk_size + leftover);
		
		for(int dest = (MASTER+1); dest < numtasks; dest++){
			MPI_Type_indexed(1, &chunk_size, &offset, MPI_DOUBLE, &indextype[dest-1]);
			MPI_Type_commit(&indextype[dest-1]);
			MPI_Send(Data, 1, indextype[dest-1], dest, tag_split, MPI_COMM_WORLD);
			offset = offset + chunk_size ;
		}		
		
		size_x = chunk_size +  leftover;
		X = (double *)malloc(size_x * sizeof(double));
		Y = (double *)malloc(size_x * sizeof(double));
		Z = (double *)malloc(size_x * sizeof(double));
		if(X == NULL || Y == NULL || Z == NULL){
			exit(-6);
		}
		
		for(int i = 0; i < size_x; ++i){
			X[i] = Data[i];
		}		
		
		for(int i = 0;i < (numtasks - 1); ++i){
			MPI_Type_free(&indextype[i]);
		}
		
		free(indextype);
		free(Data);
	}else{
		int source = MASTER;
		
		size_x = chunk_size;
		X = (double *)malloc((size_x + leftover) * sizeof(double));
		Y = (double *)malloc((size_x + leftover) * sizeof(double));
		Z = (double *)malloc((size_x + leftover) * sizeof(double));
		if(X == NULL || Y == NULL || Z == NULL){
			printf("Process %i fail to allocate memory 1\n",id);
			exit(-6);
		}
		MPI_Recv(X, chunk_size, MPI_DOUBLE, source, tag_split, MPI_COMM_WORLD, &status);
	}
	//========================================================================================================
	printf("%i = %i\n",id,size_x/d);
	MPI_Barrier(MPI_COMM_WORLD);
	if(id == MASTER){
		start = MPI_Wtime();
	}
	
	//copy data
	size_y = size_z = size_x;
	for(int i = 0;i < size_x; ++i){
		Y[i] = X[i];
		Z[i] = X[i];
	}
	
	int prev_process = ((id - 1) >= 0)?(id - 1):(numtasks - 1);
	int next_process = (id + 1)%numtasks;
	int curr_process = id;
	
	int dist_size = size_x / d * n;
	double *D = (double *)malloc(dist_size * sizeof(double));
	double *Dtemp = (double *)malloc((size_x/d)*(n/numtasks + n%numtasks) * sizeof(double));
	double *Xsum = (double *)malloc((size_x + leftover) / d * sizeof(double));
	double *Ysum = (double *)malloc((size_x + leftover) / d * sizeof(double));
	if(Xsum == NULL || Ysum == NULL || D == NULL){
		printf("Process %i fail to allocate memory 2\n",id);
		exit(-5);
	}
	
	int position = (curr_process == MASTER)?(0):((curr_process * chunk_size + leftover)/d);
	calcDistances(X, Y, Xsum, Ysum, Dtemp, (size_x/d), (size_y/d), d);
	int m = size_y/d;
	for(int i = 0;i < (size_x/d); ++i){
		for(int j = 0; j < m; ++j){
			D[i * n + position + j] = Dtemp[i*m + j];
		}
	}
		
	if(id == MASTER)
			printf("Writing at position %i until %i with total size: %i at iteration -1\n",position,position+(size_x/d)*(size_y/d),n * size_x / d);
	
	for(int i = 0;i < (numtasks - 1); ++i){
		curr_process = ((curr_process - 1) >= 0)?(curr_process - 1):(numtasks - 1);
		size_z = (curr_process != MASTER)?(size_z = chunk_size):(size_z = chunk_size + leftover);
		MPI_Sendrecv(Y,size_y, MPI_DOUBLE,next_process, 3,Z, size_z, MPI_DOUBLE,prev_process, 3, MPI_COMM_WORLD, &status);
		//printf("At it %i process %i received chuck %lf with size %i\n",i+1,id,Z[0],size_z);
		
		size_y = size_z;
		for(int i = 0; i < size_y; ++i){
				Y[i] = Z[i];
		}
		
		position = (curr_process == MASTER)?(0):((curr_process * chunk_size + leftover)/d);
		calcDistances(X, Y, Xsum, Ysum, Dtemp, (size_x/d), (size_y/d), d);
		int m = size_y/d;
		for(int i = 0;i < (size_x/d); ++i){
			for(int j = 0; j < m; ++j){
				D[i * n + position + j] = Dtemp[i*m + j];
			}
		}
		

		if(id == MASTER)
			printf("Writing at position %i until %i with total size: %i at iteration %i\n",position,position+(size_x/d)*(size_y/d),n * size_x / d,i);
	}
	
	
	//quickselect the results distances
	for(int i = 0; i < size_x/d; ++i){
		kthSmallest(&D[i*n],0,(n-1),k);
		qsort((void*)&D[i*n], k, sizeof(D[0]), comparator);
	}
	
	
	//MPI_Barrier(MPI_COMM_WORLD);
	
	int tag_combine = 5;
	knnresult *Result;
	
	if(id == MASTER){
		Result = (knnresult *)malloc(sizeof(knnresult));
		Result->ndist = (double *)malloc(n * k * sizeof(double));
		Result->m = n;
		Result->k = k;
		
		for(int i = 0;i < size_x/d; ++i){
			memcpy(&Result->ndist[i*k],&D[i*n],(k * sizeof(double)));
		}
		//read sended element from index var
		//printf("%i %i\n",(n/numtasks),(n%numtasks));
		for(int i = 1;i < numtasks; ++i){
			int pos = i * ((n/numtasks)*k) + (n%numtasks)*k;
			printf("From procees %i received chuck from %i to %i with total %i\n",i,pos,pos+((n/numtasks)*k),n * k);
			MPI_Recv(&Result->ndist[pos], ((n/numtasks)*k), MPI_DOUBLE, i, tag_combine, MPI_COMM_WORLD, &status);
		}
		printf("Result is:\n");
		for(int i = 0; i < n; ++i){
			for(int j = 0; j < k; ++j){
				Result->ndist[i*k + j] = sqrt(fabs(Result->ndist[i*k + j]));
				printf("%lf ,",Result->ndist[i*k + j]);
			}
			printf("\n");
		}
		
	}else{
		//send element in index var
		MPI_Datatype indextype;
		int *blocklengths = (int *)malloc(size_x/d * sizeof(int));
		int *displacements = (int *)malloc(size_x/d * sizeof(int));
		for(int i = 0; i < size_x/d; ++i){
			displacements[i] = i*n;
			blocklengths[i] = k;
		}
		MPI_Type_indexed((size_x/d), blocklengths, displacements, MPI_DOUBLE, &indextype);
		MPI_Type_commit(&indextype);
		MPI_Send(D, 1, indextype, MASTER, tag_combine, MPI_COMM_WORLD);
	}
	
	if(id == MASTER){
		end = MPI_Wtime();
		printf("Master process %i terminated succesfully in %lf!\n",id,(end - start));
	}else{
		printf("Process %i terminated succesfully!\n",id);
	}
	
	MPI_Finalize();
	
	free(Dtemp);
	free(Xsum);
	free(Ysum);
	free(D);
	free(X); 
	free(Y); 
	free(Z);
	
	return 0;
}



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

double getRandom(double min, double max){
    double random = ((float)rand()) / (float)RAND_MAX;
	double diff = max - min;
	double r = random * diff;
	return (min + r);
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

