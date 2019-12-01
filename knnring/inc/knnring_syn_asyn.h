#ifndef KNN_H
#define KNN_H

typedef struct knnresult{
    int * nidx; //!< Indices (0-based) of nearest neighbors [m-by-k]
    double * ndist; //!< Distance of nearest neighbors [m-by-k]
	double max; //!< maximum distance
	double min; //!< minimum distance
    int m; //!< Number of query points [scalar]
    int k; //!< Number of nearest neighbors [scalar]
} knnresult;


typedef struct data{
    double dist;
    int index;
}data;


typedef struct{
	int type;		
	int stage;
	int isFinished;
}Comm;


typedef struct{
	double max;
	double min;
}Limits;


knnresult distrAllkNN(double *X,int n,int d,int k);

void calcDistances(double *X ,double *Y, double *Xsum, double *Ysum,double *D,int size_x, int size_y, int d);

void updateData(data *Data,double *Dist,int curr,int n, int k);

void calcKnn(knnresult *r,data *D, int n ,int k);

//void calcKnnAsyn(knnresult *r,data *D, Limits *l, int n ,int k);

int quickSelect(data *a, int left, int right, int k);

int partitionArray(data *arr, int low, int high);

void swapData(data *a,data *b);

int dataComparator(const void *p, const void *q);

void printArray(double *Arr,int m,int n);

void printResult(knnresult result);

#endif // KNN_H
