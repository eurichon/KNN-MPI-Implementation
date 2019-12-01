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


knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

void calcDistances(double *X ,double *Y, double *Xsum, double *Ysum,double *D,int size_x, int size_y, int d);

int kthSmallest(data *a, int left, int right, int k);

int partition(data *arr, int low, int high);

void swap(data *a,data *b);

int comparator(const void *p, const void *q);

void printResult(knnresult result);

#endif // KNN_H
