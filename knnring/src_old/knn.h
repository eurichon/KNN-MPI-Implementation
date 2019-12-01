#ifndef KNN_H
#define KNN_H

typedef struct knnresult{
    int * nidx; //!< Indices (0-based) of nearest neighbors [m-by-k]
    double * ndist; //!< Distance of nearest neighbors [m-by-k]
    int m; //!< Number of query points [scalar]
    int k; //!< Number of nearest neighbors [scalar]
} knnresult;

typedef struct data{
    double dist;
    int index;
}data;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

double calculateDistance(double *p1, double *p2, int dim);

int comparator(const void *p, const void *q);

int partition(double *arr, int low, int high);

void swap(double *a,double *b);

int kthSmallest(double *a, int left, int right, int k);


#endif // KNN_H
