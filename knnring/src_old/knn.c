#include "knn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


double calculateDistance(double *p1, double *p2, int dim){
    double sum = 0;
    for(int i = 0;i < dim;i++){
        sum = sum + pow(p1[i] - p2[i],2);
    }
    return sqrt(sum);
}

int comparator(const void *p, const void *q)
{
    double l = *(const double *)p;
    double r = *(const double *)q;
    if(l > r){
        return 1;
    }else if(l < r){
        return -1;
    }else{
        return 0;;
    }
}

int partition(double *arr, int low, int high)
{
    double pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void swap(double *a,double *b){
    double temp = *a;
    *a = *b;
    *b = temp;
}

int kthSmallest(double *a, int left, int right, int k)
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
