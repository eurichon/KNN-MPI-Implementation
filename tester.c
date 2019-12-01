/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"

#include "tester_helper.h"

int main()
{
  int n=100;                    // corpus
  int m=100;                    // query 
  int d=30;                      // dimensions
  int k=50;                     // # neighbors

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );
  double  * query  = (double * ) malloc( m*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
    corpus[i] = ( (double) (rand()) ) / (double) RAND_MAX;

  for (int i=0;i<m*d;i++)
    query[i]  = ( (double) (rand()) ) / (double) RAND_MAX;

  knnresult knnres = kNN( corpus, query, n, m, d, k );

  int isValidC = validateResult( knnres, corpus, query, n, m, d, k, COLMAJOR );

  int isValidR = validateResult( knnres, corpus, query, n, m, d, k, ROWMAJOR );
  
  printf("Tester validation: %s NEIGHBORS\n",
         STR_CORRECT_WRONG[isValidC||isValidR]);

  free( corpus );
  free( query );

  return 0;
  
}