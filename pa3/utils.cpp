/*
 * Utilities for the Aliev-Panfilov code
 *
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD, 11/2/2015
 */

#include <cstdlib>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <assert.h>
#include <iomanip>
#include <string>
#include <math.h>
#include "apf.h"

using namespace std;

//
// Report statistics about the computation: the sums of the squares
// and the max value of the mesh
// These values should not vary (except to within roundoff)
// when we use different numbers of processes to solve the problem

// We use the sum of squares to compute the L2 norm, which is a normalized
// square root of this sum; see L2Norm() in Helper.cpp

void stats(double *E, int m, int n, double *_mx, double *sumSq)
{
    double mx = -1;
    double _sumSq = 0;
    int i, j;

    for (i = 0; i < (m + 2) * (n + 2); i++)
    {
        int rowIndex = i / (n + 2); // gives the current row number in 2D array representation
        int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

        if (colIndex == 0 || colIndex == (n + 1) || rowIndex == 0 || rowIndex == (m + 1))
            continue;

        _sumSq += E[i] * E[i];
        double fe = fabs(E[i]);
        if (fe > mx)
            mx = fe;
    }
    *_mx = mx;
    *sumSq = _sumSq;
}
