/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#include <cstring>

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);
double *alloc1D(int m, int n);

extern control_block cb;
static int TAG_E_PREV_INIT = 0;
static int TAG_R_INIT = 1;

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init(double *E, double *E_prev, double *R, int m, int n)
{
    // query the rank
    int my_rank = 0;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int my_rank_x = my_rank % cb.px;
    int my_rank_y = my_rank / cb.px;
    // compute sub-matrix size
    int quotient_n = n / cb.px,
        quotient_m = m / cb.py,
        reminder_n = n % (cb.px),
        reminder_m = m % (cb.py);
#endif

    // only init when self is rank-0 process
    if (my_rank == 0)
    {
        int i;

        for (i = 0; i < (m + 2) * (n + 2); i++)
            E_prev[i] = R[i] = 0;

        for (i = (n + 2); i < (m + 1) * (n + 2); i++)
        {
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
                continue;

            E_prev[i] = 1.0;
        }

        for (i = 0; i < (m + 2) * (n + 2); i++)
        {
            int rowIndex = i / (n + 2); // gives the current row number in 2D array representation
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
                continue;

            R[i] = 1.0;
        }
        // We only print the meshes if they are small enough
#if VERBOSE_PRINT
        cout << "0: inited" << endl;
        printMat("Init Matrix E_prev", E_prev, m + 2, n + 2);
        printMat("Init Matrix R", R, m + 2, n + 2);
#endif

#ifdef _MPI_
        // send to others
        double *E_prev_init_buffer = alloc1D(quotient_m + 3, quotient_n + 3);
        double *R_init_buffer = alloc1D(quotient_m + 3, quotient_n + 3);
        // distribute init values to peer processes
        int m_offset, n_offset;
        m_offset = 0;
        for (int receive_rank_y = 0; receive_rank_y < cb.py; receive_rank_y++)
        {
            int sub_m = (receive_rank_y < reminder_m) ? (quotient_m + 1) : (quotient_m); // compute the sub matrix size in y direction;

            n_offset = 0;
            for (int receive_rank_x = 0; receive_rank_x < cb.px; receive_rank_x++)
            {
                int sub_n = (receive_rank_x < reminder_n) ? (quotient_n + 1) : (quotient_n); // compute the sub matrix size in x directiond

                if (receive_rank_x != 0 || receive_rank_y != 0) // skip sending to self
                {
                    // pack the desired section
                    for (int mm = 0; mm < sub_m + 2; mm++)
                    {
                        for (int nn = 0; nn < sub_n + 2; nn++)
                        {
                            int local_idx = mm * (sub_n + 2) + nn;
                            int global_idx = (m_offset + mm) * (cb.m + 2) + (n_offset + nn);
                            E_prev_init_buffer[local_idx] = E_prev[global_idx];
                            R_init_buffer[local_idx] = R[global_idx];
                        }
                    }
                    // send init values
                    int receive_rank = receive_rank_y * cb.px + receive_rank_x;
#if VERBOSE_PRINT
                    cout << "0: sending init to " << receive_rank << ", offset:" << n_offset << m_offset << endl;
#endif
                    MPI_Send(E_prev_init_buffer,
                             (sub_n + 2) * (sub_m + 2), MPI_DOUBLE,
                             receive_rank, TAG_E_PREV_INIT, MPI_COMM_WORLD);
                    MPI_Send(R_init_buffer,
                             (sub_n + 2) * (sub_m + 2), MPI_DOUBLE,
                             receive_rank, TAG_R_INIT, MPI_COMM_WORLD);
                }

                // update offset
                n_offset += sub_n;
            }
            m_offset += sub_m;
        }
        free(E_prev_init_buffer);
        free(R_init_buffer);
        // pack the self's E_prev and R
        int self_sub_m = (0 < reminder_m) ? (quotient_m + 1) : (quotient_m);
        int self_sub_n = (0 < reminder_n) ? (quotient_n + 1) : (quotient_n);
        for (int mm = 1; mm < self_sub_m + 2; mm++)
        {
            for (int nn = 0; nn < self_sub_n + 2; nn++)
            {
                int local_idx = mm * (self_sub_n + 2) + nn;
                int global_idx = mm * (cb.n + 2) + nn;
                E_prev[local_idx] = E_prev[global_idx];
                R[local_idx] = R[global_idx];
                // can't use memcpy here because the range may overlap
                // rely on compiler vectorization
            }
        }
#if VERBOSE_PRINT
        cout << "self_sub_mn" << self_sub_m << self_sub_n << endl;
        printMat("Rank 0 init Matrix E_prev", E_prev, self_sub_m + 2, self_sub_n + 2);
        printMat("Rank 0 init Matrix R", R, self_sub_m + 2, self_sub_n + 2);
#endif
#endif
    }
    else
    {
#ifdef _MPI_
        // receive init value from rank-0 process
        int sub_m = (my_rank_y < reminder_m) ? (quotient_m + 1) : (quotient_m);
        int sub_n = (my_rank_x < reminder_n) ? (quotient_n + 1) : (quotient_n);
        MPI_Recv(E_prev,
                 (sub_m + 2) * (sub_n + 2), MPI_DOUBLE,
                 0, TAG_E_PREV_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(R,
                 (sub_m + 2) * (sub_n + 2), MPI_DOUBLE,
                 0, TAG_R_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if VERBOSE_PRINT
        cout << my_rank << ": received init values with shape: " << sub_m << "x" << sub_n << endl;
        printMat("Rank i init Matrix E_prev", E_prev, sub_m + 2, sub_n + 2);
        printMat("Rank i init Matrix R", R, sub_m + 2, sub_n + 2);
#endif
#endif
    }
}

double *alloc1D(int m, int n)
{
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return (E);
}

void printMat(const char mesg[], double *E, int m, int n)
{
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m) * (n); i++)
    {
        int rowIndex = i / (n);
        int colIndex = i % (n);
        printf("%6.3f ", E[i]);
        if (colIndex == n - 1)
            printf("\n");
    }
}
