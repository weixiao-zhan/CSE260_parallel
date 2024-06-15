/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <malloc.h>
#include <emmintrin.h>
#include <cstring>

#ifdef _MPI_
#include <mpi.h>
#endif

#define FUSED 1
#define FUSED_TILED 0
#define SIMD 1

using namespace std;

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double *alloc1D2(int m, int n);
void unpack_self_and_receive_update_from_all_rank(double *E_prev, int m, int n,
                                                  int quotient_m, int quotient_n,
                                                  int reminder_m, int reminder_n);
void send_update_to_rank_0(double *E_prev, int m, int n);
void pack_self(double *E_prev, int m, int n);

extern control_block cb;
static int TAG_E_PREV_UPDATE = 2;
static int TAG_E_PREV_FINAL = 3;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq)
{
    double l2norm = sumSq / (double)((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{

    // Simulated time is different from the integer timestep number
    // double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double mx, sumSq;

    int my_rank = 0;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif

    int my_rank_x = my_rank % cb.px;
    int my_rank_y = my_rank / cb.px;

#ifndef _MPI_
    int m = cb.m;
    int n = cb.n;
#else
    // compute sub-matrix size
    int quotient_n = cb.n / cb.px,
        quotient_m = cb.m / cb.py,
        reminder_n = cb.n % (cb.px),
        reminder_m = cb.m % (cb.py);
    int m = (my_rank_y < reminder_m) ? (quotient_m + 1) : (quotient_m);
    int n = (my_rank_x < reminder_n) ? (quotient_n + 1) : (quotient_n);

    double *send_left_buffer = alloc1D2(m, 1);
    double *recv_left_buffer = alloc1D2(m, 1);
    double *send_right_buffer = alloc1D2(m, 1);
    double *recv_right_buffer = alloc1D2(m, 1);
    // top and bottom cells can be send and receive in-place

    // MPI sending and receiving communicator
    MPI_Request send_left, send_right, send_top, send_bot;
    MPI_Request recv_left, recv_right, recv_top, recv_bot;
#endif

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (int niter = 0; niter < cb.niters; niter++)
    {

        if (cb.debug && (niter == 0) && my_rank == 0)
        {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E_prev, -1, m + 1, n + 1);
        }

        /////////////////////////////////////////////////////////////////////////////////
        // handle boundary exchanges
#ifndef _MPI_
        /*
         * Copy data from boundary of the computational box to the
         * padding region, set up for differencing computational box's boundary
         *
         * These are physical boundary conditions, and are not to be confused
         * with ghost cells that we would use in an MPI implementation
         *
         * The reason why we copy boundary conditions is to avoid
         * computing single sided differences at the boundaries
         * which increase the running time of solve()
         *
         */

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i, j;

        // Fills in the LEFT Ghost Cells
        for (i = 0; i < (m + 2) * (n + 2); i += (n + 2))
        {
            E_prev[i] = E_prev[i + 2];
        }

        // Fills in the RIGHT Ghost Cells
        for (i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2))
        {
            E_prev[i] = E_prev[i - 2];
        }

        // Fills in the TOP Ghost Cells
        for (i = 0; i < (n + 2); i++)
        {
            E_prev[i] = E_prev[i + (n + 2) * 2];
        }

        // Fills in the BOTTOM Ghost Cells
        for (i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++)
        {
            E_prev[i] = E_prev[i - (n + 2) * 2];
        }
#else
        // left cells
        if (1 <= my_rank_x && my_rank_x < cb.px && !cb.noComm)
        { // not left edge processes
            // pack send left
            int nn = 1;
#ifdef SIMD
#pragma simd
#endif
            for (int mm = 1; mm < m + 1; mm++)
            {
                send_left_buffer[mm - 1] = E_prev[mm * (n + 2) + nn];
            }
            // send and receive
            // cout << "send left" << my_rank << "->" << my_rank -1 << endl;
            MPI_Isend(send_left_buffer, m, MPI_DOUBLE, my_rank - 1, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(send_left));
            // cout << "recv left" << my_rank << "<-" << my_rank -1 << endl;
            MPI_Irecv(recv_left_buffer, m, MPI_DOUBLE, my_rank - 1, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(recv_left));
        }
        else
        { // left edge processes
#ifdef SIMD
#pragma simd
#endif
            for (int i = 0; i < (m + 2) * (n + 2); i += (n + 2))
            {
                E_prev[i] = E_prev[i + 2];
            }
        }

        // right cells
        if (0 <= my_rank_x && my_rank_x < cb.px - 1 && !cb.noComm)
        {
            // pack send right
            int nn = n;
#ifdef SIMD
#pragma simd
#endif
            for (int mm = 1; mm < m + 1; mm++)
            {
                send_right_buffer[mm - 1] = E_prev[mm * (n + 2) + nn];
            }
            // send and receive
            // cout << "send right" << my_rank << "->" << my_rank +1 << endl;
            MPI_Isend(send_right_buffer, m, MPI_DOUBLE, my_rank + 1, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(send_right));
            // cout << "recv right" << my_rank << "<-" << my_rank +1 << endl;
            MPI_Irecv(recv_right_buffer, m, MPI_DOUBLE, my_rank + 1, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(recv_right));
        }
        else
        { // right edge processes
#ifdef SIMD
#pragma simd
#endif
            for (int i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2))
            {
                E_prev[i] = E_prev[i - 2];
            }
        }

        // top cells
        if (1 <= my_rank_y && my_rank_y < cb.py && !cb.noComm)
        {
            // cout << "send top" << my_rank << "->" << my_rank -cb.px << endl;
            MPI_Isend(E_prev + n + 3, n, MPI_DOUBLE, my_rank - cb.px, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(send_top));
            // cout << "recv top" << my_rank << "<-" << my_rank -cb.px << endl;
            MPI_Irecv(E_prev + 1, n, MPI_DOUBLE, my_rank - cb.px, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(recv_top));
        }
        else
        { // top edge processes
#ifdef SIMD
#pragma simd
#endif
            for (int i = 0; i < (n + 2); i++)
            {
                E_prev[i] = E_prev[i + (n + 2) * 2];
            }
        }

        // bot cells
        if (0 <= my_rank_y && my_rank_y < cb.py - 1 && !cb.noComm)
        {
            // cout << "recv bot" << my_rank << "<-" << my_rank +cb.px << endl;
            MPI_Isend(E_prev + m * (n + 2) + 1, n, MPI_DOUBLE, my_rank + cb.px, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(send_bot));
            // cout << "send bot" << my_rank << "->" << my_rank +cb.px << endl;
            MPI_Irecv(E_prev + (m + 1) * (n + 2) + 1, n, MPI_DOUBLE, my_rank + cb.px, TAG_E_PREV_UPDATE, MPI_COMM_WORLD, &(recv_bot));
        }
        else
        { // bot edge processes
#ifdef SIMD
#pragma simd
#endif
            for (int i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++)
            {
                E_prev[i] = E_prev[i - (n + 2) * 2];
            }
        }

        // Synchronization
        if (my_rank_x > 0 && !cb.noComm)
        {
            MPI_Wait(&(recv_left), MPI_STATUS_IGNORE);
            MPI_Wait(&(send_left), MPI_STATUS_IGNORE);
            // unpack received left
            int nn = 0;
#ifdef SIMD
#pragma simd
#endif
            for (int mm = 1; mm < m + 1; mm++)
            {
                E_prev[mm * (n + 2) + nn] = recv_left_buffer[mm - 1];
            }
        }
        if (my_rank_x < cb.px - 1 && !cb.noComm)
        {
            MPI_Wait(&(recv_right), MPI_STATUS_IGNORE);
            MPI_Wait(&(send_right), MPI_STATUS_IGNORE);
            // unpack received right
            int nn = n + 1;
#ifdef SIMD
#pragma simd
#endif
            for (int mm = 1; mm < m + 1; mm++)
            {
                E_prev[mm * (n + 2) + nn] = recv_right_buffer[mm - 1];
            }
        }
        if (my_rank_y > 0 && !cb.noComm)
        {
            MPI_Wait(&(recv_top), MPI_STATUS_IGNORE);
            MPI_Wait(&(send_top), MPI_STATUS_IGNORE);
        }
        if (my_rank_y < cb.py - 1 && !cb.noComm)
        {
            MPI_Wait(&(recv_bot), MPI_STATUS_IGNORE);
            MPI_Wait(&(send_bot), MPI_STATUS_IGNORE);
        }
#endif

        //////////////////////////////////////////////////////////////////////////////

#if FUSED
        // Solve for the excitation, a PDE
#if FUSED_TILED
        int TILE_LENGTH = 64;
        for (int n_tile_offset = 1; n_tile_offset < n + 1; n_tile_offset += TILE_LENGTH)
        {
            for (int mm = 1; mm < m + 1; mm++)
            {
#ifdef SIMD
#pragma simd
#endif
                for (int nn = n_tile_offset; nn < min(n + 1, n_tile_offset + TILE_LENGTH); nn++)
                {
                    int i = mm * (n + 2) + nn;
                    E[i] = E_prev[i] + alpha * (E_prev[i + 1] + E_prev[i - 1] - 4 * E_prev[i] + E_prev[i + (n + 2)] + E_prev[i - (n + 2)]);
                    E[i] += -dt * (kk * E_prev[i] * (E_prev[i] - a) * (E_prev[i] - 1) + E_prev[i] * R[i]);
                    R[i] += dt * (epsilon + M1 * R[i] / (E_prev[i] + M2)) * (-R[i] - kk * E_prev[i] * (E_prev[i] - b - 1));
                }
            }
        }
#else
        for (int mm = 1; mm < m + 1; mm++)
        {
#ifdef SIMD
#pragma simd
#endif
            for (int nn = 1; nn < n + 1; nn++)
            {
                int i = mm * (n + 2) + nn;
                E[i] = E_prev[i] + alpha * (E_prev[i + 1] + E_prev[i - 1] - 4 * E_prev[i] + E_prev[i + (n + 2)] + E_prev[i - (n + 2)]);
                E[i] += -dt * (kk * E_prev[i] * (E_prev[i] - a) * (E_prev[i] - 1) + E_prev[i] * R[i]);
                R[i] += dt * (epsilon + M1 * R[i] / (E_prev[i] + M2)) * (-R[i] - kk * E_prev[i] * (E_prev[i] - b - 1));
            }
        }
#endif
#else
        // Solve for the excitation, a PDE
#if FUSED_TILED
        int TILE_LENGTH = 64;
        for (int n_tile_offset = 1; n_tile_offset < n + 1; n_tile_offset += TILE_LENGTH)
        {
            for (int mm = 1; mm < m + 1; mm++)
            {
#ifdef SIMD
#pragma simd
#endif
                for (int nn = n_tile_offset; nn < min(n + 1, n_tile_offset + TILE_LENGTH); nn++)
                {
                    int i = mm * (n + 2) + nn;
                    E[i] = E_prev[i] + alpha * (E_prev[i + 1] + E_prev[i - 1] - 4 * E_prev[i] + E_prev[i + (n + 2)] + E_prev[i - (n + 2)]);
                }
            }
        }
#else
        for (int mm = 1; mm < m + 1; mm++)
        {
#ifdef SIMD
#pragma simd
#endif
            for (int nn = 1; nn < n + 1; nn++)
            {
                int i = mm * (n + 2) + nn;
                E[i] = E_prev[i] + alpha * (E_prev[i + 1] + E_prev[i - 1] - 4 * E_prev[i] + E_prev[i + (n + 2)] + E_prev[i - (n + 2)]);
            }
        }
#endif
        /*
         * Solve the ODE, advancing excitation and recovery variables
         *     to the next timtestep
         */
        for (int mm = 1; mm < m + 1; mm++)
        {
#ifdef SIMD
#pragma simd
#endif
            for (int nn = 1; nn < n + 1; nn++)
            {
                int i = mm * (n + 2) + nn;
                E[i] += -dt * (kk * E_prev[i] * (E_prev[i] - a) * (E_prev[i] - 1) + E_prev[i] * R[i]);
                R[i] += dt * (epsilon + M1 * R[i] / (E_prev[i] + M2)) * (-R[i] - kk * E_prev[i] * (E_prev[i] - b - 1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq)
        {
            if (!(niter % cb.stats_freq))
            {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq && !(niter % cb.plot_freq))
        {
#ifdef _MPI_
            if (my_rank == 0)
            {
                unpack_self_and_receive_update_from_all_rank(E_prev, m, n, quotient_m, quotient_n, reminder_m, reminder_n);
                plotter->updatePlot(E_prev, niter, cb.m, cb.n);
                pack_self(E_prev, m, n);
            }
            else
            {
                send_update_to_rank_0(E_prev, m, n);
            }
#else
                   plotter->updatePlot(E_prev, niter, cb.m, cb.n);
#endif 
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;

    } // end of 'niter' loop at the beginning

#ifdef _MPI_
    // free buffer
    free(send_left_buffer);
    free(recv_left_buffer);
    free(send_right_buffer);
    free(recv_right_buffer);
#endif

    // return the L2 and infinity norms via in-out parameters
    stats(E_prev, m, n, &Linf, &sumSq);
    double fsumSq, fLinf;
#ifdef _MPI_
    if (!cb.noComm)
    {
        MPI_Reduce(&sumSq, &fsumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Linf, &fLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        Linf = fLinf;
    }
    else
        fsumSq = sumSq;
#else
    fsumSq = sumSq;
#endif

    L2 = L2Norm(fsumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}

double *alloc1D2(int m, int n)
{
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return (E);
}
#ifdef _MPI_
void send_update_to_rank_0(double *E_prev, int m, int n)
{
    // send E_prev to rank0
#if VERBOSE_PRINT
    cout << my_rank << ": sending final result with size " << m << n << endl;
    printMat2("Rank i final Matrix E_prev", E_prev, m + 2, n + 2);
#endif
    MPI_Send(E_prev,
             (n + 2) * (m + 2), MPI_DOUBLE,
             0, TAG_E_PREV_FINAL, MPI_COMM_WORLD);
}

/**
 * @brief unpack E from (m x n) to E (cb.m x cb.n) and receive other pack from workers
 */
void unpack_self_and_receive_update_from_all_rank(double *E_prev, int m, int n,
                                                  int quotient_m, int quotient_n,
                                                  int reminder_m, int reminder_n)
{
#if VERBOSE_PRINT
    printMat2("Rank 0 final Matrix E_prev", E_prev, m + 2, n + 2);
    cout << "0: unpack self" << endl;
#endif
    // unpack self's E_prev
    for (int mm = m; mm > 0; mm--)
    {
        for (int nn = n; nn > 0; nn--)
        {
            int local_idx = mm * (n + 2) + nn;
            int global_idx = mm * (cb.n + 2) + nn;
            E_prev[global_idx] = E_prev[local_idx];
            // can't use memcpy here because the range may overlap
            // rely on compiler vectorization
        }
    }
    // receive E_prev from gather_rank
    double *E_prev_final_buffer = alloc1D2(quotient_m + 3, quotient_n + 3);
    int m_offset, n_offset;
    m_offset = 0;
    for (int receive_rank_y = 0; receive_rank_y < cb.py; receive_rank_y++)
    {
        int sub_m = (receive_rank_y < reminder_m) ? (quotient_m + 1) : (quotient_m); // compute the sub matrix size in y direction

        n_offset = 0;
        for (int receive_rank_x = 0; receive_rank_x < cb.px; receive_rank_x++)
        {
            int sub_n = (receive_rank_x < reminder_n) ? (quotient_n + 1) : (quotient_n); // compute the sub matrix size in x directiond

            if (receive_rank_x != 0 || receive_rank_y != 0) // skip recving from self
            {
                // recv final values
                int receive_rank = receive_rank_y * cb.px + receive_rank_x;
#if VERBOSE_PRINT
                cout << "0: recving final results from " << receive_rank << " with size " << sub_m << sub_n << endl;
#endif
                MPI_Recv(E_prev_final_buffer,
                         (sub_n + 2) * (sub_m + 2), MPI_DOUBLE,
                         receive_rank, TAG_E_PREV_FINAL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // unpack the desired section
                for (int mm = 1; mm < sub_m + 1; mm++)
                {
                    for (int nn = 1; nn < sub_n + 1; nn++)
                    {
                        int local_idx = mm * (sub_n + 2) + nn;
                        int global_idx = (m_offset + mm) * (cb.m + 2) + (n_offset + nn);
                        E_prev[global_idx] = E_prev_final_buffer[local_idx];
                    }
                }
            }

            // update offset
            n_offset += sub_n;
        }
        m_offset += sub_m;
    }
    free(E_prev_final_buffer);
    printMat2("Final Matrix E_prev", E_prev, cb.m + 2, cb.n + 2);
}

void pack_self(double *E_prev, int m, int n)
{
    for (int mm = 1; mm < m + 2; mm++)
    {
        for (int nn = 0; nn < n + 2; nn++)
        {
            int local_idx = mm * (n + 2) + nn;
            int global_idx = mm * (cb.n + 2) + nn;
            E_prev[local_idx] = E_prev[global_idx];
        }
    }
}
#endif

void printMat2(const char mesg[], double *E, int m, int n)
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
