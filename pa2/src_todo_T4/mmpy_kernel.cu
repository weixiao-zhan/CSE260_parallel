#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
#ifdef SQUARE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    // helper naming
    int M = N;
    int K = N;
#else
__global__ void matMul(int M, int K, int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
#endif
    int block_n = blockIdx.x;
    int block_m = blockIdx.y;
    int thread_n = threadIdx.x;
    int thread_m = threadIdx.y;

    // declare double caching in shared memory
    extern __shared__ _FTYPE_ smem[];
    _FTYPE_ (*A_tile_current)[TILE_BLOCK_K] = (_FTYPE_ (*)[TILE_BLOCK_K])smem;
    _FTYPE_ (*B_tile_current)[TILE_BLOCK_N] = (_FTYPE_ (*)[TILE_BLOCK_N])(smem + 2 * TILE_BLOCK_K * TILE_BLOCK_M);
    _FTYPE_ (*A_tile_next)[TILE_BLOCK_K] = (_FTYPE_ (*)[TILE_BLOCK_K])(smem + 1 * TILE_BLOCK_K * TILE_BLOCK_M);
    _FTYPE_ (*B_tile_next)[TILE_BLOCK_N] = (_FTYPE_ (*)[TILE_BLOCK_N])(smem + 3 * TILE_BLOCK_K * TILE_BLOCK_M);

    // decalre and init accumulator 
    register _FTYPE_ C_frag[TILE_THREAD_M][TILE_THREAD_N];
    #pragma unroll
    for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
    #pragma unroll
    for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
        C_frag[i_m][i_n] = 0;
    }
    }

    // first fetch
    int block_k = 0;
    #pragma unroll
    for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
    #pragma unroll
    for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
        int Am = block_m*TILE_BLOCK_M + i_m*BLOCK_DIM_M + thread_m;
        int Ak = block_k*TILE_BLOCK_K + i_n*BLOCK_DIM_N + thread_n;
        int Bk = block_k*TILE_BLOCK_K + i_m*BLOCK_DIM_M + thread_m;
        int Bn = block_n*TILE_BLOCK_N + i_n*BLOCK_DIM_N + thread_n;

        _FTYPE_ Aval = (Am < M && Ak < K) ? A[Am*K + Ak] : 0;
        A_tile_current[i_m*BLOCK_DIM_M + thread_m][i_n*BLOCK_DIM_N + thread_n] = Aval;

        _FTYPE_ Bval = (Bk < K && Bn < N) ? B[Bk*N + Bn] : 0;
        B_tile_current[i_m*BLOCK_DIM_M + thread_m][i_n*BLOCK_DIM_N + thread_n] = Bval;
    }
    }
    
    // main loop
    #pragma unroll
    for (block_k = 1; block_k*TILE_BLOCK_K <  K; ++block_k) {
        // consume one block tile
        __syncthreads();
        #pragma unroll
        for (int thread_k = 0; thread_k < TILE_BLOCK_K; ++thread_k) {
            #pragma unroll
            for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
            #pragma unroll
            for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
                C_frag[i_m][i_n]
                        += A_tile_current[i_m*BLOCK_DIM_M + thread_m][thread_k]
                            * B_tile_current[thread_k][i_n*BLOCK_DIM_N + thread_n];
            }
        }
        }

        // load next block tile
        #pragma unroll
        for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
        #pragma unroll
        for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
            int Am = block_m*TILE_BLOCK_M + i_m*BLOCK_DIM_M + thread_m;
            int Ak = block_k*TILE_BLOCK_K + i_n*BLOCK_DIM_N + thread_n;
            int Bk = block_k*TILE_BLOCK_K + i_m*BLOCK_DIM_M + thread_m;
            int Bn = block_n*TILE_BLOCK_N + i_n*BLOCK_DIM_N + thread_n;

            _FTYPE_ Aval = (Am < M && Ak < K) ? A[Am*K + Ak] : 0;
            A_tile_next[i_m*BLOCK_DIM_M + thread_m][i_n*BLOCK_DIM_N + thread_n] = Aval;

            _FTYPE_ Bval = (Bk < K && Bn < N) ? B[Bk*N + Bn] : 0;
            B_tile_next[i_m*BLOCK_DIM_M + thread_m][i_n*BLOCK_DIM_N + thread_n] = Bval;
        }
        }

        // swap pointer
        _FTYPE_ (*temp_A)[TILE_BLOCK_K] = A_tile_current;
        A_tile_current = A_tile_next;
        A_tile_next = temp_A;
        _FTYPE_ (*temp_B)[TILE_BLOCK_N] = B_tile_current;
        B_tile_current = B_tile_next;
        B_tile_next = temp_B;
    }

    // consume the last block tile
    __syncthreads();
    #pragma unroll
    for (int thread_k = 0; thread_k < TILE_BLOCK_K; ++thread_k) {
        #pragma unroll
        for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
        #pragma unroll
        for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
            C_frag[i_m][i_n]
                    += A_tile_current[i_m*BLOCK_DIM_M + thread_m][thread_k]
                        * B_tile_current[thread_k][i_n*BLOCK_DIM_N + thread_n];
        }
        }
    }
    __syncthreads();

    // store results
    #pragma unroll
    for (int i_m = 0; i_m < TILE_THREAD_M; ++i_m) {
    #pragma unroll
    for (int i_n = 0; i_n < TILE_THREAD_N; ++i_n) {
        int Cm = block_m*TILE_BLOCK_M + i_m*BLOCK_DIM_M + thread_m;
        int Cn = block_n*TILE_BLOCK_N + i_n*BLOCK_DIM_N + thread_n;
        if (Cm < M && Cn < N) {
            C[Cm*N + Cn] = C_frag[i_m][i_n];
        }
    }
    }
}

#endif