#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
                   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr- m:%d, k:%d, n:%d", m, k, n);
    printf("\n-bl_dgemm_ukr- a%f, b%f", *a, *b);
#endif

    int l, j, i;
    for ( l = 0; l < k; ++l ) {
        for ( j = 0; j < n; ++j ) {
            for ( i = 0; i < m; ++i ) {
            #ifdef PACK
                // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
                c(i,j,ldc) += a(l,i,m) * b(l,j,n);
            #else
                // ldc is used here because a[] and b[] are not packed by the
                // starter code
                c( i, j, ldc ) += a( i, l, ldc) * b( l, j, ldc );  
            #endif
            }
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

// Add sve broadcasting without unrolling
void bl_dgemm_ukr_bc(
        int k,
        int m,
        int n,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data )
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr_bc- m:%d, k:%d, n:%d", m, k, n);
    printf("\n-bl_dgemm_ukr_bc- a%f, b%f", *a, *b);
#endif

    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t cx;
    svbool_t npred =svwhilelt_b64_u64(0,n);
    for(int e = 0; e < m; ++e){
        cx = svld1_f64(npred, c + e*ldc);
        for(int kk = 0; kk < k; ++kk){
            ax = svdup_f64(*(a + kk*m + e));
            bx = svld1_f64(npred, b + kk*n);
            cx = svmla_f64_m(npred, cx, ax, bx);
        }
        svst1_f64(npred, c + e*ldc, cx);
    }
}

// only works for 4xkx4.
void bl_dgemm_ukr_bc_4k4( 
        int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data )
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr_bc_4k4- m:4, k:%d, n:4", k);
    printf("\n-bl_dgemm_ukr_bc_4k4- a%f, b%f", *a, *b);
#endif
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    register svbool_t pred = svptrue_b64();
    c0x = svld1_f64(pred, c);
    c1x = svld1_f64(pred, c + ldc);
    c2x = svld1_f64(pred, c + 2*ldc);
    c3x = svld1_f64(pred, c + 3*ldc);
    for (int kk = 0; kk < k; ++kk) {
        bx = svld1_f64(pred, b + kk*DGEMM_NR);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 0));
        c0x = svmla_f64_m(pred, c0x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 1));
        c1x = svmla_f64_m(pred, c1x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 2));
        c2x = svmla_f64_m(pred, c2x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 3));
        c3x = svmla_f64_m(pred, c3x, ax, bx);
    }

    svst1_f64(pred, c, c0x);
    svst1_f64(pred, c + ldc, c1x);
    svst1_f64(pred, c + 2*ldc, c2x);
    svst1_f64(pred, c + 3*ldc, c3x);
}

void bl_dgemm_ukr_bc_8k4(
        int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data)
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr_bc_8k4- m:8, k:%d, n:4", k);
    printf("\n-bl_dgemm_ukr_bc_8k4- a%f, b%f", *a, *b);
#endif

    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    register svbool_t pred = svptrue_b64();
    c0x = svld1_f64(pred, c);
    c1x = svld1_f64(pred, c + ldc);
    c2x = svld1_f64(pred, c + 2*ldc);
    c3x = svld1_f64(pred, c + 3*ldc);
    c4x = svld1_f64(pred, c + 4*ldc);
    c5x = svld1_f64(pred, c + 5*ldc);
    c6x = svld1_f64(pred, c + 6*ldc);
    c7x = svld1_f64(pred, c + 7*ldc);

    for (int kk = 0; kk < k; ++kk) {
        bx = svld1_f64(pred, b + kk*DGEMM_NR);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 0));
        c0x =svmla_f64_m(pred, c0x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 1));
        c1x =svmla_f64_m(pred, c1x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 2));
        c2x =svmla_f64_m(pred, c2x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 3));
        c3x =svmla_f64_m(pred, c3x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 4));
        c4x =svmla_f64_m(pred, c4x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 5));
        c5x =svmla_f64_m(pred, c5x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 6));
        c6x =svmla_f64_m(pred, c6x, ax, bx);

        ax  = svdup_f64(*(a + kk*DGEMM_MR + 7));
        c7x =svmla_f64_m(pred, c7x, ax, bx);
    }

    svst1_f64(pred, c + 0*ldc, c0x);
    svst1_f64(pred, c + 1*ldc, c1x);
    svst1_f64(pred, c + 2*ldc, c2x);
    svst1_f64(pred, c + 3*ldc, c3x);
    svst1_f64(pred, c + 4*ldc, c4x);
    svst1_f64(pred, c + 5*ldc, c5x);
    svst1_f64(pred, c + 6*ldc, c6x);
    svst1_f64(pred, c + 7*ldc, c7x);
}


long int static_bc_8k8_offset[16];
void bl_dgemm_ukr_bc_8k8(
        int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data)
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr_bc_8k8- m:8, k:%d, n:8", k);
    printf("\n-bl_dgemm_ukr_bc_8k8- a%f, b%f", *a, *b);
#endif
    register svfloat64_t ax;
    register svfloat64_t b0x, b1x;
    register svfloat64_t 
    c00x, c01x, 
    c10x, c11x,
    c20x, c21x,
    c30x, c31x,
    c40x, c41x,
    c50x, c51x,
    c60x, c61x,
    c70x, c71x;
    register svbool_t pred = svptrue_b64();
#ifdef BC_8k8_OFFSET
    c00x = svld1_f64(pred, c                           );
    c01x = svld1_f64(pred, c + 4                       );
    c10x = svld1_f64(pred, c + static_bc_8k8_offset[ 2]);
    c11x = svld1_f64(pred, c + static_bc_8k8_offset[ 3]);
    c20x = svld1_f64(pred, c + static_bc_8k8_offset[ 4]);
    c21x = svld1_f64(pred, c + static_bc_8k8_offset[ 5]);
    c30x = svld1_f64(pred, c + static_bc_8k8_offset[ 6]);
    c31x = svld1_f64(pred, c + static_bc_8k8_offset[ 7]);
    c40x = svld1_f64(pred, c + static_bc_8k8_offset[ 8]);
    c41x = svld1_f64(pred, c + static_bc_8k8_offset[ 9]);
    c50x = svld1_f64(pred, c + static_bc_8k8_offset[10]);
    c51x = svld1_f64(pred, c + static_bc_8k8_offset[11]);
    c60x = svld1_f64(pred, c + static_bc_8k8_offset[12]);
    c61x = svld1_f64(pred, c + static_bc_8k8_offset[13]);
    c70x = svld1_f64(pred, c + static_bc_8k8_offset[14]);
    c71x = svld1_f64(pred, c + static_bc_8k8_offset[15]);
#else
    c00x = svld1_f64(pred, c);
    c01x = svld1_f64(pred, c + 4);
    c10x = svld1_f64(pred, c + 1*ldc);
    c11x = svld1_f64(pred, c + 1*ldc + 4);
    c20x = svld1_f64(pred, c + 2*ldc);
    c21x = svld1_f64(pred, c + 2*ldc + 4);
    c30x = svld1_f64(pred, c + 3*ldc);
    c31x = svld1_f64(pred, c + 3*ldc + 4);
    c40x = svld1_f64(pred, c + 4*ldc);
    c41x = svld1_f64(pred, c + 4*ldc + 4);
    c50x = svld1_f64(pred, c + 5*ldc);
    c51x = svld1_f64(pred, c + 5*ldc + 4);
    c60x = svld1_f64(pred, c + 6*ldc);
    c61x = svld1_f64(pred, c + 6*ldc + 4);
    c70x = svld1_f64(pred, c + 7*ldc);
    c71x = svld1_f64(pred, c + 7*ldc + 4);
#endif
    for (int kk = 0; kk < k; ++kk) {
        b0x = svld1_f64(pred, b + kk*DGEMM_NR);
        b1x = svld1_f64(pred, b + kk*DGEMM_NR + 4);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 0));
        c00x = svmla_f64_m(pred, c00x, ax, b0x);
        c01x = svmla_f64_m(pred, c01x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 1));
        c10x = svmla_f64_m(pred, c10x, ax, b0x);
        c11x = svmla_f64_m(pred, c11x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 2));
        c20x = svmla_f64_m(pred, c20x, ax, b0x);
        c21x = svmla_f64_m(pred, c21x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 3));
        c30x = svmla_f64_m(pred, c30x, ax, b0x);
        c31x = svmla_f64_m(pred, c31x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 4));
        c40x = svmla_f64_m(pred, c40x, ax, b0x);
        c41x = svmla_f64_m(pred, c41x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 5));
        c50x = svmla_f64_m(pred, c50x, ax, b0x);
        c51x = svmla_f64_m(pred, c51x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 6));
        c60x = svmla_f64_m(pred, c60x, ax, b0x);
        c61x = svmla_f64_m(pred, c61x, ax, b1x);

        ax   = svdup_f64(*(a + kk*DGEMM_MR + 7));
        c70x = svmla_f64_m(pred, c70x, ax, b0x);
        c71x = svmla_f64_m(pred, c71x, ax, b1x);
    }
#ifdef BC_8k8_OFFSET
    svst1_f64(pred, c                           , c00x);
    svst1_f64(pred, c + 4                       , c01x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 2], c10x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 3], c11x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 4], c20x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 5], c21x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 6], c30x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 7], c31x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 8], c40x);
    svst1_f64(pred, c + static_bc_8k8_offset[ 9], c41x);
    svst1_f64(pred, c + static_bc_8k8_offset[10], c50x);
    svst1_f64(pred, c + static_bc_8k8_offset[11], c51x);
    svst1_f64(pred, c + static_bc_8k8_offset[12], c60x);
    svst1_f64(pred, c + static_bc_8k8_offset[13], c61x);
    svst1_f64(pred, c + static_bc_8k8_offset[14], c70x);
    svst1_f64(pred, c + static_bc_8k8_offset[15], c71x);
#else
    svst1_f64(pred, c,             c00x);
    svst1_f64(pred, c + 4,         c01x);
    svst1_f64(pred, c + 1*ldc,     c10x);
    svst1_f64(pred, c + 1*ldc + 4, c11x);
    svst1_f64(pred, c + 2*ldc,     c20x);
    svst1_f64(pred, c + 2*ldc + 4, c21x);
    svst1_f64(pred, c + 3*ldc,     c30x);
    svst1_f64(pred, c + 3*ldc + 4, c31x);
    svst1_f64(pred, c + 4*ldc,     c40x);
    svst1_f64(pred, c + 4*ldc + 4, c41x);
    svst1_f64(pred, c + 5*ldc,     c50x);
    svst1_f64(pred, c + 5*ldc + 4, c51x);
    svst1_f64(pred, c + 6*ldc,     c60x);
    svst1_f64(pred, c + 6*ldc + 4, c61x);
    svst1_f64(pred, c + 7*ldc,     c70x);
    svst1_f64(pred, c + 7*ldc + 4, c71x);
#endif
}

long int static_bf_4k4_offset[16];
void bl_dgemm_ukr_bf_4k4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data )
{
#ifdef VERBOSE_DEBUG
    printf("\n-bl_dgemm_ukr_bc_4k4- m:4, k:%d, n:4", k);
    printf("\n-bl_dgemm_ukr_bc_4k4- a%f, b%f", *a, *b);
#endif
    register svbool_t pred = svptrue_b64();
    // load offset and load C
    // [00,11,22,33]
    svint64_t c0x_offsets = svld1_s64(pred, static_bf_4k4_offset);
    register svfloat64_t c0x = svld1_gather_s64offset_f64(pred, c, c0x_offsets);
    // [03,12,21,30]
    svint64_t c1x_offsets = svld1_s64(pred, static_bf_4k4_offset+4);
    register svfloat64_t c1x = svld1_gather_s64offset_f64(pred, c, c1x_offsets);
    // [01,10,23,32]
    svint64_t c2x_offsets = svld1_s64(pred, static_bf_4k4_offset+8);
    register svfloat64_t c2x = svld1_gather_s64offset_f64(pred, c, c2x_offsets);
    // [02,13,20,31]
    svint64_t c3x_offsets = svld1_s64(pred, static_bf_4k4_offset+12);
    register svfloat64_t c3x = svld1_gather_s64offset_f64(pred, c, c3x_offsets);
 
    // butterfly method
    register svfloat64_t ax;
    register svfloat64_t bx;
    for (int kk = 0; kk < k; ++kk) {
        ax = svld1_f64(pred, a + kk*DGEMM_MR);
        bx = svld1_f64(pred, b + kk*DGEMM_NR);
        c0x =svmla_f64_m(pred, c0x, ax, bx);

        bx = svrev_f64(bx);
        c1x =svmla_f64_m(pred, c1x, ax, bx);

        bx = svext_f64(bx, bx, 2);
        c2x =svmla_f64_m(pred, c2x, ax, bx);

        bx = svrev_f64(bx);
        c3x =svmla_f64_m(pred, c3x, ax, bx);
    }
    
    // store back
    svst1_scatter_s64offset_f64(pred, c, c0x_offsets, c0x);
    svst1_scatter_s64offset_f64(pred, c, c1x_offsets, c1x);
    svst1_scatter_s64offset_f64(pred, c, c2x_offsets, c2x);
    svst1_scatter_s64offset_f64(pred, c, c3x_offsets, c3x);
}

void print_vector_f64(svfloat64_t vec) {
    double data[svcntd()] __attribute__((aligned(64))); // Assume max possible vector length for alignment
    svst1_f64(svptrue_b64(), data, vec);
    for (int i = 0; i < svcntd(); i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}

void print_vector_s64(svint64_t vec) {
    long int data[svcntd()] __attribute__((aligned(64))); // Assume max possible vector length for alignment
    svst1_s64(svptrue_b64(), data, vec);
    for (int i = 0; i < svcntd(); i++) {
        printf("%ld ", data[i]);
    }
    printf("\n");
}