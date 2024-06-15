/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order  
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"

const char* dgemm_desc = "my blislab ";


/* 
 * pack one subpanel of A
 *
 * pack like this 
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *     
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 *     0 0 0 0
 */
static inline
void packA_mcxkc_d(
        int    m,
        int    k,
        double *XA,
        int    ldXA,
        double *packA
        )
{
#ifdef VERBOSE_DEBUG
    printf("\n-packA_mcxkc_d- m:%d, k:%d", m, k);
    printf("\n-packA_mcxkc_d- a:%f", *XA);
#endif

#ifdef PAD
  int diff_m = DGEMM_MR - m;
  if (diff_m > 0) {
    for (int j = 0; j < k; ++j) { // cols
      for (int i = 0; i < m; ++i) { // rows
        packA[i + j*DGEMM_MR] = XA[i*ldXA + j];
      }
      memset(packA + j*DGEMM_MR + m, 0, diff_m * sizeof(double));
    }
  } else {
    for (int j = 0; j < k; ++j) { // cols
      for (int i = 0; i < m; ++i) { // rows
        packA[i + j*DGEMM_MR] = XA[i*ldXA + j];
      }
    }
  }
#else
  for (int j = 0; j < k; ++j) { // cols
    for (int i = 0; i < m; ++i) { // rows
      packA[i + j*m] = XA[i*ldXA + j];
    }
  }
#endif
}

/*
 * --------------------------------------------------------------------------
 */

/* 
 * pack one subpanel of B
 * 
 * pack like this 
 * if B is 
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 *
 * Then pack 
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */
static inline
void packB_kcxnc_d(
        int    n,
        int    k,
        double *XB,
        int    ldXB, // ldXB is the original k
        double *packB
        )
{
#ifdef VERBOSE_DEBUG
    printf("\n-packB_kcxnc_d- k:%d, n:%d", k, n);
    printf("\n-packB_kcxnc_d- a:%f", *XB);
#endif

#ifdef PAD
  int diff_n = DGEMM_NR-n;
  if (diff_n > 0){
    for (int i = 0; i < k; ++i) {
      memcpy(packB + i*DGEMM_NR, XB + i*ldXB, n*sizeof(double));
      memset(packB + i*DGEMM_NR + n, 0, DGEMM_NR-n);
    }
  } else {
    for (int i = 0; i < k; ++i) {
      memcpy(packB + i*DGEMM_NR, XB + i*ldXB, n*sizeof(double));
    }
  }
#else
  for (int i = 0; i < k; ++i) {
    memcpy(packB + i*n, XB + i*ldXB, n*sizeof(double));
  }
#endif
}

/*
 * --------------------------------------------------------------------------
 */
static
inline
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        const double * packA,
        const double * packB,
        double * C,
        int    ldc
        )
{
#ifdef VERBOSE_DEBUG
  printf("\n-bl_macro_kernel- m:%d, k:%d, n:%d", m, k, n);
#endif

  int i, j;
  aux_t  aux;
  for ( i = 0; i < m; i += DGEMM_MR ) {                      // 2-th loop around micro-kernel
    for ( j = 0; j < n; j += DGEMM_NR ) {                    // 1-th loop around micro-kernel

#ifdef FIXED_R
      ( *bl_micro_kernel_4k4 ) (
        k,
        &packA[i * DGEMM_KC],
        &packB[j * DGEMM_KC],
        &C[ i * ldc + j ],
        (unsigned long long) ldc,
        &aux
      );
#else
      ( *bl_micro_kernel ) (
#ifdef PAD
        k,
        DGEMM_MR,
        DGEMM_NR,
        &packA[i * DGEMM_KC],
        &packB[j * DGEMM_KC],
#else
        k,
        min(m-i, DGEMM_MR),
        min(n-j, DGEMM_NR),
        &packA[i * k],
        &packB[j * k],
#endif
        &C[ i * ldc + j ],
        (unsigned long long) ldc,
        &aux
      );
#endif
    }                                                        // 1-th loop around micro-kernel
  }                                                          // 2-th loop around micro-kernel
}




void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *XA,
        int    lda,
        double *XB,
        int    ldb,
        double *C,       
        int    ldc       
        )
{
  int    ic, ib, jc, jb, pc, pb;
  double *packA, *packB;
  
#ifdef PAD
  int padded;
  double* pad_C;
  int pad_ldc = ( m + DGEMM_MC - 1 )/DGEMM_MC*DGEMM_MC;
  if (m%DGEMM_MC > 0 || n%DGEMM_NC > 0){
    int h = ( n + DGEMM_NC - 1 )/DGEMM_NC*DGEMM_NC;
    pad_C = bl_malloc_aligned(pad_ldc, h, sizeof(double) );
    memset(pad_C, 0, pad_ldc * h * sizeof(double));
    padded = 1;
  } else {
    pad_C = C;
    padded = 0;
  }
#endif

#ifdef BC_8k8_OFFSET
  static_bc_8k8_offset[ 0] = 0;
  static_bc_8k8_offset[ 1] = 4;
  static_bc_8k8_offset[ 2] = pad_ldc;
  static_bc_8k8_offset[ 3] = pad_ldc + 4;
  static_bc_8k8_offset[ 4] = 2*pad_ldc;
  static_bc_8k8_offset[ 5] = 2*pad_ldc + 4;
  static_bc_8k8_offset[ 6] = 3*pad_ldc;
  static_bc_8k8_offset[ 7] = 3*pad_ldc + 4;
  static_bc_8k8_offset[ 8] = 4*pad_ldc;
  static_bc_8k8_offset[ 9] = 4*pad_ldc + 4;
  static_bc_8k8_offset[10] = 5*pad_ldc;
  static_bc_8k8_offset[11] = 5*pad_ldc + 4;
  static_bc_8k8_offset[12] = 6*pad_ldc;
  static_bc_8k8_offset[13] = 6*pad_ldc + 4;
  static_bc_8k8_offset[14] = 7*pad_ldc;
  static_bc_8k8_offset[15] = 7*pad_ldc + 4;
  // printf("$%d: ", static_bc_8k8_offset);
  // for (int i = 0; i < 16; i++){
  //   printf("%d, ", static_bc_8k8_offset[i]);
  // }
#endif

#ifdef BF_4k4_OFFSET
  svbool_t pred = svptrue_b64();
  svint64_t size_vector = svdup_s64(sizeof(double));
  svint64_t ldc_vector = svdup_s64(pad_ldc);
  svint64_t row_offsets = svmul_s64_z(pred, ldc_vector, svindex_s64(0, 1));
  svint64_t col_offsets, offsets;
  // [00,11,22,33]
  col_offsets = svindex_s64(0, 1);
  offsets = svadd_s64_z(pred, row_offsets, col_offsets);
  svint64_t c0x_offsets = svmul_s64_z(pred, offsets, size_vector);
  svst1_s64(pred, static_bf_4k4_offset, c0x_offsets);
  // [03,12,21,30]
  col_offsets = svrev_s64(col_offsets);
  offsets = svadd_s64_z(pred, row_offsets, col_offsets);
  svint64_t c1x_offsets = svmul_s64_z(pred, offsets, size_vector);
  svst1_s64(pred, static_bf_4k4_offset+4, c1x_offsets);
  // [01,10,23,32]
  col_offsets = svext_s64(col_offsets, col_offsets, 2);
  offsets = svadd_s64_z(pred, row_offsets, col_offsets);
  svint64_t c2x_offsets = svmul_s64_z(pred, offsets, size_vector);
  svst1_s64(pred, static_bf_4k4_offset+8, c2x_offsets);
  // [02,13,20,31]
  col_offsets = svrev_s64(col_offsets);
  offsets = svadd_s64_z(pred, row_offsets, col_offsets);
  svint64_t c3x_offsets = svmul_s64_z(pred, offsets, size_vector);
  svst1_s64(pred, static_bf_4k4_offset+12, c3x_offsets);
#endif


#ifdef PACK
  packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC/DGEMM_MR + 1 )* DGEMM_MR, sizeof(double) );
  packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC/DGEMM_NR + 1 )* DGEMM_NR, sizeof(double) );
#endif

  for ( ic = 0; ic < m; ic += DGEMM_MC ) {              // 5-th loop around micro-kernel
    ib = min( m - ic, DGEMM_MC );
    for ( pc = 0; pc < k; pc += DGEMM_KC ) {          // 4-th loop around micro-kernel
      pb = min( k - pc, DGEMM_KC );

#ifdef PACK
	    int i, j;
      for ( i = 0; i < ib; i += DGEMM_MR ) {
        packA_mcxkc_d(
          min( ib - i, DGEMM_MR ), /* m */
          pb,                      /* k */
          &XA[ pc + lda*(ic + i)], /* XA - start of micropanel in A */
          k,                       /* ldXA */
#ifdef PAD
          &packA[ i * DGEMM_KC ]
#else
          &packA[ i * pb ]         /* packA */
#endif
        );
      }
#else
	    packA = &XA[pc + ic * lda ];
#endif

	    for ( jc = 0; jc < n; jc += DGEMM_NC ) {        // 3-rd loop around micro-kernel
        jb = min( m - jc, DGEMM_NC );

#ifdef PACK
        for ( j = 0; j < jb; j += DGEMM_NR ) {
          packB_kcxnc_d(
            min( jb - j, DGEMM_NR ), /* n */
            pb,                      /* k */
            &XB[ ldb * pc + jc + j], /* XB - starting row and column for this panel */
            n,                      /* ldXB */
#ifdef PAD
            &packB[ j * DGEMM_KC ]
#else
            &packB[ j * pb ]        /* packB */
#endif
          );
        }
#else
        packB = &XB[ldb * pc + jc ];
#endif

	      bl_macro_kernel(
          ib,
          jb,
          pb,
          packA,
          packB,
#ifdef PAD
          &pad_C[ic * pad_ldc +jc],
          pad_ldc
#else
          &C[ ic * ldc + jc ], 
          ldc
#endif
          );
	      }                                               // End 3.rd loop around micro-kernel
      }                                                 // End 4.th loop around micro-kernel
  }                                                     // End 5.th loop around micro-kernel

#ifdef PAD
if (padded) {
  for(int i = 0; i < m; ++i) {
    memcpy(C + i*ldc, pad_C + i*pad_ldc, n*sizeof(double));
  }
  free( pad_C );
}
#endif

#ifdef PACK
  free( packA );
  free( packB );
#endif
}

void square_dgemm(int lda, double *A, double *B, double *C){
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C,  lda);
}