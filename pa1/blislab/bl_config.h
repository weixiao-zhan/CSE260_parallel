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
 * bl_config.h
 *
 *
 * Purpose:
 * this header file contains configuration parameters.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_SIMD_ALIGN_SIZE 32 
#define DGEMM_MR 8 // svcntd() = 4
#define DGEMM_NR 8 // svcntd() = 4

#define DGEMM_KC 512
#define DGEMM_NC 64
#define DGEMM_MC 1024

#define PACK
#define PAD

// #define BL_MICRO_KERNEL bl_dgemm_ukr
// #define BL_MICRO_KERNEL bl_dgemm_ukr_bc
#define FIXED_R
// #define BL_MICRO_KERNEL bl_dgemm_ukr_bc_4k4 // only work with PAD
// #define BL_MICRO_KERNEL bl_dgemm_ukr_bc_8k4 // only work with PAD
#define BL_MICRO_KERNEL bl_dgemm_ukr_bc_8k8 // only work with PAD
// #define BC_8k8_OFFSET

// #define BL_MICRO_KERNEL bl_dgemm_ukr_bf_4k4 // only work with PAD
// #define BF_4k4_OFFSET

// #define VERBOSE_DEBUG


// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
