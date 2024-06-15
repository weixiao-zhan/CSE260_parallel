#ifndef MYTYPES_H
#define MYTYPES_H

// #define NAIVE

#define SQUARE

#ifdef NAIVE
#define TILE_BLOCK_M 32
#define TILE_BLOCK_K 32
#define TILE_BLOCK_N 32

#define BLOCK_DIM_M 32
#define BLOCK_DIM_N 32

#else

// size handled by one block
#define TILE_BLOCK_M 64
#define TILE_BLOCK_K 64
#define TILE_BLOCK_N 64

// size handled by one thread
#define TILE_THREAD_M 4
#define TILE_THREAD_N 4

// number of threads
// ensure TILE_block is multiple of TILE_thread
#define BLOCK_DIM_M TILE_BLOCK_M/TILE_THREAD_M
#define BLOCK_DIM_N TILE_BLOCK_N/TILE_THREAD_N

#endif
#endif
