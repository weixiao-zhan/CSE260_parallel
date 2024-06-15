
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   gridDim.x = n / TILE_BLOCK_N;
   gridDim.y = n / TILE_BLOCK_M;

   // you can overwrite blockDim here if you like.
   if (n % TILE_BLOCK_N != 0)
      gridDim.x++;
   if (n % TILE_BLOCK_M != 0)
      gridDim.y++;

   blockDim.x = BLOCK_DIM_M;
   blockDim.y = BLOCK_DIM_N;
}
