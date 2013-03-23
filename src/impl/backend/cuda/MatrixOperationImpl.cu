/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Matrix transpose with Cuda
* Device code.
*/


#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

#include <gpumatrix/impl/backend/MatrixOperationInterface.h>

#include "shared_mem.cuh"

#define BLOCK_DIM 16
namespace gpumatrix
{
	namespace impl
	{

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
template <typename T> __global__ void _row_major_transpose(T *odata, const T *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

template <typename T> void transpose( T *odata, const T *idata,  int r, int c)  
{									


	dim3 dimGrid(BLOCK_DIM,BLOCK_DIM,1);;
	dim3 dimBlock(int(ceil(float(r)/BLOCK_DIM)),int(ceil(float(c)/BLOCK_DIM)));	

	_row_major_transpose<T><<<dimBlock,dimGrid>>>(odata, idata, r,c);						
}			



template void transpose<double>( double *odata, const double *idata,  int r, int c) ; 
template void transpose<float>( float *odata, const float *idata,  int r, int c)  ;

//
//def _row_wise_sum(tgt, src):
//
//
//    krnl = _get_row_wise_sum_kernel()
//
//    h, w = src.shape
//    assert tgt.shape == (h,)
//
//    threadsize = min(512,int(2** ceil(log(w, 2))));
//
//    gridsize = (h,1);
//    blocksize = (threadsize,1,1)
//
//    sharedsize = threadsize*sizeof(c_float)
//
//    krnl(tgt, src, numpy.int32(h), numpy.int32(w), block = blocksize, grid = gridsize, shared = sharedsize);
//
//// This naive transpose kernel suffers from completely non-coalesced writes.
//// It can be up to 10x slower than the kernel above for large matrices.
//__global__ void transpose_naive(float *odata, float* idata, int width, int height)
//{
//   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
//   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
//   
//   if (xIndex < width && yIndex < height)
//   {
//       unsigned int index_in  = xIndex + width * yIndex;
//       unsigned int index_out = yIndex + height * xIndex;
//       odata[index_out] = idata[index_in]; 
//   }
//}

}
}

