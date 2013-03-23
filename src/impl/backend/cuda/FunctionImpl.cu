#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

#include <gpumatrix/impl/backend/FunctionInterface.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>


#include "shared_mem.cuh"

#define BLOCK_DIM 16
namespace gpumatrix
{
	namespace impl
	{

		  
		#define BINARY_ARRAY_FUNC(FUNCNAME, FUNC, TYPE) \
	\
	__global__ void _array_##FUNCNAME (TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] = FUNC(idata1[index],idata2[index])  ;								\
			}																		\
			\
			}																			\
			\
			void array_##FUNCNAME( TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_array_##FUNCNAME<<<numGrid,256>>>(odata, idata1, idata2, size);						\
			}			


__device__ double cross_entropy( double x, double act)
			{
				return log(1+exp(act)) + x*act;
			}

__device__ double cross_entropy_diff( double x, double r)
			{
				return (1-x)*r - x*(1-r);
			}

			BINARY_ARRAY_FUNC(cross_entropy, cross_entropy, double)
			BINARY_ARRAY_FUNC(cross_entropy_diff, cross_entropy_diff, double)

		  
			__device__ double arrayinv(double val)
			{
				return 1.0/val;
			}

			__device__ float arrayinv(float val)
			{
				return 1.0/val;
			}


			__device__ double logistic(double val)
			{
				return 1.0/(1.0+exp(-val));
			}

			__device__ float logistic(float val)
			{
				return 1.0/(1.0+exp(-val));
			}

#define UNARY_ARRAY_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _unary_array_op (TYPE *odata, const TYPE  * idata, int size, const Fcnl_##OPNAME<TYPE> * func) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] = OP(idata[index]);								\
			}																		\
			\
			}																			\
			\
			void unary_array_op( TYPE *odata, const TYPE  * idata, int size,const Fcnl_##OPNAME<TYPE> & func)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_unary_array_op<<<numGrid,256>>>(odata, idata , size,&func);						\
			}	

			UNARY_ARRAY_OP(exp, exp, double)
			UNARY_ARRAY_OP(exp, exp, float)
			UNARY_ARRAY_OP(log, log, double)
			UNARY_ARRAY_OP(log, log, float)
			UNARY_ARRAY_OP(neg, - , double)
			UNARY_ARRAY_OP(neg, - , float)
			UNARY_ARRAY_OP(arrayinv, arrayinv, double)
			UNARY_ARRAY_OP(arrayinv, arrayinv, float)
			UNARY_ARRAY_OP(logistic, logistic, double)
			UNARY_ARRAY_OP(logistic, logistic, float)


			template<typename T> T sum(const T * data, int size)
			{

				thrust::device_ptr<T> dev_ptr(const_cast<T *>(data));

				T  x = thrust::reduce(dev_ptr, dev_ptr+size, (T ) 0,
					thrust::plus<T >());

				return x;
			}

			template double sum<double>(const double * data, int size);
			template float sum<float>(const float * data, int size);

			template<typename T> T max_element(const T * data, int size)
			{
//#ifdef _DEBUG
//				return 255;
//#endif
				thrust::device_ptr<T> dev_ptr(const_cast<T *>(data));

				thrust::device_ptr<T> x = thrust::max_element(dev_ptr, dev_ptr + size);

				return x[0];
			}


			template double max_element<double>(const double * data, int size);
			template float max_element<float>(const float * data, int size);

			template<typename T> T min_element(const T * data, int size)
			{
//#ifdef _DEBUG
//				return 0.0;
//#endif
				thrust::device_ptr<T> dev_ptr(const_cast<T *>(data));

				thrust::device_ptr<T> x = thrust::min_element(dev_ptr, dev_ptr + size);

				int index = x - dev_ptr;

				if (index > size)
					std::cout << "min_element Error occured!" << std::endl;
				
				return x[0];

			}

			template double min_element<double>(const double * data, int size);
			template float min_element<float>(const float * data, int size);
			
			
			
			// column first storage
			template <typename T> __global__ void _rowwise_sum(T *odata, const T *idata, int r, int c)
			{
			    SharedMem<T> shared;
			    T* buff = shared.getPointer();

			    int blocksize = blockDim.x;

			    int cur_row = blockIdx.x;

			    unsigned int tidx   = threadIdx.x;

			    // 累加开始点
			    unsigned int idx    = __mul24(tidx, r) + cur_row ;		  // idata(cur_row,tidx)
			    unsigned int endidx = __mul24(c-1, r) +  cur_row +1;      // idata(r-1,columnid);

			    float temp_sum = 0;

			    // Compute partial sums down the column until we
			    // have covered the whole column
			    while (idx < endidx) {
				temp_sum += idata[idx];
				idx += blocksize * r;
			    }
			    buff[tidx] = temp_sum;

			    if (tidx > blocksize) buff[tidx] = 0;

			    __syncthreads();

			    // Parallel reduction of the partial sum
			    // in shared memory
			    if (blockDim.x == 512) {
				if (tidx < 256)
				    buff[tidx] += buff[tidx + 256];
				__syncthreads();
			    }

			    if (blockDim.x >= 256) {
				if (tidx < 128)
				    buff[tidx] += buff[tidx + 128];
				__syncthreads();
			    }

			    if (blockDim.x >= 128) {
				if (tidx < 64)
				    buff[tidx] += buff[tidx + 64];

				__syncthreads();
			    }

			    if (tidx < 32) {
				if (blockDim.x >=  64) {
				    buff[tidx] += buff[tidx + 32];
				}
				if (blockDim.x >=  32) {
				    buff[tidx] += buff[tidx + 16];
				}
				if (blockDim.x >=  16) {
				    buff[tidx] += buff[tidx + 8];
				}
				if (blockDim.x >=   8) {
				    buff[tidx] += buff[tidx + 4];
				}
				if (blockDim.x >=   4) {
				    buff[tidx] += buff[tidx + 2];
				}
				if (blockDim.x >=   2) {
				    buff[tidx] += buff[tidx + 1];
				}
			    }

			    // write result for this block to global mem
			    if (tidx == 0)
				odata[cur_row] = buff[0];

			}


			template <typename T> void rowwise_sum( T *odata, const T *idata,  int r, int c)  
			{													
				int threadsize = min((int)512,(int)pow(2,ceil(log2((double)c))));
				dim3 dimBlock(r,1);
				dim3 dimGrid(threadsize,1,1);	
				int sharedsize = threadsize*sizeof(T);

				_rowwise_sum<T><<<dimBlock,dimGrid,sharedsize>>>(odata, idata, r,c);						
			}			


			// column first storage
			template <typename T> __global__ void _colwise_sum(T *odata, const T *idata, int r, int c)
			{
			    SharedMem<T> shared;
			    T* buff = shared.getPointer();

			    int blocksize = blockDim.x;

			    int cur_col = blockIdx.x;

			    unsigned int tidx   = threadIdx.x;

			    // 累加开始点
			    unsigned int idx    = __mul24(cur_col, r) + tidx ;		  // idata(tidx,cur_col)
			    unsigned int endidx = __mul24(cur_col, r) +  r;      // idata(columnid,c-1);

			    float temp_sum = 0;

			    // Compute partial sums down the column until we
			    // have covered the whole column
			    while (idx < endidx) {
				temp_sum += idata[idx];
				idx += blocksize;
			    }
			    buff[tidx] = temp_sum;

			    if (tidx > blocksize) buff[tidx] = 0;

			    __syncthreads();

			    // Parallel reduction of the partial sum
			    // in shared memory
			    if (blockDim.x == 512) {
				if (tidx < 256)
				    buff[tidx] += buff[tidx + 256];
				__syncthreads();
			    }

			    if (blockDim.x >= 256) {
				if (tidx < 128)
				    buff[tidx] += buff[tidx + 128];
				__syncthreads();
			    }

			    if (blockDim.x >= 128) {
				if (tidx < 64)
				    buff[tidx] += buff[tidx + 64];

				__syncthreads();
			    }

			    if (tidx < 32) {
				if (blockDim.x >=  64) {
				    buff[tidx] += buff[tidx + 32];
				}
				if (blockDim.x >=  32) {
				    buff[tidx] += buff[tidx + 16];
				}
				if (blockDim.x >=  16) {
				    buff[tidx] += buff[tidx + 8];
				}
				if (blockDim.x >=   8) {
				    buff[tidx] += buff[tidx + 4];
				}
				if (blockDim.x >=   4) {
				    buff[tidx] += buff[tidx + 2];
				}
				if (blockDim.x >=   2) {
				    buff[tidx] += buff[tidx + 1];
				}
			    }

			    // write result for this block to global mem
			    if (tidx == 0)
				odata[cur_col] = buff[0];

			}


			template <typename T> void colwise_sum( T *odata, const T *idata,  int r, int c)  
			{													
				int threadsize = min((int)512,(int)pow(2,ceil(log2((double)r))));
				dim3 dimBlock(c,1);
				dim3 dimGrid(threadsize,1,1);	
				int sharedsize = threadsize*sizeof(T);

				_colwise_sum<T><<<dimBlock,dimGrid,sharedsize>>>(odata, idata, r,c);						
			}	
			
			
			template void rowwise_sum<double>(double * odata, const double * idata, int r, int c);
			template void rowwise_sum<float>(float * odata, const float * idata, int r, int c);

			template void colwise_sum<double>(double * odata, const double * idata, int r, int c);
			template void colwise_sum<float>(float * odata, const float * idata, int r, int c);


		
	}
}