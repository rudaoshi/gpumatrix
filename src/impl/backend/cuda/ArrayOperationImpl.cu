


#include <gpumatrix/impl/backend/ArrayOperationInterface.h>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include "shared_mem.cuh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

namespace gpumatrix
{
	namespace impl
	{




#define SCALAR_ARRAY_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _scalar_array_##OPNAME (TYPE *odata, const TYPE  alpha, const TYPE *idata,  int size) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] = alpha OP idata[index]  ;								\
		}																		\
		\
		}																			\
		\
		void scalar_array_##OPNAME( TYPE *odata, const TYPE  alpha, const TYPE *idata,  int size)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_scalar_array_##OPNAME<<<numGrid,256>>>(odata, alpha, idata, size);						\
		}																				

			SCALAR_ARRAY_OP(add,+,float)
				SCALAR_ARRAY_OP(add,+,double)
				SCALAR_ARRAY_OP(sub,-,float)
				SCALAR_ARRAY_OP(sub,-,double)
				SCALAR_ARRAY_OP(mul,*,float)
				SCALAR_ARRAY_OP(mul,*,double)
				SCALAR_ARRAY_OP(div,/,float)
				SCALAR_ARRAY_OP(div,/,double)

#define ARRAY_ARRAY_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _array_##OPNAME (TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] = idata1[index] OP idata2[index]  ;								\
			}																		\
			\
			}																			\
			\
			void array_##OPNAME( TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_array_##OPNAME<<<numGrid,256>>>(odata, idata1, idata2, size);						\
			}			


			ARRAY_ARRAY_OP(add,+,float)
			ARRAY_ARRAY_OP(add,+,double)
			ARRAY_ARRAY_OP(sub,-,float)
			ARRAY_ARRAY_OP(sub,-,double)
			ARRAY_ARRAY_OP(mul,*,float)
			ARRAY_ARRAY_OP(mul,*,double)
			ARRAY_ARRAY_OP(div,/,float)
			ARRAY_ARRAY_OP(div,/,double)


#define ARRAY_ARRAY_COMPOUND_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _array_compound_op (TYPE *odata, const TYPE  * idata, int size, const Fcnl_##OPNAME<TYPE,TYPE> * func) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] OP idata[index];								\
			}																		\
			\
			}																			\
			\
			void array_compound_op( TYPE *odata, const TYPE  * idata, int size,const Fcnl_##OPNAME<TYPE,TYPE> & func)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_array_compound_op<<<numGrid,256>>>(odata, idata,  size,&func);						\
			}			

				ARRAY_ARRAY_COMPOUND_OP(add_eq, +=, double)
				ARRAY_ARRAY_COMPOUND_OP(add_eq, +=, float)
				ARRAY_ARRAY_COMPOUND_OP(sub_eq, -=, double)
				ARRAY_ARRAY_COMPOUND_OP(sub_eq, -=, float)
				ARRAY_ARRAY_COMPOUND_OP(mul_eq, *=, double)
				ARRAY_ARRAY_COMPOUND_OP(mul_eq, *=, float)
				ARRAY_ARRAY_COMPOUND_OP(div_eq, /=, double)
				ARRAY_ARRAY_COMPOUND_OP(div_eq, /=, float)


#define SCALAR_ARRAY_COMPOUND_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _scalar_array_compound_op (TYPE *odata, TYPE  alpha, int size, const Fcnl_##OPNAME<TYPE,TYPE> * func) \
			{																			\
			\
			unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;				\
			\
			if(index < size )														\
			{																		\
			odata[index] OP alpha;								\
			}																		\
			\
			}																			\
			\
			void scalar_array_compound_op( TYPE *odata, TYPE  alpha,int size,const Fcnl_##OPNAME<TYPE,TYPE> & func)  \
			{																						\
			int numGrid = (size + 256 -1)/256;													\
			_scalar_array_compound_op<<<numGrid,256>>>(odata, alpha, size,&func);						\
			}			

			SCALAR_ARRAY_COMPOUND_OP(add_eq, +=, double)
			SCALAR_ARRAY_COMPOUND_OP(add_eq, +=, float)
			SCALAR_ARRAY_COMPOUND_OP(sub_eq, -=, double)
			SCALAR_ARRAY_COMPOUND_OP(sub_eq, -=, float)
			SCALAR_ARRAY_COMPOUND_OP(mul_eq, *=, double)
			SCALAR_ARRAY_COMPOUND_OP(mul_eq, *=, float)
			SCALAR_ARRAY_COMPOUND_OP(div_eq, /=, double)
			SCALAR_ARRAY_COMPOUND_OP(div_eq, /=, float)


#define BLOCK_DIM 16

#define ROWWISE_ARRAY_COMPOUND_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _rowwise_array_compound_op (TYPE *odata, int row, int col, const TYPE * x , const Fcnl_rowwise_##OPNAME<TYPE,TYPE> * func) \
			{																			\
			\
			unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;				\
			unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;				\
			if((xIndex < row) && (yIndex < col))							    	\
			{																		\
			odata[yIndex*row+xIndex] OP x[yIndex];								\
			}																		\
			\
			}																			\
			\
			\
			void rowwise_array_compound_op( TYPE *odata, int row, int col, const TYPE * x , const Fcnl_rowwise_##OPNAME<TYPE,TYPE> & func)	\
			{																															\
			dim3 dimGrid(BLOCK_DIM,BLOCK_DIM,1);																					\
			dim3 dimBlock(int(ceil(float(row)/BLOCK_DIM)),int(ceil(float(col)/BLOCK_DIM)));											\
			\
			_rowwise_array_compound_op<<<dimBlock,dimGrid>>>(odata, row, col, x, & func);											\
			}																															\


			ROWWISE_ARRAY_COMPOUND_OP(add_eq, +=, double)
			ROWWISE_ARRAY_COMPOUND_OP(add_eq, +=, float)

#define COLWISE_ARRAY_COMPOUND_OP(OPNAME, OP, TYPE) \
	\
	__global__ void _colwise_array_compound_op (TYPE *odata, int row, int col, const TYPE * x , const Fcnl_colwise_##OPNAME<TYPE,TYPE> * func) \
			{																			\
			\
			unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;				\
			unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;				\
			if((xIndex < row) && (yIndex < col))							    	\
			{																		\
			odata[yIndex*row+xIndex] OP x[xIndex];								\
			}																		\
			\
			}																			\
			\
			\
			void colwise_array_compound_op( TYPE *odata, int row, int col, const TYPE * x , const Fcnl_colwise_##OPNAME<TYPE,TYPE> & func)	\
			{																															\
			dim3 dimGrid(BLOCK_DIM,BLOCK_DIM,1);																					\
			dim3 dimBlock(int(ceil(float(row)/BLOCK_DIM)),int(ceil(float(col)/BLOCK_DIM)));											\
			\
			_colwise_array_compound_op<<<dimBlock,dimGrid>>>(odata, row, col, x, & func);											\
			}																															\


			COLWISE_ARRAY_COMPOUND_OP(add_eq, +=, double)
			COLWISE_ARRAY_COMPOUND_OP(add_eq, +=, float)



		
	}
}


