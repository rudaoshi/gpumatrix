#ifndef MATRIX_KERNEL_H
#define MATRIX_KERNEL_H
namespace gpumatrix
{
	namespace gpu
	{
		namespace matrix_kernel
		{

			template <typename T> void rowwise_sum(T * odata, const T * idata, int r, int c);
			template <typename T> void colwise_sum(T * odata, const T * idata, int r, int c);
			template <typename T> void transpose( T *odata, const T *idata,  int r, int c) ;
		}
	}
}


#endif