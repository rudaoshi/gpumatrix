#ifndef CUBLAS_TEMPLATE_H
#define CUBLAS_TEMPLATE_H

#include <cublas.h>
#include <stdexcept>

namespace gpumatrix
{
	namespace gpu
	{
		/* C = alpha * op(A) * op(B) + beta * C */
		template< typename T> void cublas_gemm(char transa, char transb, int m, int n, int k, 
			T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc);

		


		/* y = alpha*x + y */
		template< typename T>  void cublas_axpy (int n, T alpha, const T *x, int incx, T *y, int incy);

		

		/* x = alpha*x*/
		template< typename T>  void cublas_scal (int n, T alpha, T *x, int incx);

		

		/* y = alpha * op(A) * x + beta * y */
		template <typename T> void cublas_gemv (char trans, int m, int n, T alpha, const T *A, int lda, 
			const T *x, int incx, T beta, T *y, int incy);

		/* res = norm(x) */
		template <typename T> T cublas_nrm2 (int n, const T *x, int incx);

		///* res = sum(x) */
		//template <typename T> T cublas_asum(int n, const T *x, int incx);

		///* id = min_id(x) */
		//template <typename T> int cublas_Imin(int n, const T *x, int incx);

		///* res = sum(x) */
		//template <typename T> T cublas_Imax(int n, const T *x, int incx);
		
		/* res = sum(x) */
		template <typename T> T cublas_dot(int n, const T *x, int incx, const T * y ,int incy);
	}
}

#endif