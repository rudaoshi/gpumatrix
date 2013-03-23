#include <gpumatrix/impl/backend/BlasInterface.h>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

#include <cstdio>
using namespace std;

namespace gpumatrix
{
  
	namespace impl
	{

			template<> void gemm<double>(char transa, char transb, int m, int n, int k, 
				double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc)
			{
				cublasDgemm(transa, transb, m, n, k,alpha, A, lda, B, ldb, beta,  C, ldc);
				cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template<> void gemm<float>(char transa, char transb, int m, int n, int k, 
				float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
			{
				cublasSgemm(transa, transb, m, n, k,alpha, A, lda, B, ldb, beta,  C, ldc);
							cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}


			template<> void axpy<double>(int n, double alpha, const double *x, int incx, double *y, int incy)
			{
				cublasDaxpy (n, alpha, x, incx, y, incy);
						cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template<> void axpy<float>(int n, float alpha, const float *x, int incx, float *y, int incy)
			{
				cublasSaxpy (n, alpha, x, incx, y, incy);
						cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template<>  void scal<double > (int n, double alpha, double *x, int incx)
			{
				cublasDscal (n, alpha, x, incx);
						cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template<>  void scal<float> (int n, float alpha, float *x, int incx)
			{
				cublasSscal (n, alpha, x, incx);
							cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template <> void gemv<float> (char trans, int m, int n, float alpha, const float *A, int lda, 
				const float *x,int incx, float beta, float *y, int incy)
			{
				cublasSgemv ( trans, m, n, alpha,A,lda, x, incx, beta, y, incy);
							cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			
			template <> void gemv<double > (char trans, int m, int n, double alpha, const double *A, int lda, 
				const double *x,int incx, double beta, double *y, int incy)
			{
				cublasDgemv ( trans, m, n, alpha,A,lda, x, incx, beta, y, incy);
							cublasStatus err  = cublasGetError();

				if( CUBLAS_STATUS_SUCCESS != err) { 
					fprintf(stderr, "Cublas error in file '%s' in line %i \n", __FILE__, __LINE__);
					throw "Cublas error occured "; 
				}
			}

			template <> double nrm2 <double> (int n, const double *x, int incx)
			{
				return cublasDnrm2(n, x, incx);
			}

			template <> float nrm2 <float> (int n, const float *x, int incx)
			{
				return cublasSnrm2(n, x, incx);
			}


			template <> double dot <double> (int n, const double *x, int incx, const double * y, int incy)
			{
				return cublasDdot(n, x, incx, y, incy);
			}

			template <> float dot <float> (int n, const float *x, int incx, const float * y, int incy)
			{
				return cublasSdot(n, x, incx, y, incy);
			}

			//template <> double cublas_asum <double> (int n, const double *x, int incx)
			//{
			//	return cublasDasum(n, x, incx);
			//}

			//template <> float cublas_nrm2 <float> (int n, const float *x, int incx)
			//{
			//	return cublasSasum(n, x, incx);
			//}
		
	}
}