#ifndef FUNCTION_H
#define FUNCTION_H

#include <gpumatrix/backend/cuda/cublas_template.h>
#include <gpumatrix/backend/cuda/EvalImpl.h>

#include <gpumatrix/backend/cuda/array_kernel.h>



namespace gpumatrix
{
	template <typename E> class Map;
	template <class T> class Matrix;
	template <class T> class Vector;
	template <class T, int D> class Array;
	template <class T> class MatrixConstReference;
	template <class T> class VectorConstReference;
	template <class T, int D> class ArrayConstReference;

	namespace gpu
	{

		template <typename E>
		typename E::value_type squaredNorm(const E & m)
		{
			typename E::value_type norm =  cublas_nrm2(m.size(),m.data(),1) ;
			return norm*norm;
		}



		template <typename E>
		typename E::value_type sum(const E & m)
		{
			return array_kernel::sum(m.data(),m.size());
			
		}

		template <typename E>
		typename E::value_type min(const E & m)
		{
			return array_kernel::min_element(m.data(),m.size());
			
		}

		template <typename E>
		typename E::value_type max(const E & m)
		{
			return array_kernel::max_element(m.data(),m.size());
			
		}

		template <typename E1, typename E2>
		typename E1::value_type dot(const E1 & v1, const E2 & v2)
		{
			return cublas_dot(v1.size(),v1.data(),1,v2.data(),1) ;
		}

		//template <typename T>
		//T squaredNorm(const MatrixConstReference<T> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T>
		//T squaredNorm(const Map<Matrix<T>> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T>
		//T squaredNorm(const Vector<T> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T>
		//T squaredNorm(const VectorConstReference<T> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T>
		//T squaredNorm(const Map<Vector<T>> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T, int D>
		//T squaredNorm(const Array<T,D> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T, int D>
		//T squaredNorm(const ArrayConstReference<T,D> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}

		//template <typename T, int D>
		//T squaredNorm(const Map<Array<T,D>> & m)
		//{
		//	T norm =  cublas_nrm2(m.size(),m.data(),1) ;
		//	return norm*norm;
		//}
	}
}
#endif