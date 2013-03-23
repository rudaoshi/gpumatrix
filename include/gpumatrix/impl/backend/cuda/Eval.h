#ifndef TVMET_XPR_EVAL_SPECIALIZATION_H
#define TVMET_XPR_EVAL_SPECIALIZATION_H



#include <gpumatrix/backend/cuda/cublas_template.h>
#include <gpumatrix/backend/cuda/EvalImpl.h>


namespace gpumatrix
{
	template <typename T> class MatrixConstReference;

	namespace gpu
	{
// 		template <typename T>
// 		MatrixConstReference<T> eval(const MatrixConstReference<T> & m) 
// 		{
// 			return m;
// 		}
// 
// 		template <typename T>
// 		VectorConstReference<T> eval(const VectorConstReference<T> & m) 
// 		{
// 			return m;
// 		}
// 
// 		template <typename T, int D>
// 		ArrayConstReference<T,D> eval(const ArrayConstReference<T,D> & m) 
// 		{
// 			return m;
// 		}

		////template <typename E> 
		////typename XprResultType<XprMatrix<E>>:: result_type eval(const XprMatrix<E> & expr) 
		////{
		////	return expr.expr().eval();
		////}


		//template <int D, typename POD,typename E> 
		//Array<typename E::value_type,D> eval(const XprBinOp<Fcnl_mul<POD,typename E::value_type>,XprLiteral< POD >,	XprArray<E,D>> & expr)
		//{
		//	typename XprResultType<E>:: result_type result;

		//	eval_impl::eval(result,expr,Fcnl_assign<E::value_type,E::value_type>());

		//	return result;
		//}

		//Matrix<double> eval(const XprBinOp<Fcnl_mul<int,double>,XprLiteral<int>,	XprMatrix<MatrixConstReference<double>>> & expr)
		//{
		//	Matrix<double> result;

		//	eval_impl::eval(result,expr,Fcnl_assign<double,double>());

		//	return result;
		//}

// 		template <typename T, int D, typename POD>
// 		Array<T,D> eval(const XprBinOp<Fcnl_mul<POD,T>,XprLiteral<POD>,	XprArray<ArrayConstReference<T,D>,D>> & expr)
// 		{
// 			Array<T,D> result;
// 
// 			eval_impl::eval(result,expr,Fcnl_assign<T,T>());
// 
// 			return result;
// 		}

		template <typename E> 
		typename XprResultType<E>:: result_type eval(const E & expr) 
		{
			typename XprResultType<E>::result_type result;

			eval_impl::eval(result,expr,Fcnl_assign<typename E::value_type,typename E::value_type>());

			return result;
			/*if (expr.lhs().cols() != expr.rhs().rows())
				throw runtime_error("Dimension not Match for Matrix Multiplication");

			auto mat1 = expr.lhs().eval();
			auto mat2 = expr.rhs().eval();

			Matrix<value_type> result(m1.rows(),m2.cols());

			cublas_gemm<value_type> ('N', 'N', mat1.rows(), mat2.cols(), mat1.cols(), 1.0, mat1.data(),
				mat1.rows(), mat2.data(), mat2.rows(),
				0, result.data(), mat1.rows());

			return result;*/

		}
	}

	
}

#endif