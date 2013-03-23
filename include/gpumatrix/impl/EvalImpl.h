#ifndef EVAL_IMPL_H
#define EVAL_INPL_H


#include <gpumatrix/impl/EvalInterface.h>
#include <gpumatrix/impl/Interface.h>


namespace gpumatrix
{

	namespace impl
	{
		
		template <class Dest> 
		void check_size(Dest & dest, int size)
		{
			dest.resize(size);
		}

		template <class Dest> 
		void check_size(Map<Dest> & dest, int size)
		{
			if (dest.size() != size)
				throw runtime_error("Dimensionality donot Match");
		}

		template <class Dest> 
		void check_size(Dest & dest, int rows, int cols)
		{
			dest.resize(rows,cols);

		}

		
		template <class Dest> 
		void check_size(Map<Dest> & dest, int rows, int cols)
		{
			if (dest.rows() != rows || dest.cols() != cols)
				throw runtime_error("Dimensionality donot Match");
		}
		
		
				template <typename T>
		MatrixConstReference<T> eval(const MatrixConstReference<T> & m) 
		{
			return m;
		}

		template <typename T>
		VectorConstReference<T> eval(const VectorConstReference<T> & m) 
		{
			return m;
		}

		template <typename T, int D>
		ArrayConstReference<T,D> eval(const ArrayConstReference<T,D> & m) 
		{
			return m;
		}


		template <typename E> 
		typename XprResultType<E>:: result_type eval(const E & expr) 
		{
			typename XprResultType<E>::result_type result;

			impl::eval(result,expr,Fcnl_assign<typename E::value_type,typename E::value_type>());

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


  #pragma region algebra operation

		// Dest = Matrix.transpose()
		template <typename E,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMatrixTranspose<E> & trans, const Assign& assign_fn)
		{
			check_size(dest,trans.rows(),trans.cols());

			typename E::result_type A = trans.expr().eval();
	
			impl::transpose<typename E::value_type> (dest.data(),A.data(),A.rows(),A.cols());
		}

		// Dest = M1*M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMMProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.rows(),prod.cols());
			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();

			typedef typename XprMMProduct<E1,E2>::value_type value_type;

			impl::gemm<value_type> ('N', 'N', A.rows(), B.cols(), A.cols(), 1, A.data(),
				A.rows(), B.data(), B.rows(), 0, dest.data(), dest.rows());
		}

		// Dest = M1.transpose()*M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtMProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.rows(),prod.cols());
			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();

			typedef typename XprMtMProduct<E1,E2>::value_type value_type;

			impl::gemm<value_type> ('T', 'N', A.cols(), B.cols(), A.rows(), 1, A.data(),
				A.rows(), B.data(), B.rows(), 0, dest.data(), dest.rows());
		}

		// Dest = M1*M2.transpose()
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMMtProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.rows(),prod.cols());
			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();

			typedef typename XprMMtProduct<E1,E2>::value_type value_type;

			impl::gemm<value_type> ('N', 'T', A.rows(), B.rows(), A.cols(), 1, A.data(),
				A.rows(), B.data(), B.rows(), 0, dest.data(), dest.rows());
		}

		// Dest = M1.transpose()*M2.transpose()
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtMtProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.rows(),prod.cols());
			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();
		

			typedef typename XprMMtProduct<E1,E2>::value_type value_type;

			impl::gemm<value_type> ('T', 'T', A.cols(), B.rows(), A.rows(), 1, A.data(),
				A.rows(), B.data(), B.rows(), 0, dest.data(), dest.rows());
		}

		// Dest = M1*V2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMVProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.size());

			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();

			typedef typename XprMVProduct<E1,E2>::value_type value_type;

			impl::gemv <value_type>('N', A.rows(),A.cols(), 1, A.data(),  A.rows(), B.data(), 1, 0, dest.data(),1);

		}

		// Dest = M1.transpose()*V2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtVProduct<E1,E2> & prod, const Assign& assign_fn)
		{
			check_size(dest,prod.size());
			typename E1::result_type A = prod.lhs().eval();
			typename E2::result_type B = prod.rhs().eval();

			typedef typename XprMVProduct<E1,E2>::value_type value_type;

			impl::gemv <value_type>('T', A.rows(),A.cols(), 1, A.data(),  A.rows(), B.data(), 1, 0, dest.data(),1);

		}

  #pragma endregion

  #pragma region arithmetic operation
		
					// Dest = alpha*Matrix
		template < typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprMatrix<E>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());
			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprMatrix<E>::result_type B = expr.rhs().eval();

			impl::scalar_array_mul(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest = Matrix*alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprMatrix<E>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprMatrix<E>::result_type B = expr.lhs().eval();

			scalar_array_mul(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest =  Matrix /alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprMatrix<E>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprMatrix<E>::result_type B = expr.lhs().eval();

			impl::scalar_array_mul(dest.data(),1.0/alpha,B.data(),dest.size());
		}

		// Dest = alpha*Vector
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprVector<E>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprVector<E>::result_type B = expr.rhs().eval();

			impl::scalar_array_mul(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest = Vector*alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprVector<E>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprVector<E>::result_type B = expr.lhs().eval();

			impl::scalar_array_mul(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest =  Vector /alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprVector<E>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprVector<E>::result_type B = expr.rhs().eval();

			impl::scalar_array_mul(dest.data(),1.0/alpha,B.data(),dest.size());
		}

		// Dest = M1+M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprMatrix<E1>,							
					XprMatrix<E2>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprMatrix<E1>::result_type A = expr.lhs().eval();
			typename XprMatrix<E2>::result_type B = expr.rhs().eval();

			array_add(dest.data(),A.data(),B.data(),dest.size());
		
		}

		// Dest = M1 - M2
		template < typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprMatrix<E1>,							
					XprMatrix<E2>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprMatrix<E1>::result_type A = expr.lhs().eval();
			typename XprMatrix<E2>::result_type B = expr.rhs().eval();

			array_sub(dest.data(),A.data(),B.data(),dest.size());
		
		}

		// Dest = V1+V2
		template < typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprVector<E1>,							
					XprVector<E2>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename XprVector<E1>::result_type A = expr.lhs().eval();
			typename XprVector<E2>::result_type B = expr.rhs().eval();

			array_add(dest.data(),A.data(),B.data(),dest.size());
		
		}

		// Dest = v1 - v2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprVector<E1>,							
					XprVector<E2>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename XprVector<E1>::result_type A = expr.lhs().eval();
			typename XprVector<E1>::result_type B = expr.rhs().eval();

			impl::array_sub(dest.data(),A.data(),B.data(),dest.size());
		
		}


		//template <int D, typename POD,typename T, typename Dest,typename Assign> 
		//void eval(Dest& dest, 
		//	const XprBinOp<
		//			Fcnl_mul<POD,typename T>,
		//			XprLiteral< POD >,							
		//			XprArray<ArrayConstReference<T,D>,D>
		//	> & expr, 
		//	const Assign& assign_fn)
		//{
		//	dest.resize(expr.rows(),expr.cols());

		//	T alpha = (T)expr.lhs().eval();
		//	typename XprArray<ArrayConstReference<T,D>,D>::result_type B = expr.rhs().eval();

		//	scalar_array_mul(dest.data(),alpha,B.data(),dest.size());
		//}


					// Dest = alpha*Array
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprArray<E,D>::result_type B = expr.rhs().eval();

			impl::scalar_array_mul(dest.data(),alpha,B.data(),dest.size());
		}

		// Dest = Array*alpha
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprArray<E,D>::result_type B = expr.lhs().eval();

			impl::scalar_array_mul(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest = M1+M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprArray<E1,D>::result_type A = expr.lhs().eval();
			typename XprArray<E2,D>::result_type B = expr.rhs().eval();

			impl::array_add(dest.data(),A.data(),B.data(),dest.size());
		
		}

		// Dest = M1 - M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprArray<E1,D>::result_type A = expr.lhs().eval();
			typename XprArray<E2,D>::result_type B = expr.rhs().eval();

			impl::array_sub(dest.data(),A.data(),B.data(),dest.size());
		
		}

		// Dest = alpha + Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprArray<E,D>::result_type B = expr.rhs().eval();

			impl::scalar_array_add(dest.data(),alpha,B.data(),dest.size());
		}

		// Dest = Array + alpha
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprArray<E,D>::result_type B = expr.lhs().eval();

			impl::scalar_array_add(dest.data(),alpha,B.data(),dest.size());

		}

		// Dest = alpha - Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprArray<E,D>::result_type B = expr.rhs().eval();

			impl::scalar_array_sub(dest.data(),alpha,B.data(),dest.size());
		}

		// Dest = Array - alpha
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprArray<E,D>::result_type B = expr.lhs().eval();

			impl::scalar_array_add(dest.data(),-alpha,B.data(),dest.size());

		}

		// Dest =  Array /alpha
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprArray<E,D>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());
			typename E::value_type alpha = (typename E::value_type)expr.rhs().eval();
			typename XprArray<E,D>::result_type B = expr.lhs().eval();

			impl::scalar_array_mul(dest.data(),1.0/alpha,B.data(),dest.size());
		}


		// Dest = alpha / Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());
			typename E::value_type alpha = (typename E::value_type)expr.lhs().eval();
			typename XprArray<E,D>::result_type B = expr.rhs().eval();

			impl::scalar_array_div(dest.data(),alpha,B.data(),dest.size());
		}

		//// Dest = M1+M2
		//template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		//void eval(Dest& dest, 
		//	const XprBinOp<
		//			Fcnl_add<typename E1::value_type,typename E2::value_type>,
		//			XprArray<E1,D>,							
		//			XprArray<E2,D>
		//	> & expr, 
		//	const Assign& assign_fn)
		//{
		//	dest.resize(expr.rows(),expr.cols());

		//	typename XprArray<E1,D>::result_type A = expr.lhs().eval();
		//	typename XprArray<E2,D>::result_type B = expr.rhs().eval();

		//	array_add(dest.data(),A.data(),B.data(),dest.size());
		//
		//}

		//// Dest = M1 - M2
		//template <int D, typename E1, typename E2,typename Dest,typename Assign> 
		//void eval(Dest& dest, 
		//	const XprBinOp<
		//			Fcnl_sub<typename E1::value_type,typename E2::value_type>,
		//			XprArray<E1,D>,							
		//			XprArray<E2,D>
		//	> & expr, 
		//	const Assign& assign_fn)
		//{
		//	dest.resize(expr.rows(),expr.cols());

		//	typename XprArray<E1,D>::result_type A = expr.lhs().eval();
		//	typename XprArray<E2,D>::result_type B = expr.rhs().eval();

		//	array_sub(dest.data(),A.data(),B.data(),dest.size());
		//
		//}

  #pragma endregion


		// Dest = M1 * M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprArray<E1,D>::result_type A = expr.lhs().eval();
			typename XprArray<E2,D>::result_type B = expr.rhs().eval();

			impl::array_mul(dest.data(),A.data(),B.data(),dest.size());
		
		} 

		// Dest = M1 / M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprArray<E1,D>::result_type A = expr.lhs().eval();
			typename XprArray<E2,D>::result_type B = expr.rhs().eval();

			impl::array_div(dest.data(),A.data(),B.data(),dest.size());
		
		} 


		// Dest = M.rowwise().sum()
		template <typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const RowWiseSum<E> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename E::result_type M = expr.expr().eval();

			rowwise_sum(dest.data(),M.data(),M.rows(),M.cols());
		
		} 

		// Dest = M.colwise().sum()
		template <typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const ColWiseSum<E> & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename E::result_type M = expr.expr().eval();

			impl::colwise_sum(dest.data(),M.data(),M.rows(),M.cols());
		
		} 

		// Dest = - M
		template <typename UnOP, typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprMatrix<E> > & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());

			typename XprMatrix<E>::result_type M = expr.expr().eval();

			impl::unary_array_op( dest.data(), M.data(), M.size(),UnOP()) ;
		
		} 

		// Dest = M.exp()
		template <typename UnOP, typename E, int D, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprArray<E,D> > & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.rows(),expr.cols());
			typename XprArray<E,D>::result_type M = expr.expr().eval();

			unary_array_op( dest.data(), M.data(), M.size(),UnOP()) ;
		
		} 

		// Dest = -M
		template <typename UnOP, typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprVector<E> > & expr, 
			const Assign& assign_fn)
		{
			check_size(dest,expr.size());

			typename XprVector<E>::result_type M = expr.expr().eval();

			impl::unary_array_op( dest.data(), M.data(), M.size(),UnOP()) ;
		
		} 


		//// Dest = M.sqnorm()
		//template <typename E,typename Dest,typename Assign> 
		//void eval(Dest& dest, 
		//	const XprUnOp<Fcnl_sqnorm<typename E::value_type>, E > & expr, 
		//	const Assign& assign_fn)
		//{
		//	typename E::result_type M = expr.expr().eval();

		//	dest = cublas_nrm2(M.size(),M.data(),1) ;

		//	dest = dest * dest;
		//
		//} 
		
	}
	

}



#endif