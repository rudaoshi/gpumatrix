#ifndef COMPOUND_ASSIGN_IMPL_H
#define COMPOUND_ASSIGN_IMPL_H

#include <gpumatrix/impl/CompoundAssignInterface.h>
#include <gpumatrix/impl/backend/Interface.h>

namespace gpumatrix
{
	template <class T/**/> class Matrix;
	template <class T/**/> class Vector;
	template <class T, int D> class Array;

	template <class T/**/> class MatrixConstReference;
	template <class T/**/> class VectorConstReference;
	template <class T, int D> class ArrayConstReference;
	template <class C> class Map;
	template <class C> class NoAliasProxy;

	namespace impl
	{
		// Dest op= value
		template <typename POD,typename Dest,typename Func> 
		void do_scalar_compound_assign(Dest& dest, POD alpha, const Func& fn)
		{
			impl::scalar_array_compound_op(dest.data(),(typename Dest::value_type)alpha,dest.size(),fn);
		}

	// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Matrix<T> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Matrix Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
		// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const MatrixConstReference<T> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Matrix Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
			// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Matrix<T>> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Matrix Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
		
		// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Vector<T> & m, const Func& fn)
		{
			if (dest.size() != m.size() )
				throw runtime_error("Dimensionality donot Match for Vector Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
		// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const VectorConstReference<T> & m, const Func& fn)
		{
			if (dest.size() != m.size() )
				throw runtime_error("Dimensionality donot Match for Vector Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(), m.size(),fn);
		}
				// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Vector<T>> & m, const Func& fn)
		{
			if (dest.size() != m.size() )
				throw runtime_error("Dimensionality donot Match for Vector Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
		

		// Dest = Array
		template <typename T,int D, typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Array<T,D> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}
		// Dest = Array
		template <typename T,int D,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const ArrayConstReference<T,D> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}

				// Dest = Array
		template <typename T,int D, typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Array<T,D>> & m, const Func& fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array Compound Assignment");
			impl::array_compound_op(dest.data(),m.data(),m.size(),fn);
		}

		template <typename E,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const E & expr, const Func& fn)
		{
			typename XprResultType<E>:: result_type  result = expr.eval();
			do_compound_assign(dest,result,fn);
		}

		template <typename E,typename Dest,typename Func> 
		void do_compound_assign(NoAliasProxy<Dest> & dest, const E & expr, const Func& fn)
		{
			do_compound_assign(dest.lord(),expr,fn);
		}
		
		
	}

}

#endif
