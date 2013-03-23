#ifndef ASSIGN_IMPL_H
#define ASSIGN_IMPL_H


#include<gpumatrix/impl/backend/Interface.h>
#include<gpumatrix/impl/EvalInterface.h>

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
	// Dest = Matrix
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Matrix<T> & m, const Assign& assign_fn)
		{
			dest.resize(m.rows(),m.cols());
			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Dest = Matrix
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const MatrixConstReference<T> & m, const Assign& assign_fn)
		{
			dest.resize(m.rows(),m.cols());
			impl::copy<T>(dest.data(),m.data(), m.size());
		}

		// Map<Matrix>  = Matrix
		template <typename T,typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const Matrix<T> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Matrix to Map Assignment");

			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Map<Matrix> = Matrix
		template <typename T,typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const MatrixConstReference<T> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Matrix to Map Assignment");

			impl::copy<T>(dest.data(),m.data(), m.size());
		}

		// Dest = Vector
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Vector<T> & m, const Assign& assign_fn)
		{
			dest.resize(m.size());
			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Dest = Vector
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const VectorConstReference<T> & m, const Assign& assign_fn)
		{
			dest.resize(m.size());
			impl::copy<T>(dest.data(),m.data(), m.size());
		}

		// Map<Vector> = Vector
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const Vector<T> & m, const Assign& assign_fn)
		{
			if (dest.size() != m.size())
				throw runtime_error("Dimensionality donot Match for Vector to Map Assignment");

			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Dest = Vector
		template <typename T,typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const VectorConstReference<T> & m, const Assign& assign_fn)
		{
			if (dest.size() != m.size())
				throw runtime_error("Dimensionality donot Match for Vector to Map Assignment");

			impl::copy<T>(dest.data(),m.data(), m.size());
		}


		// Dest = Array
		template <typename T,int D, typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Array<T,D> & m, const Assign& assign_fn)
		{
			dest.resize(m.rows(),m.cols());
			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Dest = Array
		template <typename T,int D,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const ArrayConstReference<T,D> & m, const Assign& assign_fn)
		{
			dest.resize(m.rows(),m.cols());
			impl::copy<T>(dest.data(),m.data(), m.size());
		}

		// Map<Vector>  = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const Array<T,1> & m, const Assign& assign_fn)
		{
			if (dest.size() != m.size())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Map<Vector> = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const ArrayConstReference<T,1> & m, const Assign& assign_fn)
		{
			if (dest.size() != m.size())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(),m.data(), m.size());
		}

		// Map<Matrix>  = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const Array<T,2> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Map<Matrix> = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const ArrayConstReference<T,2> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(),m.data(), m.size());
		}


		// Map<Array>  = Array
		template <typename T,int D, typename Assign> 
		void do_assign(Map<Array<T,D>>& dest, const Array<T,D> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(), m.data(),m.size());
		}
		// Map<Array> = Array
		template <typename T,int D, typename Assign> 
		void do_assign(Map<Array<T,D>>& dest, const ArrayConstReference<T,D> & m, const Assign& assign_fn)
		{
			if (dest.rows() != m.rows() || dest.cols() != m.cols())
				throw runtime_error("Dimensionality donot Match for Array to Map Assignment");

			impl::copy<T>(dest.data(),m.data(), m.size());
		}


		template <typename E,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const E & expr, const Assign& assign_fn)
		{
			typename XprResultType<E>:: result_type  result = expr.eval();
			impl::do_assign(dest,result,assign_fn);
		}

		template <typename E,typename Dest,typename Assign> 
		void do_assign(NoAliasProxy<Dest> & dest, const E & expr, const Assign& assign_fn)
		{
			impl::eval(dest.lord(),expr,assign_fn);
		}
	}

}

#endif
