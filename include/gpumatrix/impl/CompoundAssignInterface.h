#ifndef COMPOUND_ASSIGN_INTERFACE_H
#define COMPOUND_ASSIGN_INTERFACE_H

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

	namespace backend
	{
		// Dest op= value
		template <typename POD,typename Dest,typename Func> 
		void do_scalar_compound_assign(Dest& dest, POD alpha, const Func& fn);

	// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Matrix<T> & m, const Func& fn);
		// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const MatrixConstReference<T> & m, const Func& fn);
			// Dest = Matrix
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Matrix<T>> & m, const Func& fn);
		
		// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Vector<T> & m, const Func& fn);
		// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const VectorConstReference<T> & m, const Func& fn);
				// Dest = Vector
		template <typename T,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Vector<T>> & m, const Func& fn);
		

		// Dest = Array
		template <typename T,int D, typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Array<T,D> & m, const Func& fn);
		// Dest = Array
		template <typename T,int D,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const ArrayConstReference<T,D> & m, const Func& fn);

				// Dest = Array
		template <typename T,int D, typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const Map<Array<T,D>> & m, const Func& fn);

		template <typename E,typename Dest,typename Func> 
		void do_compound_assign(Dest& dest, const E & expr, const Func& fn);

		template <typename E,typename Dest,typename Func> 
		void do_compound_assign(NoAliasProxy<Dest> & dest, const E & expr, const Func& fn);
		

	}

}


#endif
