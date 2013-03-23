#ifndef GPU_DECLEAR_H
#define GPU_DECLEAR_H



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
  	template <class T/**/> class XprResultType;
	
	namespace gpu
	{
	  	template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Matrix<T> & m, const Assign& assign_fn);
		// Dest = Matrix
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const MatrixConstReference<T> & m, const Assign& assign_fn);

		// Map<Matrix>  = Matrix
		template <typename T,typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const Matrix<T> & m, const Assign& assign_fn);
		
		// Map<Matrix> = Matrix
		template <typename T,typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const MatrixConstReference<T> & m, const Assign& assign_fn);

		// Dest = Vector
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Vector<T> & m, const Assign& assign_fn);
		
		// Dest = Vector
		template <typename T,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const VectorConstReference<T> & m, const Assign& assign_fn);

		// Map<Vector> = Vector
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const Vector<T> & m, const Assign& assign_fn);
		// Dest = Vector
		template <typename T,typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const VectorConstReference<T> & m, const Assign& assign_fn);


		// Dest = Array
		template <typename T,int D, typename Dest,typename Assign> 
		void do_assign(Dest& dest, const Array<T,D> & m, const Assign& assign_fn);
		// Dest = Array
		template <typename T,int D,typename Dest,typename Assign> 
		void do_assign(Dest& dest, const ArrayConstReference<T,D> & m, const Assign& assign_fn);

		// Map<Vector>  = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const Array<T,1> & m, const Assign& assign_fn);
		// Map<Vector> = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Vector<T>>& dest, const ArrayConstReference<T,1> & m, const Assign& assign_fn);
		

		// Map<Matrix>  = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const Array<T,2> & m, const Assign& assign_fn);
		
		// Map<Matrix> = Array
		template <typename T, typename Assign> 
		void do_assign(Map<Matrix<T>>& dest, const ArrayConstReference<T,2> & m, const Assign& assign_fn);


		// Map<Array>  = Array
		template <typename T,int D, typename Assign> 
		void do_assign(Map<Array<T,D>>& dest, const Array<T,D> & m, const Assign& assign_fn);
		
		// Map<Array> = Array
		template <typename T,int D, typename Assign> 
		void do_assign(Map<Array<T,D>>& dest, const ArrayConstReference<T,D> & m, const Assign& assign_fn);
		

		template <typename E, typename Dest,typename Assign> 
		void do_assign(Dest& dest, const E & e, const Assign& assign_fn);
		
		template <typename E,typename Dest,typename Assign> 
		void do_assign(NoAliasProxy<Dest> & dest, const E & expr, const Assign& assign_fn);

		
		template <typename E,typename Dest,typename Func> 
		void do_compound_assign(Dest & dest, const E & expr, const Func& fn);
		
		template <typename POD,typename Dest,typename Func> 
		void do_scalar_compound_assign(Dest& dest, POD alpha, const Func& fn);

		template <typename E>
		typename E::value_type squaredNorm(const E & m);
		
		template <typename E>
		typename E::value_type sum(const E & m);
		
		template <typename E>
		typename E::value_type min(const E & m);
		
		template <typename E>
		typename E::value_type max(const E & m);
		
		template <typename E1, typename E2>
		typename E1::value_type dot(const E1 & v1, const E2 & v2);
		
	
		template <typename E>
		typename XprResultType<E>:: result_type eval(const E & expr) ;
	}
}
#endif