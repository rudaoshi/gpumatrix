#ifndef EVAL_INTERFACE_H
#define EVAL_INTERFACE_H

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
	
	template <class T> class XprMatrix;
	template <class T> class XprVector;
	template <class T, int D> class XprArray;
	template <class T> class XprLiteral;
	template<typename BinOp, typename E1, typename E2> class XprBinOp;
	template<typename UnOp, typename E> class XprUnOp;

	template<typename E1, typename E2>	class XprMMProduct;
	template<typename E1, typename E2>	class XprMMtProduct;
	template<typename E1, typename E2>	class XprMtMProduct;
	template<typename E1, typename E2>	class XprMtMtProduct;
	template<typename E1, typename E2>	class XprMVProduct;
	template<typename E1, typename E2>	class XprMtVProduct;

	template<typename E>	class RowWiseSum;
	template<typename E>	class ColWiseSum;

	template <class E> class XprResultType;
	template <class E> class XprMatrixTranspose;

	namespace impl
	{
		template <typename T>
		MatrixConstReference<T> eval(const MatrixConstReference<T> & m) ;
		template <typename T>
		VectorConstReference<T> eval(const VectorConstReference<T> & m) ;

		template <typename T, int D>
		ArrayConstReference<T,D> eval(const ArrayConstReference<T,D> & m) ;


		template <typename E> 
		typename XprResultType<E>::result_type eval(const E & expr) ;
	
#pragma region algebra operation

		// Dest = Matrix.transpose()
		template <typename E,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMatrixTranspose<E> & trans, const Assign& assign_fn);

		// Dest = M1*M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMMProduct<E1,E2> & prod, const Assign& assign_fn);

		// Dest = M1.transpose()*M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtMProduct<E1,E2> & prod, const Assign& assign_fn);

		// Dest = M1*M2.transpose()
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMMtProduct<E1,E2> & prod, const Assign& assign_fn);

		// Dest = M1.transpose()*M2.transpose()
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtMtProduct<E1,E2> & prod, const Assign& assign_fn);

		// Dest = M1*V2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMVProduct<E1,E2> & prod, const Assign& assign_fn);

		// Dest = M1.transpose()*V2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, const XprMtVProduct<E1,E2> & prod, const Assign& assign_fn);

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
			const Assign& assign_fn);

		// Dest = Matrix*alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprMatrix<E>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn);

		// Dest =  Matrix /alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprMatrix<E>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn);

		// Dest = alpha*Vector
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprVector<E>
			> & expr, 
			const Assign& assign_fn);

		// Dest = Vector*alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprVector<E>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn);

		// Dest =  Vector /alpha
		template <typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprVector<E>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn);

		// Dest = M1+M2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprMatrix<E1>,							
					XprMatrix<E2>
			> & expr, 
			const Assign& assign_fn);

		// Dest = M1 - M2
		template < typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprMatrix<E1>,							
					XprMatrix<E2>
			> & expr, 
			const Assign& assign_fn);

		// Dest = V1+V2
		template < typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprVector<E1>,							
					XprVector<E2>
			> & expr, 
			const Assign& assign_fn);

		// Dest = v1 - v2
		template <typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprVector<E1>,							
					XprVector<E2>
			> & expr, 
			const Assign& assign_fn);



					// Dest = alpha*Array
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn);
		
		// Dest = Array*alpha
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn);

		// Dest = M1+M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn);

		// Dest = M1 - M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn);

		// Dest = alpha + Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn);

		// Dest = Array + alpha
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_add<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn);

		// Dest = alpha - Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn);

		// Dest = Array - alpha
		template <int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_sub<typename E::value_type,POD>,			
					XprArray<E,D>,
					XprLiteral< POD >
			> & expr, 
			const Assign& assign_fn);

		// Dest =  Array /alpha
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E::value_type,POD>,
					XprArray<E,D>,
					XprLiteral< POD >
					
			> & expr, 
			const Assign& assign_fn);


		// Dest = alpha / Array
		template < int D, typename POD,typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<POD,typename E::value_type>,
					XprLiteral< POD >,							
					XprArray<E,D>
			> & expr, 
			const Assign& assign_fn);


#pragma endregion


		// Dest = M1 * M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_mul<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn);

		// Dest = M1 / M2
		template < int D, typename E1, typename E2,typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprBinOp<
					Fcnl_div<typename E1::value_type,typename E2::value_type>,
					XprArray<E1,D>,							
					XprArray<E2,D>
			> & expr, 
			const Assign& assign_fn);


		// Dest = M.rowwise().sum()
		template <typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const RowWiseSum<E> & expr, 
			const Assign& assign_fn);

		// Dest = M.colwise().sum()
		template <typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const ColWiseSum<E> & expr, 
			const Assign& assign_fn);

		// Dest = - M
		template <typename UnOP, typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprMatrix<E> > & expr, 
			const Assign& assign_fn);

		// Dest = M.exp()
		template <typename UnOP, typename E, int D, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprArray<E,D> > & expr, 
			const Assign& assign_fn);

		// Dest = -M
		template <typename UnOP, typename E, typename Dest,typename Assign> 
		void eval(Dest& dest, 
			const XprUnOp<UnOP,XprVector<E> > & expr, 
			const Assign& assign_fn);

		
	}
	

}

#endif