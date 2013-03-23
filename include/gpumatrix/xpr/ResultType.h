#ifndef XPR_RESULT_TYPE_H
#define XPR_RESULT_TYPE_H

#include <gpumatrix/Functional.h>

namespace gpumatrix 
{
	template <class T> class Matrix;
	template <class T> class Vector;
	template <class T, int D> class Array;
	template <class T> class MatrixConstReference;
	template <class T> class VectorConstReference;
	template <class T, int D> class ArrayConstReference;
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

	template<typename E>	class RowWiseSum;
	template<typename E>	class ColWiseSum;

	template <class E> class XprResultType;
	template <class E> class XprMatrixTranspose;

	template<typename T> 
	class XprResultType<MatrixConstReference<T>>
	{
	public:
		typedef MatrixConstReference<T> result_type;
	};

	template<typename T> 
	class XprResultType<VectorConstReference<T>>
	{
	public:
		typedef VectorConstReference<T> result_type;
	};

		template<typename T, int D> 
	class XprResultType<ArrayConstReference<T,D>>
	{
	public:
		typedef ArrayConstReference<T,D> result_type;
	};

	template<typename T> 
	class XprResultType<XprLiteral<T>>
	{
	public:
		typedef T result_type;
	};

	template<typename UnOp, typename E> 
	class XprResultType<XprUnOp<UnOp, XprMatrix<E>>>
	{
	public:
		typedef Matrix<typename E::value_type> result_type;
	};

	template<typename UnOp, typename E> 
	class XprResultType<XprUnOp<UnOp, XprVector<E>>>
	{
	public:
		typedef Vector<typename E::value_type> result_type;
	};

		template<typename UnOp, typename E, int D> 
	class XprResultType<XprUnOp<UnOp, XprArray<E,D>>>
	{
	public:
		typedef Array<typename E::value_type,D> result_type;
	};

	template<typename BinOp, typename E1, typename E2> 
	class XprResultType<XprBinOp<BinOp, XprMatrix<E1>, XprMatrix<E2>>>
	{
	public:
		typedef Matrix<typename BinOp::value_type> result_type;
	};

	template<typename BinOp, typename E1, typename E2> 
	class XprResultType<XprBinOp<BinOp, XprVector<E1>, XprVector<E2>>>
	{
	public:
		typedef Vector<typename BinOp::value_type> result_type;
	};

	template<typename BinOp, typename T, typename E> 
	class XprResultType<XprBinOp<BinOp, XprLiteral<T>, XprMatrix<E>>>
	{
	public:
		typedef Matrix<typename E::value_type> result_type;
	};

	template<typename BinOp, typename T, typename E> 
	class XprResultType<XprBinOp<BinOp, XprMatrix<E>, XprLiteral<T>>>
	{
	public:
		typedef Matrix<typename E::value_type> result_type;
	};
	
		template<typename BinOp, typename T, typename E> 
	class XprResultType<XprBinOp<BinOp, XprLiteral<T>, XprVector<E>>>
	{
	public:
		typedef Vector<typename E::value_type> result_type;
	};

	template<typename BinOp, typename T, typename E> 
	class XprResultType<XprBinOp<BinOp, XprVector<E>, XprLiteral<T>>>
	{
	public:
		typedef Vector<typename E::value_type> result_type;
	};

	template<template <class, class> class Xpr, typename E1, typename E2>
	class XprResultType<Xpr<XprMatrix<E1>, XprMatrix<E2> > >
	{
	public:
		typedef Matrix<typename E1::value_type> result_type; 
	};

	template<template <class, class> class Xpr, typename E1, typename E2>
	class XprResultType<Xpr<XprMatrix<E1>, XprVector<E2>>>
	{
	public:
		typedef Vector<typename E1::value_type> result_type; 
	};

		template<template <class, class> class Xpr, typename E1, typename E2>
	class XprResultType<Xpr<XprVector<E1>, XprVector<E2>>>
	{
	public:
		typedef Vector<typename E1::value_type> result_type; 
	};


	template< typename OP, typename E1, typename E2, int D>
	class XprResultType<XprBinOp<OP, XprArray<E1,D>, XprArray<E2,D>>>
	{
	public:
		typedef Array<typename E1::value_type,D> result_type; 
	};

		template< typename OP, typename POD, typename E, int D>
	class XprResultType<XprBinOp<OP, XprLiteral<POD>, XprArray<E,D>>>
	{
	public:
		typedef Array<typename E::value_type,D> result_type; 
	};

	template< typename OP, typename POD, typename E, int D>
	class XprResultType<XprBinOp<OP, XprArray<E,D>, XprLiteral<POD>>>
	{
	public:
		typedef Array<typename E::value_type,D> result_type; 
	};
	//template<typename E1, typename E2>
	//class XprResultType<XprMMProduct<E1,E2>>
	//{
	//public:
	//	typedef Matrix<typename E1::value_type> result_type;
	//};

	//template<typename E1, typename E2>
	//class XprResultType<XprMMtProduct<E1,E2>>
	//{
	//public:
	//	typedef Matrix<typename E1::value_type> result_type;
	//};

	//	template<typename E1, typename E2>
	//class XprResultType<XprMtMProduct<E1,E2>>
	//{
	//public:
	//	typedef Matrix<typename E1::value_type> result_type;
	//};

	//template<typename E1, typename E2>
	//class XprResultType<XprMtMtProduct<E1,E2>>
	//{
	//public:
	//	typedef Matrix<typename E1::value_type> result_type;
	//};

	//template<typename E1, typename E2>
	//class XprResultType<XprMVProduct<E1,E2>>
	//{
	//public:
	//	typedef Vector<typename E1::value_type> result_type;
	//};

	//template<typename E1, typename E2>
	//class XprResultType<XprMVProduct<E1,E2>>
	//{
	//public:
	//	typedef Vector<typename E1::value_type> result_type;
	//};
	//
	template<typename E>
	class XprResultType<XprMatrixTranspose<E>>
	{
	public:
		typedef Matrix<typename E::value_type> result_type;
	};

	template<typename E>
	class XprResultType<RowWiseSum<E>>
	{
	public:
		typedef Vector<typename E::value_type> result_type;
	};

	template<typename E>
	class XprResultType<ColWiseSum<E>>
	{
	public:
		typedef Vector<typename E::value_type> result_type;
	};

	template<typename E, int D >
	class XprResultType<XprUnOp<Fcnl_exp<typename E::value_type>,XprArray<E,D> > >
	{
	public:
		typedef Array<typename E::value_type,D> result_type;
	};

		template<typename E, int D >
	class XprResultType<XprUnOp<Fcnl_arrayinv<typename E::value_type>,XprArray<E,D> > >
	{
	public:
		typedef Array<typename E::value_type,D> result_type;
	};


	template<typename E >
	class XprResultType<XprUnOp<Fcnl_sqnorm<typename E::value_type>,E > >
	{
	public:
		typedef typename E::value_type result_type;
	};
	

}

#endif