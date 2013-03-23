/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2007 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: MatrixFunctions.h,v 1.65 2007-06-23 15:58:58 opetzold Exp $
 */

#ifndef TVMET_MATRIX_FUNCTIONS_H
#define TVMET_MATRIX_FUNCTIONS_H

#include <gpumatrix/Extremum.h>
#include <gpumatrix/NumericTraits.h>
#include <gpumatrix/xpr/Vector.h>

namespace gpumatrix {

/* forwards */
template<class T> class Vector;
template<class T> class VectorConstReference;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Matrix<T1>, Matrix<T2>)
 * function(XprMatrix<E>, Matrix<T>)
 * function(Matrix<T>, XprMatrix<E>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class T1, class T2>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    XprMatrix<MatrixConstReference<T1>>,				\
    XprMatrix<MatrixConstReference<T2>>				\
  >								\
>									\
NAME (const Matrix<T1>& lhs,				\
      const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, class T>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprMatrix<E>,						\
    XprMatrix<MatrixConstReference<T>>					\
  >								\
>									\
NAME (const XprMatrix<E>& lhs,				\
      const Matrix<T>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class T, class E>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprMatrix<MatrixConstReference<T>>,				\
    XprMatrix<E>						\
  >								\
>									\
NAME (const Matrix<T>& lhs,					\
      const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
  TVMET_DECLARE_MACRO(div)			// not defined for matrizes
}

#undef TVMET_DECLARE_MACRO


/*
 * function(Matrix<T>, POD)
 * function(POD, Matrix<T>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class T>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    XprMatrix<MatrixConstReference<T>>,				\
    XprLiteral<POD >							\
  >								\
>									\
NAME (const Matrix<T>& lhs, 				\
      POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
									\
template<class T>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprMatrix<MatrixConstReference<T>>					\
  >								\
>									\
NAME (POD lhs, 								\
      const Matrix<T>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, int)
TVMET_DECLARE_MACRO(sub, int)
TVMET_DECLARE_MACRO(mul, int)
TVMET_DECLARE_MACRO(div, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(add, long long int)
TVMET_DECLARE_MACRO(sub, long long int)
TVMET_DECLARE_MACRO(mul, long long int)
TVMET_DECLARE_MACRO(div, long long int)
#endif

TVMET_DECLARE_MACRO(add, float)
TVMET_DECLARE_MACRO(sub, float)
TVMET_DECLARE_MACRO(mul, float)
TVMET_DECLARE_MACRO(div, float)

TVMET_DECLARE_MACRO(add, double)
TVMET_DECLARE_MACRO(sub, double)
TVMET_DECLARE_MACRO(mul, double)
TVMET_DECLARE_MACRO(div, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(add, long double)
TVMET_DECLARE_MACRO(sub, long double)
TVMET_DECLARE_MACRO(mul, long double)
TVMET_DECLARE_MACRO(div, long double)
#endif

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * function(Matrix<T>, complex<T>)
 * function(complex<T>, Matrix<T>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)						\
template<class T>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    MatrixConstReference< std::complex<T>>,				\
    XprLiteral<std::complex<T> >						\
  >									\
>										\
NAME (const Matrix< std::complex<T>>& lhs,				\
      const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;			\
										\
template<class T>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    XprLiteral< std::complex<T> >,						\
    MatrixConstReference< std::complex<T>>				\
  >									\
>										\
NAME (const std::complex<T>& lhs,						\
      const Matrix< std::complex<T>>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)
TVMET_DECLARE_MACRO(sub)
TVMET_DECLARE_MACRO(mul)
TVMET_DECLARE_MACRO(div)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1,
	 class T2>
XprMatrix<
  XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>> 		// M2(Cols1)
  >/*	*/						// return Dim
>
prod(const Matrix<T1>& lhs,
     const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1,	 class T2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,			// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Cols1)
  >/*	*/						// return Dim
>
prod(const XprMatrix<E1>& lhs,
     const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1,
	 class E2>
XprMatrix<
  XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<E2>				// M2(Cols1)
  >							// return Dim
>
prod(const Matrix<T1>& lhs,
     const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1,
	 class T2>
XprMatrix<
  XprMtMtProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Cols1)
  >							// return Dim
>
MtMt_prod(const Matrix<T1>& lhs,
	   const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1,
	 class T2>	// Rows2 = Rows1
XprMatrix<
  XprMtMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Rows1)
  >							// return Dim
>
MtM_prod(const Matrix<T1>& lhs,
	 const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1,
	 class T2>
XprMatrix<
  XprMMtProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>> 		// M2(Rows2)
  >							// return Dim
>
MMt_prod(const Matrix<T1>& lhs,
	 const Matrix<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1, class T2>
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
	XprVector<VectorConstReference<T2>>
  >
>
prod(const Matrix<T1>& lhs,
     const Vector<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class T1, class T2>
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
	XprVector<VectorConstReference<T2>>
  >
>
prod(const Matrix<T1>& lhs,
     const Map<Vector<T2>>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class T1, class E2>
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<E2>
  >
>
prod(const Matrix<T1>& lhs,
     const XprVector<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, class T2>
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
	XprVector<VectorConstReference<T2>>
  >
>
prod(const XprMatrix<E1>& lhs,
     const Vector<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class E1, class T2>
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
	XprVector<VectorConstReference<T2>>
  >
>
prod(const XprMatrix<E1>& lhs,
     const Map<Vector<T2>>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class T1, class T2>
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,	   // M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const Matrix<T1>& lhs,
	 const Vector<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class T1, class T2>
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,	   // M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const Matrix<T1>& lhs,
	 const Map<Vector<T2>>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class T1, class E2>
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<E2>
  >
>
MtV_prod(const Matrix<T1>& lhs,
     const XprVector<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, class T2>
XprVector<
  XprMtVProduct<
    XprMatrix<E1>,
	XprVector<VectorConstReference<T2>>
  >
>
MtV_prod(const XprMatrix<E1>& lhs,
     const Vector<T2>& rhs) TVMET_CXX_ALWAYS_INLINE;

template<class E1, class T2>
XprVector<
  XprMtVProduct<
    XprMatrix<E1>,
	XprVector<VectorConstReference<T2>>
  >
>
MtV_prod(const XprMatrix<E1>& lhs,
     const Map<Vector<T2>>& rhs) TVMET_CXX_ALWAYS_INLINE;
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T>
XprMatrix<
  XprMatrixTranspose<
    XprMatrix<MatrixConstReference<T>>
  >
>
trans(const Matrix<T>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T>
typename NumericTraits<T>::sum_type
trace(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


template<class T>
XprVector<
  XprMatrixRow<
    XprMatrix<MatrixConstReference<T>>
  >
>
row(const Matrix<T>& m,
    std::size_t no) TVMET_CXX_ALWAYS_INLINE;


template<class T>
XprVector<
  XprMatrixCol<
    XprMatrix<MatrixConstReference<T>>
  >
>
col(const Matrix<T>& m,
    std::size_t no) TVMET_CXX_ALWAYS_INLINE;


template<class T>
XprVector<
  XprMatrixDiag<
    XprMatrix<MatrixConstReference<T>>
  >
>
diag(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * min/max unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E>
Extremum<typename E::value_type, std::size_t, matrix_tag>
maximum(const XprMatrix<E>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T>
Extremum<T, std::size_t, matrix_tag>
maximum(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E>
Extremum<typename E::value_type, std::size_t, matrix_tag>
minimum(const XprMatrix<E>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T>
Extremum<T, std::size_t, matrix_tag>
minimum(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E>
typename E::value_type
max(const XprMatrix<E>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T>
T max(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E>
typename E::value_type
min(const XprMatrix<E>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T>
T min(const Matrix<T>& m) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * other unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T>
XprMatrix<
  XprIdentity<T>
>
identity() TVMET_CXX_ALWAYS_INLINE;


template<class M>
XprMatrix<
  XprIdentity<
    typename M::value_type
    >
>
identity() TVMET_CXX_ALWAYS_INLINE;


template<class T>
XprMatrix<
  XprMatrix<MatrixConstReference<T>>
>
cmatrix_ref(const T* mem) TVMET_CXX_ALWAYS_INLINE;


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Matrix<T1>, Matrix<T2>)
 * function(XprMatrix<E>, Matrix<T>)
 * function(Matrix<T>, XprMatrix<E>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						\
template<class T1, class T2>		\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    XprMatrix<MatrixConstReference<T1>>,					\
    XprMatrix<MatrixConstReference<T2>>					\
  >									\
>										\
NAME (const Matrix<T1>& lhs, const Matrix<T2>& rhs) {	\
  typedef XprBinOp <								\
    Fcnl_##NAME<T1, T2>,							\
    XprMatrix<MatrixConstReference<T1>>,					\
    XprMatrix<MatrixConstReference<T2>>					\
  >							expr_type;		\
  return XprMatrix<expr_type>(					\
    expr_type(lhs.as_expr(), rhs.as_expr()));				\
}										\
										\
template<class E, class T>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E>,							\
    XprMatrix<MatrixConstReference<T>>						\
  >									\
>										\
NAME (const XprMatrix<E>& lhs, const Matrix<T>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E>,							\
    XprMatrix<MatrixConstReference<T>>						\
  > 							 expr_type;		\
  return XprMatrix<expr_type>(					\
    expr_type(lhs, rhs.as_expr()));						\
}										\
										\
template<class T, class E>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<MatrixConstReference<T>>,					\
    XprMatrix<E>							\
  >									\
>										\
NAME (const Matrix<T>& lhs, const XprMatrix<E>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,					\
    XprMatrix<MatrixConstReference<T>>,					\
    XprMatrix<E>							\
  >	 						 expr_type;		\
  return XprMatrix<expr_type>(					\
    expr_type(lhs.as_expr(), rhs));						\
}

TVMET_IMPLEMENT_MACRO(add)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul)			// not defined for matrizes
  TVMET_IMPLEMENT_MACRO(div)			// not defined for matrizes
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * function(Matrix<T>, POD)
 * function(POD, Matrix<T>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class T>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    XprMatrix<MatrixConstReference<T>>,				\
    XprLiteral<POD >							\
  >								\
>									\
NAME (const Matrix<T>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<T, POD >,						\
    XprMatrix<MatrixConstReference<T>>,				\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs.as_expr(), XprLiteral< POD >(rhs)));		\
}									\
									\
template<class T>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprMatrix<MatrixConstReference<T>>					\
  >								\
>									\
NAME (POD lhs, const Matrix<T>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprMatrix<MatrixConstReference<T>>					\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< POD >(lhs), rhs.as_expr()));		\
}

TVMET_IMPLEMENT_MACRO(add, int)
TVMET_IMPLEMENT_MACRO(sub, int)
TVMET_IMPLEMENT_MACRO(mul, int)
TVMET_IMPLEMENT_MACRO(div, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(add, long long int)
TVMET_IMPLEMENT_MACRO(sub, long long int)
TVMET_IMPLEMENT_MACRO(mul, long long int)
TVMET_IMPLEMENT_MACRO(div, long long int)
#endif

TVMET_IMPLEMENT_MACRO(add, float)
TVMET_IMPLEMENT_MACRO(sub, float)
TVMET_IMPLEMENT_MACRO(mul, float)
TVMET_IMPLEMENT_MACRO(div, float)

TVMET_IMPLEMENT_MACRO(add, double)
TVMET_IMPLEMENT_MACRO(sub, double)
TVMET_IMPLEMENT_MACRO(mul, double)
TVMET_IMPLEMENT_MACRO(div, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(add, long double)
TVMET_IMPLEMENT_MACRO(sub, long double)
TVMET_IMPLEMENT_MACRO(mul, long double)
TVMET_IMPLEMENT_MACRO(div, long double)
#endif

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * function(Matrix<T>, complex<T>)
 * function(complex<T>, Matrix<T>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class T>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    MatrixConstReference< std::complex<T>>,			\
    XprLiteral<std::complex<T> >					\
  >								\
>									\
NAME (const Matrix< std::complex<T>>& lhs,			\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    MatrixConstReference< std::complex<T>>,			\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs.as_expr(), XprLiteral< std::complex<T> >(rhs)));	\
}									\
									\
template<class T>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    MatrixConstReference< std::complex<T>>			\
  >								\
>									\
NAME (const std::complex<T>& lhs,					\
      const Matrix< std::complex<T>>& rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    MatrixConstReference<std::complex<T>>			\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs.as_expr()));	\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs)
 * \brief Function for the matrix-matrix-product.
 * \ingroup _binary_function
 * \note The rows2 has to be equal to cols1.
 */
template<class T1,
	 class T2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>> 		// M2(Cols1)
  >							// return Dim
>
prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprMatrix<MatrixConstReference<T2>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}


/**
 * \fn prod(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs)
 * \brief Evaluate the product of XprMatrix and Matrix.
 * \ingroup _binary_function
 */
template<class E1,
	 class T2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,			// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Cols1)
  >							// return Dim
>
prod(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<E1>,
    XprMatrix<MatrixConstReference<T2>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs, rhs.as_expr()));
}


/**
 * \fn prod(const Matrix<T1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Evaluate the product of Matrix and XprMatrix.
 * \ingroup _binary_function
 */
template<class T1,
	 class E2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<E2>				// M2(Cols1)
  >							// return Dim
>
prod(const Matrix<T1>& lhs, const XprMatrix<E2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprMatrix<E2>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs.as_expr(), rhs));
}


/**
 * \fn trans_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs)
 * \brief Function for the trans(matrix-matrix-product)
 * \ingroup _binary_function
 * Perform on given Matrix M1 and M2:
 * \f[
 * (M_1\,M_2)^T
 * \f]
 */
template<class T1,
	 class T2>
inline
XprMatrix<
  XprMtMtProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Cols1)
  >							// return Dim
>
MtMt_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs) {
  typedef XprMtMtProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprMatrix<MatrixConstReference<T2>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}


/**
 * \fn MtM_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs)
 * \brief Function for the trans(matrix)-matrix-product.
 * \ingroup _binary_function
 *        using formula
 *        \f[
 *        M_1^{T}\,M_2
 *        \f]
 * \note The number of cols of matrix 2 have to be equal to number of rows of
 *       matrix 1, since matrix 1 is trans - the result is a (Cols1 x Cols2)
 *       matrix.
 */
template<class T1,
	 class T2>	// Rows2 = Rows1
inline
XprMatrix<
  XprMtMProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>>		// M2(Rows1)
  >							// return Dim
>
MtM_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs) {
  typedef XprMtMProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprMatrix<MatrixConstReference<T2>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}


/**
 * \fn MMt_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs)
 * \brief Function for the matrix-trans(matrix)-product.
 * \ingroup _binary_function
 * \note The Cols2 has to be equal to Cols1.
 */
template<class T1,
	 class T2>
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<MatrixConstReference<T1>>,	// M1(Rows1)
    XprMatrix<MatrixConstReference<T2>> 		// M2(Rows2)
  >							// return Dim
>
MMt_prod(const Matrix<T1>& lhs, const Matrix<T2>& rhs) {
  typedef XprMMtProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprMatrix<MatrixConstReference<T2>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const Matrix<T1>& lhs, const Vector<T2>& rhs)
 * \brief Function for the matrix-vector-product
 * \ingroup _binary_function
 */
template<class T1, class T2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  >
>
prod(const Matrix<T1>& lhs, const Vector<T2>& rhs) {
  typedef XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}

template<class T1, class T2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  >
>
prod(const Matrix<T1>& lhs, const Map<Vector<T2>>& rhs) {
  typedef XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}


/**
 * \fn prod(const Matrix<T1>& lhs, const XprVector<E2>& rhs)
 * \brief Function for the matrix-vector-product
 * \ingroup _binary_function
 */
template<class T1, class E2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<E2>
  >
>
prod(const Matrix<T1>& lhs, const XprVector<E2>& rhs) {
  typedef XprMVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<E2>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs));
}


/*
 * \fn prod(const XprMatrix<E>& lhs, const Vector<T>& rhs)
 * \brief Compute the product of an XprMatrix with a Vector.
 * \ingroup _binary_function
 */
template<class E1, class T2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							
>
prod(const XprMatrix<E1>& lhs, const Vector<T2>& rhs) {
  typedef XprMVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs, rhs.as_expr()));
}

template<class E1, class T2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							
>
prod(const XprMatrix<E1>& lhs, const Map<Vector<T2>>& rhs) {
  typedef XprMVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs, rhs.as_expr()));
}

/**
 * \fn Mtx_prod(const Matrix<T1>& matrix, const Vector<T2>& vector)
 * \brief Function for the trans(matrix)-vector-product
 * \ingroup _binary_function
 * Perform on given Matrix M and vector x:
 * \f[
 * M^T\, x
 * \f]
 */
template<class T1, class T2>
inline
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,// M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const Matrix<T1>& lhs, const Vector<T2>& rhs) {
  typedef XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}

template<class T1, class T2>
inline
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,// M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const Matrix<T1>& lhs, const Map<Vector<T2>>& rhs) {
  typedef XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs.as_expr()));
}

template<class E1, class T2>
inline
XprVector<
  XprMtVProduct<
    XprMatrix<E1>,// M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const XprMatrix<E1>& lhs, const Vector<T2>& rhs) {
  typedef XprMtVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs, rhs.as_expr()));
}

template<class E1, class T2>
inline
XprVector<
  XprMtVProduct<
    XprMatrix<E1>,// M(Rows)
    XprVector<VectorConstReference<T2>> 			// V
  >
>
MtV_prod(const XprMatrix<E1>& lhs, const Map<Vector<T2>>& rhs) {
  typedef XprMtVProduct<
    XprMatrix<E1>,
    XprVector<VectorConstReference<T2>>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs, rhs.as_expr()));
}

template<class T1, class E2>
inline
XprVector<
  XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,// M(Rows)
    XprVector<VectorConstReference<E2>> 			// V
  >
>
MtV_prod(const Matrix<T1>& lhs, const XprVector<E2>& rhs) {
  typedef XprMtVProduct<
    XprMatrix<MatrixConstReference<T1>>,
    XprVector<E2>
  > 							expr_type;
  return XprVector<expr_type>(
    expr_type(lhs.as_expr(), rhs));
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn trans(const Matrix<T>& rhs)
 * \brief Transpose the matrix
 * \ingroup _unary_function
 */
template<class T>
inline
XprMatrix<
  XprMatrixTranspose<
    XprMatrix<MatrixConstReference<T>>
  >
>
trans(const Matrix<T>& rhs) {
  typedef XprMatrixTranspose<
    XprMatrix<MatrixConstReference<T>>
  >							expr_type;
  return XprMatrix<expr_type>(
    expr_type(rhs.as_expr()));
}


/*
 * \fn trace(const Matrix<T>& m)
 * \brief Compute the trace of a square matrix.
 * \ingroup _unary_function
 *
 * Simply compute the trace of the given matrix as:
 * \f[
 *  \sum_{k = 0}^{Sz-1} m(k, k)
 * \f]
 */
// template<class T>
// inline
// typename NumericTraits<T>::sum_type
// trace(const Matrix<T>& m) {
//   return meta::Matrix<Sz, 0, 0>::trace(m);
// }


/**
 * \fn row(const Matrix<T>& m, std::size_t no)
 * \brief Returns a row vector of the given matrix.
 * \ingroup _binary_function
 */
template<class T>
inline
XprVector<
  XprMatrixRow<
    XprMatrix<MatrixConstReference<T>>
  >
>
row(const Matrix<T>& m, std::size_t no) {
  typedef XprMatrixRow<
    XprMatrix<MatrixConstReference<T>>
  >							expr_type;
  return XprVector<expr_type>(expr_type(m.as_expr(), no));
}


/**
 * \fn col(const Matrix<T>& m, std::size_t no)
 * \brief Returns a column vector of the given matrix.
 * \ingroup _binary_function
 */
template<class T>
inline
XprVector<
  XprMatrixCol<
    XprMatrix<MatrixConstReference<T>>
  >
>
col(const Matrix<T>& m, std::size_t no) {
  typedef XprMatrixCol<
    XprMatrix<MatrixConstReference<T>>
  >							expr_type;
  return XprVector<expr_type>(expr_type(m.as_expr(), no));
}


/**
 * \fn diag(const Matrix<T>& m)
 * \brief Returns the diagonal vector of the given square matrix.
 * \ingroup _unary_function
 */
template<class T>
inline
XprVector<
  XprMatrixDiag<
    XprMatrix<MatrixConstReference<T>>
  >
>
diag(const Matrix<T>& m) {
  typedef XprMatrixDiag<
    XprMatrix<MatrixConstReference<T>>
  >							expr_type;
  return XprVector<expr_type>(expr_type(m.as_expr()));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * min/max unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn maximum(const XprMatrix<E>& e)
 * \brief Find the maximum of a matrix expression
 * \ingroup _unary_function
 */
// template<class E>
// inline
// Extremum<typename E::value_type, std::size_t, matrix_tag>
// maximum(const XprMatrix<E>& e) {
//   typedef typename E::value_type 			value_type;
// 
//   value_type 						temp(e(0, 0));
//   std::size_t 						row_no(0), col_no(0);
// 
//   for(std::size_t i = 0; i != Rows; ++i) {
//     for(std::size_t j = 0; j != Cols; ++j) {
//       if(e(i, j) > temp) {
// 	temp = e(i, j);
// 	row_no = i;
// 	col_no = j;
//       }
//     }
//   }
// 
//   return Extremum<value_type, std::size_t, matrix_tag>(temp, row_no, col_no);
// }


/**
 * \fn maximum(const Matrix<T>& m)
 * \brief Find the maximum of a matrix
 * \ingroup _unary_function
 */
template<class T>
inline
Extremum<T, std::size_t, matrix_tag>
maximum(const Matrix<T>& m) { return maximum(m.as_expr()); }


/**
 * \fn minimum(const XprMatrix<E>& e)
 * \brief Find the minimum of a matrix expression
 * \ingroup _unary_function
 */
// template<class E>
// inline
// Extremum<typename E::value_type, std::size_t, matrix_tag>
// minimum(const XprMatrix<E>& e) {
//   typedef typename E::value_type 			value_type;
// 
//   value_type 						temp(e(0, 0));
//   std::size_t 						row_no(0), col_no(0);
// 
//   for(std::size_t i = 0; i != Rows; ++i) {
//     for(std::size_t j = 0; j != Cols; ++j) {
//       if(e(i, j) < temp) {
// 	temp = e(i, j);
// 	row_no = i;
// 	col_no = j;
//       }
//     }
//   }
// 
//   return Extremum<value_type, std::size_t, matrix_tag>(temp, row_no, col_no);
// }


/**
 * \fn minimum(const Matrix<T>& m)
 * \brief Find the minimum of a matrix
 * \ingroup _unary_function
 */
template<class T>
inline
Extremum<T, std::size_t, matrix_tag>
minimum(const Matrix<T>& m) { return minimum(m.as_expr()); }


/**
 * \fn max(const XprMatrix<E>& e)
 * \brief Find the maximum of a matrix expression
 * \ingroup _unary_function
 */
// template<class E>
// inline
// typename E::value_type
// max(const XprMatrix<E>& e) {
//   typedef typename E::value_type 			value_type;
// 
//   value_type 						temp(e(0, 0));
// 
//   for(std::size_t i = 0; i != Rows; ++i)
//     for(std::size_t j = 0; j != Cols; ++j)
//       if(e(i, j) > temp)
// 	temp = e(i, j);
// 
//   return temp;
// }


/**
 * \fn max(const Matrix<T>& m)
 * \brief Find the maximum of a matrix
 * \ingroup _unary_function
 */
template<class T>
inline
T max(const Matrix<T>& m) {
  typedef T			 			value_type;
  typedef typename Matrix<
   T
  >::const_iterator					const_iterator;

  const_iterator					iter(m.begin());
  const_iterator					last(m.end());
  value_type 						temp(*iter);

  for( ; iter != last; ++iter)
    if(*iter > temp)
      temp = *iter;

  return temp;
}


/**
 * \fn min(const XprMatrix<E>& e)
 * \brief Find the minimum of a matrix expression
 * \ingroup _unary_function
 */
// template<class E>
// inline
// typename E::value_type
// min(const XprMatrix<E>& e) {
//   typedef typename E::value_type			value_type;
// 
//   value_type 						temp(e(0, 0));
// 
//   for(std::size_t i = 0; i != Rows; ++i)
//     for(std::size_t j = 0; j != Cols; ++j)
//       if(e(i, j) < temp)
// 	temp = e(i, j);
// 
//   return temp;
// }


/**
 * \fn min(const Matrix<T>& m)
 * \brief Find the minimum of a matrix
 * \ingroup _unary_function
 */
template<class T>
inline
T min(const Matrix<T>& m) {
  typedef T			 			value_type;
  typedef typename Matrix<
   T
  >::const_iterator					const_iterator;

  const_iterator					iter(m.begin());
  const_iterator					last(m.end());
  value_type 						temp(*iter);

  for( ; iter != last; ++iter)
    if(*iter < temp)
      temp = *iter;

  return temp;
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * other unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn XprMatrix<XprIdentity<typename M::value_type, M::Rows, M::Cols>, M::Rows, M::Cols>identity()
 * \brief Fill a matrix to an identity matrix.
 * \ingroup _unary_function
 *
 * \note The matrix doesn't need to be square. Only the elements
 *       where the current number of rows are equal to columns
 *       will be set to 1, else to 0.
 *
 * \par Usage:
 * \code
 * typedef Matrix<double,3,3>		matrix_type;
 * ...
 * matrix_type E( identity<double, 3, 3>() );
 * \endcode
 *
 * Note, we have to specify the type, number of rows and columns
 * since ADL can't work here.
 *
 *
 *
 * \since release 1.6.0
 */
template<class T>
inline
XprMatrix<
  XprIdentity<T>
>
identity() {
  typedef XprIdentity<T>		expr_type;

  return XprMatrix<expr_type>(expr_type());
}

/**
 * \fn XprMatrix<XprIdentity<typename M::value_type, M::Rows, M::Cols>, M::Rows, M::Cols>identity()
 * \brief Fill a matrix to an identity matrix (convenience wrapper
 *        for matrix typedefs).
 * \ingroup _unary_function
 *
 * \note The matrix doesn't need to be square. Only the elements
 *       where the current number of rows are equal to columns
 *       will be set to 1, else to 0.
 *
 * \par Usage:
 * \code
 * typedef Matrix<double,3,3>		matrix_type;
 * ...
 * matrix_type E( identity<matrix_type>() );
 * \endcode
 *
 * Note, we have to specify the matrix type, since ADL can't work here.
 *
 * \since release 1.6.0
 */
template<class M>
inline
XprMatrix<
  XprIdentity<
    typename M::value_type>
>
identity() {
  return identity<typename M::value_type, M::Rows, M::Cols>();
}


/**
 * \fn cmatrix_ref(const T* mem)
 * \brief Creates an expression wrapper for a C like matrices.
 * \ingroup _unary_function
 *
 * This is like creating a matrix of external data, as described
 * at \ref construct. With this function you wrap an expression
 * around a C style matrix and you can operate directly with it
 * as usual.
 *
 * \par Example:
 * \code
 * static float lhs[3][3] = {
 *   {-1,  0,  1}, { 1,  0,  1}, {-1,  0, -1}
 * };
 * static float rhs[3][3] = {
 *   { 0,  1,  1}, { 0,  1, -1}, { 0, -1,  1}
 * };
 * ...
 *
 * typedef Matrix<float, 3, 3>			matrix_type;
 *
 * matrix_type M( cmatrix_ref<float, 3, 3>(&lhs[0][0])
 *                *  cmatrix_ref<float, 3, 3>(&rhs[0][0]) );
 * \endcode
 *
 * \since release 1.6.0
 */
template<class T>
inline
XprMatrix<
  XprMatrix<MatrixConstReference<T>>
>
cmatrix_ref(const T* mem) {
  typedef XprMatrix<MatrixConstReference<T>>	expr_type;

  return XprMatrix<expr_type>(expr_type(mem));
}


} // namespace gpumatrix

#endif // TVMET_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
