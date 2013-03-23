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
 * $Id: MatrixFunctions.h,v 1.44 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_FUNCTIONS_H
#define TVMET_XPR_MATRIX_FUNCTIONS_H

#include <gpumatrix/NumericTraits.h>

namespace gpumatrix {


/* forwards */
template<class T> class Matrix;
template<class T> class Vector;
template<class E> class XprVector;
template<class E> class XprMatrixTranspose;
template<class E> class XprMatrixDiag;
template<class E> class XprMatrixRow;
template<class E> class XprMatrixCol;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprMatrix<E1>, XprMatrix<E2>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1, class E2>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  >								\
>									\
NAME (const XprMatrix<E1>& lhs,				\
      const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
  TVMET_DECLARE_MACRO(div)			// not defined for matrizes
}

#undef TVMET_DECLARE_MACRO


/*
 * function(XprMatrix<E>, POD)
 * function(POD, XprMatrix<E>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class E>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E>,						\
    XprLiteral< POD >							\
  >								\
>									\
NAME (const XprMatrix<E>& lhs, 				\
      POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
									\
template<class E>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E>						\
  >								\
>									\
NAME (POD lhs, 								\
      const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * function(XprMatrix<E>, complex<T>)
 * function(complex<T>, XprMatrix<E>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E, class T>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
NAME (const XprMatrix<E>& lhs,				\
      const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
									\
template<class T, class E>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E>						\
  >								\
>									\
NAME (const std::complex<T>& lhs,					\
      const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)
TVMET_DECLARE_MACRO(sub)
TVMET_DECLARE_MACRO(mul)
TVMET_DECLARE_MACRO(div)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E1,	 class E2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>
  >				// return Dim
>
prod(const XprMatrix<E1>& lhs,
     const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1,	 class E2>
XprMatrix<
  XprMtMtProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Cols1)
  >			// return Dim
>
MtMt_prod(const XprMatrix<E1>& lhs,
	   const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1,	 class E2>	
XprMatrix<
  XprMtMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Rows1)
  >					// return Dim
>
MtM_prod(const XprMatrix<E1>& lhs,
	 const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1,	 class E2> 		
XprMatrix<
  XprMMtProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2> 		// M2(Rows2)
  >					// return Dim
>
MMt_prod(const XprMatrix<E1>& lhs,
	 const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E1, 
	 class E2>
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
    XprVector<E2>
  >
>
prod(const XprMatrix<E1>& lhs,
     const XprVector<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E>
XprMatrix<
  XprMatrixTranspose<
    XprMatrix<E>
  >
>
trans(const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E>
typename NumericTraits<typename E::value_type>::sum_type
trace(const XprMatrix<E>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E>
XprVector<
  XprMatrixRow<
    XprMatrix<E>
  >
>
row(const XprMatrix<E>& m,
    std::size_t no) TVMET_CXX_ALWAYS_INLINE;


template<class E>
XprVector<
  XprMatrixCol<
    XprMatrix<E>
  >
>
col(const XprMatrix<E>& m, std::size_t no) TVMET_CXX_ALWAYS_INLINE;


template<class E>
XprVector<
  XprMatrixDiag<
    XprMatrix<E>
  >
>
diag(const XprMatrix<E>& m) TVMET_CXX_ALWAYS_INLINE;


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprMatrix<E1>, XprMatrix<E2>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E1, class E2>	\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  >								\
>									\
NAME (const XprMatrix<E1>& lhs, 				\
      const XprMatrix<E2>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  > 							 expr_type;	\
  return XprMatrix<expr_type>(expr_type(lhs, rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul)			// not defined for matrizes
  TVMET_IMPLEMENT_MACRO(div)			// not defined for matrizes
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * function(XprMatrix<E>, POD)
 * function(POD, XprMatrix<E>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class E>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E>,						\
    XprLiteral< POD >							\
  >								\
>									\
NAME (const XprMatrix<E>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E>,						\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< POD >(rhs)));				\
}									\
									\
template<class E>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E>						\
  >								\
>									\
NAME (POD lhs, const XprMatrix<E>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E>						\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< POD >(lhs), rhs));				\
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
 * function(XprMatrix<E>, complex<T>)
 * function(complex<T>, XprMatrix<E>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E, class T>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
NAME (const XprMatrix<E>& lhs, 				\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class T, class E>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E>						\
  >								\
>									\
NAME (const std::complex<T>& lhs, 					\
      const XprMatrix<E>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E>						\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * Perform on given Matrix M1 and M2:
 * \f[
 * M_1\,M_2
 * \f]
 * \note The numer of Rows2 has to be equal to Cols1.
 * \ingroup _binary_function
 */
template<class E1,
	 class E2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>
  >				// return Dim
>
prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<E1>,
    XprMatrix<E2>
  >							expr_type;
  return XprMatrix<expr_type>(expr_type(lhs, rhs));
}


/**
 * \fn trans_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Function for the trans(matrix-matrix-product)
 * Perform on given Matrix M1 and M2:
 * \f[
 * (M_1\,M_2)^T
 * \f]
 * \note The numer of Rows2 has to be equal to Cols1.
 * \ingroup _binary_function
 */
template<class E1,
	 class E2>
inline
XprMatrix<
  XprMtMtProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Cols1)
  >			// return Dim
>
MtMt_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {
  typedef XprMtMtProduct<
    XprMatrix<E1>,
    XprMatrix<E2>
  >							expr_type;
  return XprMatrix<expr_type>(expr_type(lhs, rhs));
}


/**
 * \fn MtM_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Function for the trans(matrix)-matrix-product.
 *        using formula
 *        \f[
 *        M_1^{T}\,M_2
 *        \f]
 * \note The number of cols of matrix 2 have to be equal to number of rows of
 *       matrix 1, since matrix 1 is trans - the result is a (Cols1 x Cols2)
 *       matrix.
 * \ingroup _binary_function
 */
template<class E1,
	 class E2>	// Rows2 = Rows1
inline
XprMatrix<
  XprMtMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Rows1)
  >					// return Dim
>
MtM_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {
  typedef XprMtMProduct<
    XprMatrix<E1>,
    XprMatrix<E2>
  >							expr_type;
  return XprMatrix<expr_type>(expr_type(lhs, rhs));
}


/**
 * \fn MMt_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Function for the matrix-trans(matrix)-product.
 * \ingroup _binary_function
 * \note The cols2 has to be equal to cols1.
 */
template<class E1,
	 class E2> // Cols2 = Cols1
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>	 	// M2(Rows2)
  >					// return Dim
>
MMt_prod(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {
  typedef XprMMtProduct<
    XprMatrix<E1>,
    XprMatrix<E2>
  >							expr_type;
  return XprMatrix<expr_type>(expr_type(lhs, rhs));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const XprMatrix<E1>& lhs, const XprVector<E2>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_function
 */
template<class E1, 
	 class E2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
    XprVector<E2>
  >
>
prod(const XprMatrix<E1>& lhs, const XprVector<E2>& rhs) {
  typedef XprMVProduct<
    XprMatrix<E1>,
    XprVector<E2>
  >							expr_type;
  return XprVector<expr_type>(expr_type(lhs, rhs));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn trans(const XprMatrix<E>& rhs)
 * \brief Transpose an expression matrix.
 * \ingroup _unary_function
 */
// template<class E>
// inline
// XprMatrix<
//   XprMatrixTranspose<
//     XprMatrix<E>
//   >
// >
// trans(const XprMatrix<E>& rhs) {
//   typedef XprMatrixTranspose<
//     XprMatrix<E>
//   >							expr_type;
//   return XprMatrix<expr_type>(expr_type(rhs));
// }


/*
 * \fn trace(const XprMatrix<E>& m)
 * \brief Compute the trace of a square matrix.
 * \ingroup _unary_function
 *
 * Simply compute the trace of the given matrix expression as:
 * \f[
 *  \sum_{k = 0}^{Sz-1} m(k, k)
 * \f]
 */
// template<class E>
// inline
// typename NumericTraits<typename E::value_type>::sum_type
// trace(const XprMatrix<E>& m) {
//   return meta::Matrix<Sz, 0, 0>::trace(m);
// }


/**
 * \fn row(const XprMatrix<E>& m, std::size_t no)
 * \brief Returns a row vector of the given matrix.
 * \ingroup _binary_function
 */
template<class E>
inline
XprVector<
  XprMatrixRow<
    XprMatrix<E>
  >
>
row(const XprMatrix<E>& m, std::size_t no) {
  typedef XprMatrixRow<
    XprMatrix<E>
  >							expr_type;

  return XprVector<expr_type>(expr_type(m, no));
}


/**
 * \fn col(const XprMatrix<E>& m, std::size_t no)
 * \brief Returns a column vector of the given matrix.
 * \ingroup _binary_function
 */
template<class E>
inline
XprVector<
  XprMatrixCol<
    XprMatrix<E>
  >
>
col(const XprMatrix<E>& m, std::size_t no) {
  typedef XprMatrixCol<
    XprMatrix<E>
  >							expr_type;

  return XprVector<expr_type>(expr_type(m, no));
}


/**
 * \fn diag(const XprMatrix<E>& m)
 * \brief Returns the diagonal vector of the given square matrix.
 * \ingroup _unary_function
 */
template<class E>
inline
XprVector<
  XprMatrixDiag<
    XprMatrix<E>
  >
>
diag(const XprMatrix<E>& m) {
  typedef XprMatrixDiag<
    XprMatrix<E>> 						expr_type;

  return XprVector<expr_type>(expr_type(m));
}


} // namespace gpumatrix

#endif // TVMET_XPR_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
