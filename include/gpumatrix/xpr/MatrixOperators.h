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
 * $Id: MatrixOperators.h,v 1.23 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_OPERATORS_H
#define TVMET_XPR_MATRIX_OPERATORS_H

namespace gpumatrix {

	template <class C> class Map;

/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(const XprMatrix<E1>& lhs, const XprMatrix<E2,Cols2>& rhs)
 *
 * Note: operations +,-,*,/ are per se element wise. Further more,
 * element wise operations make sense only for matrices of the same
 * size [varg].
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E1,			\
         class E2>								\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,		\
    XprMatrix<E1>,						\
    XprMatrix<E2>							\
  >									\
>										\
operator OP (const XprMatrix<E1>& lhs,				\
	     const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise


#undef TVMET_DECLARE_MACRO


/*
 * operator(XprMatrix<E>,  POD)
 * operator(POD, XprMatrix<E>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)					\
template<class E>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, POD >,					\
    XprMatrix<E>,							\
    XprLiteral< POD >								\
  >								\
>										\
operator OP (const XprMatrix<E>& lhs, 				\
	     POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
										\
template<class E>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<POD, typename E::value_type>,					\
    XprLiteral< POD >,								\
    XprMatrix<E>							\
  >								\
>										\
operator OP (POD lhs, 								\
	     const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +, int)
TVMET_DECLARE_MACRO(sub, -, int)
TVMET_DECLARE_MACRO(mul, *, int)
TVMET_DECLARE_MACRO(div, /, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(add, +, long long int)
TVMET_DECLARE_MACRO(sub, -, long long int)
TVMET_DECLARE_MACRO(mul, *, long long int)
TVMET_DECLARE_MACRO(div, /, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_DECLARE_MACRO(add, +, float)
TVMET_DECLARE_MACRO(sub, -, float)
TVMET_DECLARE_MACRO(mul, *, float)
TVMET_DECLARE_MACRO(div, /, float)

TVMET_DECLARE_MACRO(add, +, double)
TVMET_DECLARE_MACRO(sub, -, double)
TVMET_DECLARE_MACRO(mul, *, double)
TVMET_DECLARE_MACRO(div, /, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(add, +, long double)
TVMET_DECLARE_MACRO(sub, -, long double)
TVMET_DECLARE_MACRO(mul, *, long double)
TVMET_DECLARE_MACRO(div, /, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprMatrix<E>, complex<>)
 * operator(complex<>, XprMatrix<E>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E,  class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,			\
    XprMatrix<E>,							\
    XprLiteral< std::complex<T> >						\
  >									\
>										\
operator OP (const XprMatrix<E>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
										\
template<class E,  class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,			\
    XprLiteral< std::complex<T> >,						\
    XprMatrix<E>							\
  >								\
>										\
operator OP (const std::complex<T>& lhs,					\
	     const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)
TVMET_DECLARE_MACRO(sub, -)
TVMET_DECLARE_MACRO(mul, *)
TVMET_DECLARE_MACRO(div, /)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1> lhs, XprMatrix<E2> rhs)
 */
template<class E1,
	 class E2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Cols1)
  >
>
operator*(const XprMatrix<E1>& lhs,
	  const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1>& lhs, const XprVector<E2>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1> lhs, XprVector<E2> rhs)
 */
template<class E1, 
	 class E2>
XprVector<
  XprMVProduct<
    XprMatrix<E1>,
    XprVector<E2>
  >
>
operator*(const XprMatrix<E1>& lhs,
	  const XprVector<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprMatrix<>, XprMatrix<>)
 * Note: operations are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E1, 				\
         class E2>								\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,		\
    XprMatrix<E1>,							\
    XprMatrix<E2>							\
  >						\
>										\
operator OP (const XprMatrix<E1>& lhs, 				\
	     const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;

// integer operators only, e.g used on double you will get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %)
  TVMET_DECLARE_MACRO(bitxor, ^)
  TVMET_DECLARE_MACRO(bitand, &)
  TVMET_DECLARE_MACRO(bitor, |)
  TVMET_DECLARE_MACRO(shl, <<)
  TVMET_DECLARE_MACRO(shr, >>)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >)
TVMET_DECLARE_MACRO(less, <)
TVMET_DECLARE_MACRO(greater_eq, >=)
TVMET_DECLARE_MACRO(less_eq, <=)
TVMET_DECLARE_MACRO(eq, ==)
TVMET_DECLARE_MACRO(not_eq, !=)
TVMET_DECLARE_MACRO(and, &&)
TVMET_DECLARE_MACRO(or, ||)

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprMatrix<E>, std::complex<>)
 * operator(std::complex<>, XprMatrix<E>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E,  class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,			\
    XprMatrix<E>,							\
    XprLiteral< std::complex<T> >						\
  >									\
>										\
operator OP (const XprMatrix<E>& lhs, 				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
										\
template<class E,  class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,			\
    XprLiteral< std::complex<T> >,						\
    XprMatrix<E>							\
  >									\
>										\
operator OP (const std::complex<T>& lhs, 					\
	     const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >)
TVMET_DECLARE_MACRO(less, <)
TVMET_DECLARE_MACRO(greater_eq, >=)
TVMET_DECLARE_MACRO(less_eq, <=)
TVMET_DECLARE_MACRO(eq, ==)
TVMET_DECLARE_MACRO(not_eq, !=)
TVMET_DECLARE_MACRO(and, &&)
TVMET_DECLARE_MACRO(or, ||)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*
 * operator(XprMatrix<E>, POD)
 * operator(POD, XprMatrix<E>)
 * Note: operations are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, TP)					\
template<class E>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, TP >,					\
    XprMatrix<E>,							\
    XprLiteral< TP >								\
  >								\
>										\
operator OP (const XprMatrix<E>& lhs, 				\
	     TP rhs) TVMET_CXX_ALWAYS_INLINE;					\
										\
template<class E>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<TP, typename E::value_type>,					\
    XprLiteral< TP >,								\
    XprMatrix<E>							\
  >								\
>										\
operator OP (TP lhs, 								\
	     const XprMatrix<E>& rhs) TVMET_CXX_ALWAYS_INLINE;

// integer operators only, e.g used on double you will get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %, int)
  TVMET_DECLARE_MACRO(bitxor, ^, int)
  TVMET_DECLARE_MACRO(bitand, &, int)
  TVMET_DECLARE_MACRO(bitor, |, int)
  TVMET_DECLARE_MACRO(shl, <<, int)
  TVMET_DECLARE_MACRO(shr, >>, int)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, int)
TVMET_DECLARE_MACRO(less, <, int)
TVMET_DECLARE_MACRO(greater_eq, >=, int)
TVMET_DECLARE_MACRO(less_eq, <=, int)
TVMET_DECLARE_MACRO(eq, ==, int)
TVMET_DECLARE_MACRO(not_eq, !=, int)
TVMET_DECLARE_MACRO(and, &&, int)
TVMET_DECLARE_MACRO(or, ||, int)

#if defined(TVMET_HAVE_LONG_LONG)
// integer operators only
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %, long long int)
  TVMET_DECLARE_MACRO(bitxor, ^, long long int)
  TVMET_DECLARE_MACRO(bitand, &, long long int)
  TVMET_DECLARE_MACRO(bitor, |, long long int)
  TVMET_DECLARE_MACRO(shl, <<, long long int)
  TVMET_DECLARE_MACRO(shr, >>, long long int)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, long long int)
TVMET_DECLARE_MACRO(less, <, long long int)
TVMET_DECLARE_MACRO(greater_eq, >=, long long int)
TVMET_DECLARE_MACRO(less_eq, <=, long long int)
TVMET_DECLARE_MACRO(eq, ==, long long int)
TVMET_DECLARE_MACRO(not_eq, !=, long long int)
TVMET_DECLARE_MACRO(and, &&, long long int)
TVMET_DECLARE_MACRO(or, ||, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, float)
TVMET_DECLARE_MACRO(less, <, float)
TVMET_DECLARE_MACRO(greater_eq, >=, float)
TVMET_DECLARE_MACRO(less_eq, <=, float)
TVMET_DECLARE_MACRO(eq, ==, float)
TVMET_DECLARE_MACRO(not_eq, !=, float)
TVMET_DECLARE_MACRO(and, &&, float)
TVMET_DECLARE_MACRO(or, ||, float)

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, double)
TVMET_DECLARE_MACRO(less, <, double)
TVMET_DECLARE_MACRO(greater_eq, >=, double)
TVMET_DECLARE_MACRO(less_eq, <=, double)
TVMET_DECLARE_MACRO(eq, ==, double)
TVMET_DECLARE_MACRO(not_eq, !=, double)
TVMET_DECLARE_MACRO(and, &&, double)
TVMET_DECLARE_MACRO(or, ||, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, long double)
TVMET_DECLARE_MACRO(less, <, long double)
TVMET_DECLARE_MACRO(greater_eq, >=, long double)
TVMET_DECLARE_MACRO(less_eq, <=, long double)
TVMET_DECLARE_MACRO(eq, ==, long double)
TVMET_DECLARE_MACRO(not_eq, !=, long double)
TVMET_DECLARE_MACRO(and, &&, long double)
TVMET_DECLARE_MACRO(or, ||, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(const XprMatrix<E>& m)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template <class E>				\
XprMatrix<									\
  XprUnOp<									\
    Fcnl_##NAME<typename E::value_type>,					\
    XprMatrix<E>							\
  >								\
>										\
operator OP (const XprMatrix<E>& m) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(not, !)
TVMET_DECLARE_MACRO(compl, ~)
TVMET_DECLARE_MACRO(neg, -)

#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(const XprMatrix<E1>& lhs, const XprMatrix<E2,Cols2>& rhs)
 *
 * Note: operations +,-,*,/ are per se element wise. Further more,
 * element wise operations make sense only for matrices of the same
 * size [varg].
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class E1,		\
         class E2>							\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,					\
    XprMatrix<E2>						\
  >								\
>									\
operator OP (const XprMatrix<E1>& lhs, 			\
	     const XprMatrix<E2>& rhs) {			\
  return NAME (lhs, rhs);						\
}

TVMET_IMPLEMENT_MACRO(add, +)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)		// per se element wise

#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(XprMatrix<E>,  POD)
 * operator(POD, XprMatrix<E>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)			\
template<class E>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, POD >,			\
    XprMatrix<E>,					\
    XprLiteral< POD >						\
  >						\
>								\
operator OP (const XprMatrix<E>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<POD, typename E::value_type>,			\
    XprLiteral< POD >,						\
    XprMatrix<E>					\
  >							\
>								\
operator OP (POD lhs, const XprMatrix<E>& rhs) {	\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +, int)
TVMET_IMPLEMENT_MACRO(sub, -, int)
TVMET_IMPLEMENT_MACRO(mul, *, int)
TVMET_IMPLEMENT_MACRO(div, /, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(add, +, long long int)
TVMET_IMPLEMENT_MACRO(sub, -, long long int)
TVMET_IMPLEMENT_MACRO(mul, *, long long int)
TVMET_IMPLEMENT_MACRO(div, /, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_IMPLEMENT_MACRO(add, +, float)
TVMET_IMPLEMENT_MACRO(sub, -, float)
TVMET_IMPLEMENT_MACRO(mul, *, float)
TVMET_IMPLEMENT_MACRO(div, /, float)

TVMET_IMPLEMENT_MACRO(add, +, double)
TVMET_IMPLEMENT_MACRO(sub, -, double)
TVMET_IMPLEMENT_MACRO(mul, *, double)
TVMET_IMPLEMENT_MACRO(div, /, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(add, +, long double)
TVMET_IMPLEMENT_MACRO(sub, -, long double)
TVMET_IMPLEMENT_MACRO(mul, *, long double)
TVMET_IMPLEMENT_MACRO(div, /, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprMatrix<E>, complex<>)
 * operator(complex<>, XprMatrix<E>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class E,  class T>	\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,	\
    XprMatrix<E>,					\
    XprLiteral< std::complex<T> >				\
  >						\
>								\
operator OP (const XprMatrix<E>& lhs,		\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E,  class T>	\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,	\
    XprLiteral< std::complex<T> >,				\
    XprMatrix<E>					\
  >					\
>								\
operator OP (const std::complex<T>& lhs,			\
	     const XprMatrix<E>& rhs) {		\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +)
TVMET_IMPLEMENT_MACRO(sub, -)
TVMET_IMPLEMENT_MACRO(mul, *)
TVMET_IMPLEMENT_MACRO(div, /)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1> lhs, XprMatrix<E2> rhs)
 */
template<class E1,
	 class E2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1>,	// M1(Rows1)
    XprMatrix<E2>		// M2(Cols1)
  >
>
operator*(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {
  return prod(lhs, rhs);
}

/**
 * \fn operator*(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs)
 * \brief Evaluate the product of XprMatrix and Matrix.
 * \ingroup _binary_operator
 * \sa prod(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs)
 */
template<class E1,
	 class T2>
inline
XprMatrix<
  XprMtMProduct<
    E1,
    XprMatrix<MatrixConstReference<T2>>
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const Matrix<T2>& rhs) {
	return MtM_prod(lhs.expr().expr(), rhs.as_expr());
}

/**
 * \fn operator*(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs)
 * \brief Evaluate the product of XprMatrix and Matrix.
 * \ingroup _binary_operator
 * \sa prod(const XprMatrix<E1>& lhs, const Matrix<T2>& rhs)
 */
template<class E1,
	 class E2>
inline
XprMatrix<
  XprMtMProduct<
    E1,
    XprMatrix<E2>
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const XprMatrix<E2>& rhs) {
  return MtM_prod(lhs.expr().expr(), rhs);
}

/**
 * \fn operator*(const Matrix<T1>& lhs, const XprMatrix<E2>& rhs)
 * \brief Evaluate the product of Matrix and XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(const Matrix<T>& lhs, const XprMatrix<E>& rhs)
 */
template<class T1,
	 class E2>
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<MatrixConstReference<T1>>,
    E2
  >
>
operator*(const Matrix<T1>& lhs, const XprMatrix<XprMatrixTranspose<E2>>& rhs) {
	return MMt_prod(lhs.as_expr(), rhs.expr().expr());
}

template<class T1,
	 class E2>
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<MatrixConstReference<T1>>,
    E2
  >
>
operator*(const Map<Matrix<T1>>& lhs, const XprMatrix<XprMatrixTranspose<E2>>& rhs) {
	return MMt_prod(lhs.as_expr(), rhs.expr().expr());
}

template<class E1,
	 class E2>
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<E1>,
    E2
  >
>
operator*(const XprMatrix<E1>& lhs, const XprMatrix<XprMatrixTranspose<E2>>& rhs) {
  return MMt_prod(lhs, rhs.expr().expr());
}

template<class E1, class E2>
inline
XprMatrix<
  XprMtMtProduct<
    E1,
    E2
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const XprMatrix<XprMatrixTranspose<E2>>& rhs) {
  return MtMt_prod(lhs.expr().expr(), rhs.expr().expr());
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1>& lhs, const XprVector<E2>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1> lhs, XprVector<E2> rhs)
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
operator*(const XprMatrix<E1>& lhs, const XprVector<E2>& rhs) {
  return prod(lhs, rhs);
}

template<class E1, 
	 class E2>
inline
XprVector<
  XprMtVProduct<
    E1,
    XprVector<E2>
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const XprVector<E2>& rhs) {
  return MtV_prod(lhs.expr().expr(), rhs);
}

template<class E1, 
	 class T>
inline
XprVector<
  XprMtVProduct<
    E1,
    XprVector<VectorConstReference<T>>
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const Vector<T>& rhs) {
  return MtV_prod(lhs.expr().expr(), rhs);
}

template<class E1, 
	 class T>
inline
XprVector<
  XprMtVProduct<
    E1,
    XprVector<VectorConstReference<T>>
  >
>
operator*(const XprMatrix<XprMatrixTranspose<E1>>& lhs, const Map<Vector<T>>& rhs) {
  return MtV_prod(lhs.expr().expr(), rhs);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprMatrix<>, XprMatrix<>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class E1, 			\
         class E2>							\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  >								\
>									\
operator OP (const XprMatrix<E1>& lhs, 			\
	     const XprMatrix<E2>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  >		    					expr_type;	\
  return XprMatrix<expr_type>(expr_type(lhs, rhs));		\
}

// integer operators only, e.g used on double you will get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %)
  TVMET_IMPLEMENT_MACRO(bitxor, ^)
  TVMET_IMPLEMENT_MACRO(bitand, &)
  TVMET_IMPLEMENT_MACRO(bitor, |)
  TVMET_IMPLEMENT_MACRO(shl, <<)
  TVMET_IMPLEMENT_MACRO(shr, >>)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >)
TVMET_IMPLEMENT_MACRO(less, <)
TVMET_IMPLEMENT_MACRO(greater_eq, >=)
TVMET_IMPLEMENT_MACRO(less_eq, <=)
TVMET_IMPLEMENT_MACRO(eq, ==)
TVMET_IMPLEMENT_MACRO(not_eq, !=)
TVMET_IMPLEMENT_MACRO(and, &&)
TVMET_IMPLEMENT_MACRO(or, ||)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprMatrix<E>, std::complex<>)
 * operator(std::complex<>, XprMatrix<E>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class E,  class T>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
operator OP (const XprMatrix<E>& lhs, 			\
	     const std::complex<T>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class E,  class T>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E>						\
  >								\
>									\
operator OP (const std::complex<T>& lhs, 				\
	     const XprMatrix<E>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E>						\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >)
TVMET_IMPLEMENT_MACRO(less, <)
TVMET_IMPLEMENT_MACRO(greater_eq, >=)
TVMET_IMPLEMENT_MACRO(less_eq, <=)
TVMET_IMPLEMENT_MACRO(eq, ==)
TVMET_IMPLEMENT_MACRO(not_eq, !=)
TVMET_IMPLEMENT_MACRO(and, &&)
TVMET_IMPLEMENT_MACRO(or, ||)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*
 * operator(XprMatrix<E>, POD)
 * operator(POD, XprMatrix<E>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, TP)				\
template<class E>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprMatrix<E>,						\
    XprLiteral< TP >							\
  >								\
>									\
operator OP (const XprMatrix<E>& lhs, TP rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprMatrix<E>,						\
    XprLiteral< TP >							\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< TP >(rhs)));				\
}									\
									\
template<class E>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<TP, typename E::value_type>,				\
    XprLiteral< TP >,							\
    XprMatrix<E>						\
  >								\
>									\
operator OP (TP lhs, const XprMatrix<E>& rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME< TP, typename E::value_type>,				\
    XprLiteral< TP >,							\
    XprMatrix<E>						\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(XprLiteral< TP >(lhs), rhs));				\
}


// integer operators only, e.g used on double you will get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %, int)
  TVMET_IMPLEMENT_MACRO(bitxor, ^, int)
  TVMET_IMPLEMENT_MACRO(bitand, &, int)
  TVMET_IMPLEMENT_MACRO(bitor, |, int)
  TVMET_IMPLEMENT_MACRO(shl, <<, int)
  TVMET_IMPLEMENT_MACRO(shr, >>, int)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, int)
TVMET_IMPLEMENT_MACRO(less, <, int)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, int)
TVMET_IMPLEMENT_MACRO(less_eq, <=, int)
TVMET_IMPLEMENT_MACRO(eq, ==, int)
TVMET_IMPLEMENT_MACRO(not_eq, !=, int)
TVMET_IMPLEMENT_MACRO(and, &&, int)
TVMET_IMPLEMENT_MACRO(or, ||, int)

#if defined(TVMET_HAVE_LONG_LONG)
// integer operators only
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %, long long int)
  TVMET_IMPLEMENT_MACRO(bitxor, ^, long long int)
  TVMET_IMPLEMENT_MACRO(bitand, &, long long int)
  TVMET_IMPLEMENT_MACRO(bitor, |, long long int)
  TVMET_IMPLEMENT_MACRO(shl, <<, long long int)
  TVMET_IMPLEMENT_MACRO(shr, >>, long long int)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, long long int)
TVMET_IMPLEMENT_MACRO(less, <, long long int)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, long long int)
TVMET_IMPLEMENT_MACRO(less_eq, <=, long long int)
TVMET_IMPLEMENT_MACRO(eq, ==, long long int)
TVMET_IMPLEMENT_MACRO(not_eq, !=, long long int)
TVMET_IMPLEMENT_MACRO(and, &&, long long int)
TVMET_IMPLEMENT_MACRO(or, ||, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, float)
TVMET_IMPLEMENT_MACRO(less, <, float)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, float)
TVMET_IMPLEMENT_MACRO(less_eq, <=, float)
TVMET_IMPLEMENT_MACRO(eq, ==, float)
TVMET_IMPLEMENT_MACRO(not_eq, !=, float)
TVMET_IMPLEMENT_MACRO(and, &&, float)
TVMET_IMPLEMENT_MACRO(or, ||, float)

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, double)
TVMET_IMPLEMENT_MACRO(less, <, double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, double)
TVMET_IMPLEMENT_MACRO(eq, ==, double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, double)
TVMET_IMPLEMENT_MACRO(and, &&, double)
TVMET_IMPLEMENT_MACRO(or, ||, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, long double)
TVMET_IMPLEMENT_MACRO(less, <, long double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, long double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, long double)
TVMET_IMPLEMENT_MACRO(eq, ==, long double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, long double)
TVMET_IMPLEMENT_MACRO(and, &&, long double)
TVMET_IMPLEMENT_MACRO(or, ||, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(const XprMatrix<E>& m)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class E>			\
inline									\
XprMatrix<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprMatrix<E>						\
  >								\
>									\
operator OP (const XprMatrix<E>& m) {			\
  typedef XprUnOp<							\
    Fcnl_##NAME<typename E::value_type>,				\
    XprMatrix<E>						\
  >  							 expr_type;	\
  return XprMatrix<expr_type>(expr_type(m));		\
}

TVMET_IMPLEMENT_MACRO(not, !)
TVMET_IMPLEMENT_MACRO(compl, ~)
TVMET_IMPLEMENT_MACRO(neg, -)

#undef TVMET_IMPLEMENT_MACRO


} // namespace gpumatrix

#endif // TVMET_XPR_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
