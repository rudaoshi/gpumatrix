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
 * $Id: MatrixOperators.h,v 1.37 2007-06-23 15:58:58 opetzold Exp $
 */

#ifndef TVMET_ARRAY_OPERATORS_H
#define TVMET_ARRAY_OPERATORS_H

namespace gpumatrix {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Array<T1,D>, Array<T2,D>)
 * operator(XprArray<E,D>, Array<T,D>)
 * operator(Array<T,D>, XprArray<E,D>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T1, class T2, int D>		\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    XprArray<ArrayConstReference<T1,D>,D>,					\
    XprArray<ArrayConstReference<T2,D>,D>					\
  >		,D							\
>										\
operator OP (const Array<T1,D>& lhs,					\
	     const Array<T2,D>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class E, class T, int D>			\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<E,D>,							\
    XprArray<ArrayConstReference<T,D>,D>						\
  >		,D							\
>										\
operator OP (const XprArray<E,D>& lhs, 				\
	     const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
										\
template<class T, class E, int D>			\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<ArrayConstReference<T,D>,D>,					\
    XprArray<E,D>							\
  >		,D							\
>										\
operator OP (const Array<T,D>& lhs, 					\
	     const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)			// per se element wise
TVMET_DECLARE_MACRO(sub, -)			// per se element wise
TVMET_DECLARE_MACRO(mul, *)			// see as prod()
TVMET_DECLARE_MACRO(div, /)			// not defined for matrizes

#undef TVMET_DECLARE_MACRO


/*
 * operator(Array<T,D>, POD)
 * operator(POD, Array<T,D>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)				\
template<class T, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    XprArray<ArrayConstReference<T,D>,D>,				\
    XprLiteral<POD >							\
  >	,D										\
>									\
operator OP (const Array<T,D>& lhs, 				\
	     POD rhs) TVMET_CXX_ALWAYS_INLINE;				\
									\
template<class T, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >	,D			\
>									\
operator OP (POD lhs, 							\
	     const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * operator(Array<T,D>, complex<T>)
 * operator(complex<T>, Array<T,D>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)							\
template<class T>					\
XprArray<										\
  XprBinOp<										\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,					\
    MatrixConstReference< std::complex<T>>,					\
    XprLiteral<std::complex<T> >							\
  >										\
>											\
operator OP (const Matrix< std::complex<T>>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;			\
											\
template<class T>					\
XprArray<										\
  XprBinOp<										\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,					\
    XprLiteral< std::complex<T> >,							\
    MatrixConstReference< std::complex<T>>					\
  >										\
>											\
operator OP (const std::complex<T>& lhs,						\
	     const Matrix< std::complex<T>>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)
TVMET_DECLARE_MACRO(sub, -)
TVMET_DECLARE_MACRO(mul, *)
TVMET_DECLARE_MACRO(div, /)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)




/*
 * operator(Array<T,D>, POD)
 * operator(POD, Array<T,D>)
 * Note: operations are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, TP)					\
template<class T, int D>				\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<T, TP >,							\
    XprArray<ArrayConstReference<T,D>,D>,					\
    XprLiteral<TP >								\
  >	,D								\
>										\
operator OP (const Array<T,D>& lhs, TP rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class T, int D>				\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME< TP, T>,							\
    XprLiteral< TP >,								\
    XprArray<ArrayConstReference<T,D>,D>						\
  >		,D							\
>										\
operator OP (TP lhs, const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

// integer operators only, e.g used on double you wil get an error
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
 * unary_operator(Array<T,D>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template <class T, int D>			\
XprArray<								\
  XprUnOp<								\
    Fcnl_##NAME<T>,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >		,D						\
>									\
operator OP (const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(not, !)
TVMET_DECLARE_MACRO(compl, ~)
TVMET_DECLARE_MACRO(neg, -)
#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/**
 * \fn operator<<(std::ostream& os, const Array<T,D>& rhs)
 * \brief Overload operator for i/o
 * \ingroup _binary_operator
 */

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Array<T1,D>, Array<T2,D>)
 * update_operator(Array<T1,D>, XprArray<E,D> rhs)
 * Note: per se element wise
 * \todo: the operator*= can have element wise mul oder product, decide!
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)						\
template<class T1, class T2, int D>		\
inline 										\
Array<T1,D>&								\
operator OP (Array<T1,D>& lhs, const Array<T2,D>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
										\
template<class T, class E, int D>			\
inline 										\
Array<T,D>&								\
operator OP (Array<T,D>& lhs, const XprArray<E,D>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}

TVMET_IMPLEMENT_MACRO(add_eq, +=)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub_eq, -=)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul_eq, *=)		// see note
  TVMET_IMPLEMENT_MACRO(div_eq, /=)		// not defined for vectors
}

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod_eq, %=)
  TVMET_IMPLEMENT_MACRO(xor_eq, ^=)
  TVMET_IMPLEMENT_MACRO(and_eq, &=)
  TVMET_IMPLEMENT_MACRO(or_eq, |=)
  TVMET_IMPLEMENT_MACRO(shl_eq, <<=)
  TVMET_IMPLEMENT_MACRO(shr_eq, >>=)
}

#undef TVMET_IMPLEMENT_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Array<T1,D>, Array<T2,D>)
 * operator(XprArray<E,D>, Array<T,D>)
 * operator(Array<T,D>, XprArray<E,D>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)						      \
template<class T1, class T2, int D>		      \
inline										      \
XprArray<									      \
  XprBinOp<									      \
    Fcnl_##NAME<T1, T2>,							      \
    XprArray<ArrayConstReference<T1,D>,D>,					      \
    XprArray<ArrayConstReference<T2,D>,D>					      \
  >	,D							      \
>										      \
operator OP (const Array<T1,D>& lhs,	const Array<T2,D>& rhs) {  \
  return NAME(lhs, rhs);							      \
}										      \
										      \
template<class E, class T, int D>			      \
inline										      \
XprArray<									      \
  XprBinOp<									      \
    Fcnl_##NAME<typename E::value_type, T>,					      \
    XprArray<E,D>,							      \
    XprArray<ArrayConstReference<T,D>,D>						      \
  >	,D							      \
>										      \
operator OP (const XprArray<E,D>& lhs, const Array<T,D>& rhs) { \
  return NAME(lhs, rhs);							      \
}										      \
										      \
template<class T, class E, int D>			      \
inline										      \
XprArray<									      \
  XprBinOp<									      \
    Fcnl_##NAME<typename E::value_type, T>,					      \
    XprArray<ArrayConstReference<T,D>,D>,					      \
    XprArray<E,D>							      \
  >		,D						      \
>										      \
operator OP (const Array<T,D>& lhs, const XprArray<E,D>& rhs) { \
  return NAME(lhs, rhs);							      \
}

TVMET_IMPLEMENT_MACRO(add, +)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul, *)			// see as prod()
  TVMET_IMPLEMENT_MACRO(div, /)			// not defined for matrizes
}
#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(Array<T,D>, POD)
 * operator(POD, Array<T,D>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)			\
template<class T, int D>		\
inline								\
XprArray<							\
  XprBinOp<							\
    Fcnl_##NAME<T, POD >,					\
    XprArray<ArrayConstReference<T,D>,D>,			\
    XprLiteral<POD >						\
  >	,D						\
>								\
operator OP (const Array<T,D>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);					\
}								\
								\
template<class T, int D>		\
inline								\
XprArray<							\
  XprBinOp<							\
    Fcnl_##NAME< POD, T>,					\
    XprLiteral< POD >,						\
    XprArray<ArrayConstReference<T,D>,D>				\
  >	,D						\
>								\
operator OP (POD lhs, const Array<T,D>& rhs) {	\
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
 * operator(Array<T,D>, complex<T>)
 * operator(complex<T>, Array<T,D>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class T>		\
inline								\
XprArray<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		\
    MatrixConstReference< std::complex<T>>,		\
    XprLiteral<std::complex<T> >				\
  >							\
>								\
operator OP (const Matrix< std::complex<T>>& lhs,	\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class T>		\
inline								\
XprArray<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		\
    XprLiteral< std::complex<T> >,				\
    MatrixConstReference< std::complex<T>>		\
  >							\
>								\
operator OP (const std::complex<T>& lhs,			\
	     const Matrix< std::complex<T>>& rhs) {	\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +)
TVMET_IMPLEMENT_MACRO(sub, -)
TVMET_IMPLEMENT_MACRO(mul, *)
TVMET_IMPLEMENT_MACRO(div, /)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Array<T1,D>, Array<T2,D>)
 * operator(XprArray<E,D>, Array<T,D>)
 * operator(Array<T,D>, XprArray<E,D>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)							\
template<class T1,					\
	 class T2, int D>									\
inline											\
XprArray<										\
  XprBinOp<										\
    Fcnl_##NAME<T1, T2>,								\
    XprArray<ArrayConstReference<T1,D>,D>,						\
    XprArray<ArrayConstReference<T2,D>,D>						\
  >	,D									\
>											\
operator OP (const Array<T1,D>& lhs,						\
	     const Array<T2,D>& rhs) {					\
  typedef XprBinOp <									\
    Fcnl_##NAME<T1, T2>,								\
    XprArray<ArrayConstReference<T1,D>,D>,						\
    XprArray<ArrayConstReference<T2,D>,D>						\
  >							expr_type;			\
  return XprArray<expr_type,D>(expr_type(lhs.as_expr(), rhs.as_expr()));	\
}											\
											\
template<class E,									\
	 class T, int D>					\
inline											\
XprArray<										\
  XprBinOp<										\
    Fcnl_##NAME<typename E::value_type, T>,						\
    XprArray<E,D>,								\
    XprArray<ArrayConstReference<T,D>,D>							\
  >	,D									\
>											\
operator OP (const XprArray<E,D>& lhs, const Array<T,D>& rhs) {	\
  typedef XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,						\
    XprArray<E,D>,								\
    XprArray<ArrayConstReference<T,D>,D>							\
  > 							 expr_type;			\
  return XprArray<expr_type,D>(expr_type(lhs, rhs.as_expr()));		\
}											\
											\
template<class T,					\
	 class E, int D>									\
inline											\
XprArray<										\
  XprBinOp<										\
    Fcnl_##NAME<typename E::value_type, T>,						\
    XprArray<ArrayConstReference<T,D>,D>,						\
    XprArray<E,D>								\
  >		,D								\
>											\
operator OP (const Array<T,D>& lhs, const XprArray<E,D>& rhs) {	\
  typedef XprBinOp<									\
    Fcnl_##NAME<T, typename E::value_type>,						\
    XprArray<ArrayConstReference<T,D>,D>,						\
    XprArray<E,D>								\
  >	 						 expr_type;			\
  return XprArray<expr_type,D>(expr_type(lhs.as_expr(), rhs));		\
}

// integer operators only, e.g used on double you wil get an error
  TVMET_IMPLEMENT_MACRO(mod, %)
  TVMET_IMPLEMENT_MACRO(bitxor, ^)
  TVMET_IMPLEMENT_MACRO(bitand, &)
  TVMET_IMPLEMENT_MACRO(bitor, |)
  TVMET_IMPLEMENT_MACRO(shl, <<)
  TVMET_IMPLEMENT_MACRO(shr, >>)


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
 * operator(Array<T,D>, complex<T>)
 * operator(complex<T>, Array<T,D>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				     \
template<class T>		     \
inline								     \
XprArray<							     \
  XprBinOp<							     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    MatrixConstReference< std::complex<T>>,		     \
    XprLiteral<std::complex<T> >				     \
  >							     \
>								     \
operator OP (const Matrix< std::complex<T>>& lhs,	     \
	     const std::complex<T>& rhs) {			     \
  typedef XprBinOp<						     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    MatrixConstReference< std::complex<T>>,		     \
    XprLiteral< std::complex<T> >				     \
  >							expr_type;   \
  return XprArray<expr_type,D>(			     \
    expr_type(lhs.as_expr(), XprLiteral< std::complex<T> >(rhs))); \
}								     \
								     \
template<class T>		     \
inline								     \
XprArray<							     \
  XprBinOp<							     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    XprLiteral< std::complex<T> >,				     \
    MatrixConstReference< std::complex<T>>		     \
  >							     \
>								     \
operator OP (const std::complex<T>& lhs,			     \
	     const Matrix< std::complex<T>>& rhs) {	     \
  typedef XprBinOp<						     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    XprLiteral< std::complex<T> >,				     \
    XprArray<ArrayConstReference<T,D>,D>				     \
  >							expr_type;   \
  return XprArray<expr_type,D>(			     \
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs.as_expr())); \
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
 * operator(Array<T,D>, POD)
 * operator(POD, Array<T,D>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, TP)			   \
template<class T, int D>		   \
inline								   \
XprArray<							   \
  XprBinOp<							   \
    Fcnl_##NAME<T, TP >,					   \
    XprArray<ArrayConstReference<T,D>,D>,			   \
    XprLiteral<TP >						   \
  >	,D						   \
>								   \
operator OP (const Array<T,D>& lhs, TP rhs) {	   \
  typedef XprBinOp<						   \
    Fcnl_##NAME<T, TP >,					   \
    XprArray<ArrayConstReference<T,D>,D>,			   \
    XprLiteral< TP >						   \
  >							expr_type; \
  return XprArray<expr_type,D>(			   \
    expr_type(lhs.as_expr(), XprLiteral< TP >(rhs)));		   \
}								   \
								   \
template<class T, int D>		   \
inline								   \
XprArray<							   \
  XprBinOp<							   \
    Fcnl_##NAME< TP, T>,					   \
    XprLiteral< TP >,						   \
    XprArray<ArrayConstReference<T,D>,D>				   \
  >	,D						   \
>								   \
operator OP (TP lhs, const Array<T,D>& rhs) {	   \
  typedef XprBinOp<						   \
    Fcnl_##NAME< TP, T>,					   \
    XprLiteral< TP >,						   \
    XprArray<ArrayConstReference<T,D>,D>				   \
  >							expr_type; \
  return XprArray<expr_type,D>(			   \
    expr_type(XprLiteral< TP >(lhs), rhs.as_expr()));		   \
}

// integer operators only, e.g used on double you wil get an error
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
 * unary_operator(Array<T,D>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				       \
template <class T, int D>		       \
inline								       \
XprArray<							       \
  XprUnOp<							       \
    Fcnl_##NAME<T>,						       \
    XprArray<ArrayConstReference<T,D>,D>				       \
  >	,D						       \
>								       \
operator OP (const Array<T,D>& rhs) {		       \
  typedef XprUnOp<						       \
    Fcnl_##NAME<T>,						       \
    XprArray<ArrayConstReference<T,D>,D>				       \
  >  							 expr_type;    \
  return XprArray<expr_type,D>(expr_type(rhs.as_expr())); \
}

TVMET_IMPLEMENT_MACRO(not, !)
TVMET_IMPLEMENT_MACRO(compl, ~)
TVMET_IMPLEMENT_MACRO(neg, -)
#undef TVMET_IMPLEMENT_MACRO


} // namespace gpumatrix

#endif // TVMET_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
