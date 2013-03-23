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

#ifndef TVMET_MAP_ARRAY_OPERATORS_H
#define TVMET_MAP_ARRAY_OPERATORS_H

namespace gpumatrix {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Array<T1,D>, Array<T2,D>)
 * update_operator(Array<T1,D>, XprArray<E,D> rhs)
 * Note: per se element wise
 * \todo: the operator*= can have element wise mul oder product, decide!
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T1, class T2, int D>		\
Map<Array<T1,D>>&								\
operator OP (Map<Array<T1,D>>& lhs, 					\
	     const Map<Array<T2,D>>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
template<class T1, class T2, int D>		\
Map<Array<T1,D>>&								\
operator OP (Map<Array<T1,D>>& lhs, 					\
	     const Array<T2,D>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class T, class E,  int D>			\
Map<Array<T,D>>&								\
operator OP (Map<Array<T,D>>& lhs, 					\
	     const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add_eq, +=)		// per se element wise
TVMET_DECLARE_MACRO(sub_eq, -=)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul_eq, *=)		// see note
  TVMET_DECLARE_MACRO(div_eq, /=)		// not defined for vectors
}

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod_eq, %=)
  TVMET_DECLARE_MACRO(xor_eq, ^=)
  TVMET_DECLARE_MACRO(and_eq, &=)
  TVMET_DECLARE_MACRO(or_eq, |=)
  TVMET_DECLARE_MACRO(shl_eq, <<=)
  TVMET_DECLARE_MACRO(shr_eq, >>=)
}

#undef TVMET_DECLARE_MACRO


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
  >	,D							\
>										\
operator OP (const Map<Array<T1,D>>& lhs,					\
	     const Map<Array<T2,D>>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
template<class T1, class T2, int D>		\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    XprArray<ArrayConstReference<T1,D>,D>,					\
    XprArray<ArrayConstReference<T2,D>,D>					\
  >		,D									\
>										\
operator OP (const Map<Array<T1,D>>& lhs,					\
	     const Array<T2,D>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
template<class T1, class T2, int D>		\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    XprArray<ArrayConstReference<T1,D>,D>,					\
    XprArray<ArrayConstReference<T2,D>,D>					\
  >	,D										\
>										\
operator OP (const Array<T1,D>& lhs,					\
	     const Map<Array<T2,D>>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class E, class T, int D>			\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<E,D>,							\
    XprArray<ArrayConstReference<T,D>,D>						\
  >	,D										\
>										\
operator OP (const XprArray<E,D>& lhs, 				\
	     const Map<Array<T,D>>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
										\
template<class T, class E, int D>			\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<ArrayConstReference<T,D>,D>,					\
    XprArray<E,D>							\
  >	,D										\
>										\
operator OP (const Map<Array<T,D>>& lhs, 					\
	     const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)			// per se element wise
TVMET_DECLARE_MACRO(sub, -)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul, *)			// see as prod()
  TVMET_DECLARE_MACRO(div, /)			// not defined for matrizes
}
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
  >		,D											\
>									\
operator OP (const Map<Array<T,D>>& lhs, 				\
	     POD rhs) TVMET_CXX_ALWAYS_INLINE;				\
									\
template<class T, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >		,D				\
>									\
operator OP (POD lhs, 							\
	     const Map<Array<T,D>>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
Map<Array<T1,D>>&								\
operator OP (Map<Array<T1,D>>& lhs, const Map<Array<T2,D>>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
template<class T1, class T2, int D>		\
inline 										\
Map<Array<T1,D>>&								\
operator OP (Map<Array<T1,D>>& lhs, const Array<T2,D>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
template<class T1, class T2, int D>		\
inline 										\
Array<T1,D>&								\
operator OP (Array<T1,D>& lhs, const Map<Array<T2,D>>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
										\
template<class T, class E,  int D>			\
inline 										\
Map<Array<T,D>>&								\
operator OP (Map<Array<T,D>>& lhs, const XprArray<E,D>& rhs) {	\
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
  >		,D								      \
>										      \
operator OP (const Map<Array<T1,D>> & lhs,	const Map<Array<T2,D>> & rhs) {  \
  return NAME(lhs, rhs);							      \
}										      \
template<class T1, class T2, int D>		      \
inline										      \
XprArray<									      \
  XprBinOp<									      \
    Fcnl_##NAME<T1, T2>,							      \
    XprArray<ArrayConstReference<T1,D>,D>,					      \
    XprArray<ArrayConstReference<T2,D>,D>					      \
  >		,D								      \
>										      \
operator OP (const Map<Array<T1,D>> & lhs,	const Array<T2,D>& rhs) {  \
  return NAME(lhs, rhs);							      \
}										      \
template<class T1, class T2, int D>		      \
inline										      \
XprArray<									      \
  XprBinOp<									      \
    Fcnl_##NAME<T1, T2>,							      \
    XprArray<ArrayConstReference<T1,D>,D>,					      \
    XprArray<ArrayConstReference<T2,D>,D>					      \
  >		,D								      \
>										      \
operator OP (const Array<T1,D>& lhs,	const Map<Array<T2,D>> & rhs) {  \
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
  >	,D									      \
>										      \
operator OP (const XprArray<E,D>& lhs, const Map<Array<T,D>> & rhs) { \
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
  >		,D								      \
>										      \
operator OP (const Map<Array<T,D>>& lhs, const XprArray<E,D>& rhs) { \
  return NAME(lhs, rhs);							      \
}

TVMET_IMPLEMENT_MACRO(add, +)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)			// per se element wise
  TVMET_IMPLEMENT_MACRO(mul, *)			// see as prod()
  TVMET_IMPLEMENT_MACRO(div, /)			// not defined for matrizes

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
  >	,D								\
>								\
operator OP (const Map<Array<T,D>>& lhs, POD rhs) {	\
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
  >		,D							\
>								\
operator OP (POD lhs, const Map<Array<T,D>>& rhs) {	\
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




} // namespace gpumatrix

#endif // TVMET_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
                                                        