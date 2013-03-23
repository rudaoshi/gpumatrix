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

#ifndef TVMET_ARRAY_FUNCTIONS_H
#define TVMET_ARRAY_FUNCTIONS_H

#include <gpumatrix/Extremum.h>
#include <gpumatrix/NumericTraits.h>
#include <gpumatrix/xpr/Vector.h>

namespace gpumatrix {




/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Array<T1,D>, Array<T2,D>)
 * function(XprArray<E,D>, Array<T,D>)
 * function(Array<T,D>, XprArray<E,D>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class T1, class T2, int D>	\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    XprArray<ArrayConstReference<T1,D>,D>,				\
    XprArray<ArrayConstReference<T2,D>,D>				\
  >	,D							\
>									\
NAME (const Array<T1,D>& lhs,				\
      const Array<T2,D>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, class T, int D>		\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprArray<E,D>,						\
    XprArray<ArrayConstReference<T,D>,D>					\
  >	,D								\
>									\
NAME (const XprArray<E,D>& lhs,				\
      const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class T, class E, int D>		\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprArray<ArrayConstReference<T,D>,D>,				\
    XprArray<E,D>						\
  >	,D								\
>									\
NAME (const Array<T,D>& lhs,					\
      const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise

TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
TVMET_DECLARE_MACRO(div)			// not defined for matrizes


#undef TVMET_DECLARE_MACRO


/*
 * function(Array<T,D>, POD)
 * function(POD, Array<T,D>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class T, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    XprArray<ArrayConstReference<T,D>,D>,				\
    XprLiteral<POD >							\
  >	,D							\
>									\
NAME (const Array<T,D>& lhs, 				\
      POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
									\
template<class T, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >	,D							\
>									\
NAME (POD lhs, 								\
      const Array<T,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * function(Array<T,D>, complex<T>)
 * function(complex<T>, Array<T,D>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)						\
template<class T, int D>				\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    ArrayConstReference< std::complex<T>>,				\
    XprLiteral<std::complex<T> >						\
  >									\
>										\
NAME (const Matrix< std::complex<T>>& lhs,				\
      const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;			\
										\
template<class T, int D>				\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    XprLiteral< std::complex<T> >,						\
    ArrayConstReference< std::complex<T>>				\
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


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Array<T1,D>, Array<T2,D>)
 * function(XprArray<E,D>, Array<T,D>)
 * function(Array<T,D>, XprArray<E,D>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						\
template<class T1, class T2, int D>		\
inline										\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    XprArray<ArrayConstReference<T1,D>,D>,					\
    XprArray<ArrayConstReference<T2,D>,D>					\
  >	,D								\
>										\
NAME (const Array<T1,D>& lhs, const Array<T2,D>& rhs) {	\
  typedef XprBinOp <								\
    Fcnl_##NAME<T1, T2>,							\
    XprArray<ArrayConstReference<T1,D>,D>,					\
    XprArray<ArrayConstReference<T2,D>,D>					\
  >							expr_type;		\
  return XprArray<expr_type,D>(					\
    expr_type(lhs.as_expr(), rhs.as_expr()));				\
}										\
										\
template<class E, class T, int D>			\
inline										\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<E,D>,							\
    XprArray<ArrayConstReference<T,D>,D>						\
  >		,D								\
>										\
NAME (const XprArray<E,D>& lhs, const Array<T,D>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<E,D>,							\
    XprArray<ArrayConstReference<T,D>,D>						\
  > 							 expr_type;		\
  return XprArray<expr_type,D>(					\
    expr_type(lhs, rhs.as_expr()));						\
}										\
										\
template<class T, class E, int D>			\
inline										\
XprArray<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprArray<ArrayConstReference<T,D>,D>,					\
    XprArray<E,D>							\
  >		,D								\
>										\
NAME (const Array<T,D>& lhs, const XprArray<E,D>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,					\
    XprArray<ArrayConstReference<T,D>,D>,					\
    XprArray<E,D>							\
  >	 						 expr_type;		\
  return XprArray<expr_type,D>(					\
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
 * function(Array<T,D>, POD)
 * function(POD, Array<T,D>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class T, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    XprArray<ArrayConstReference<T,D>,D>,				\
    XprLiteral<POD >							\
  >			,D						\
>									\
NAME (const Array<T,D>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<T, POD >,						\
    XprArray<ArrayConstReference<T,D>,D>,				\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
    expr_type(lhs.as_expr(), XprLiteral< POD >(rhs)));		\
}									\
									\
template<class T, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >		,D							\
>									\
NAME (POD lhs, const Array<T,D>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    XprArray<ArrayConstReference<T,D>,D>					\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
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
 * function(Array<T,D>, complex<T>)
 * function(complex<T>, Array<T,D>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class T, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    ArrayConstReference< std::complex<T>>,			\
    XprLiteral<std::complex<T> >					\
  >								\
>									\
NAME (const Matrix< std::complex<T>>& lhs,			\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    ArrayConstReference< std::complex<T>>,			\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
    expr_type(lhs.as_expr(), XprLiteral< std::complex<T> >(rhs)));	\
}									\
									\
template<class T, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    ArrayConstReference< std::complex<T>>			\
  >								\
>									\
NAME (const std::complex<T>& lhs,					\
      const Matrix< std::complex<T>>& rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    ArrayConstReference<std::complex<T>>			\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs.as_expr()));	\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace gpumatrix

#endif // TVMET_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
