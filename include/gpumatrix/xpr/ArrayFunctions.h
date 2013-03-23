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

#ifndef TVMET_XPR_ARRAY_FUNCTIONS_H
#define TVMET_XPR_ARRAY_FUNCTIONS_H

#include <gpumatrix/NumericTraits.h>

namespace gpumatrix {


/* forwards */
template<class T,int D> class Array;
template<class E, int D> class XprArray;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprArray<E1,D>, XprArray<E2,D>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1, class E2, int D>	\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprArray<E1,D>,						\
    XprArray<E2,D>						\
  >	,D							\
>									\
NAME (const XprArray<E1,D>& lhs,				\
      const XprArray<E2,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise
TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
TVMET_DECLARE_MACRO(div)			// not defined for matrizes


#undef TVMET_DECLARE_MACRO


/*
 * function(XprArray<E,D>, POD)
 * function(POD, XprArray<E,D>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class E, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprArray<E,D>,						\
    XprLiteral< POD >							\
  >	,D							\
>									\
NAME (const XprArray<E,D>& lhs, 				\
      POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
									\
template<class E, int D>			\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprArray<E,D>						\
  >	,D							\
>									\
NAME (POD lhs, 								\
      const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * function(XprArray<E,D>, complex<T>)
 * function(complex<T>, XprArray<E,D>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E, class T>		\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprArray<E,D>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
NAME (const XprArray<E,D>& lhs,				\
      const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
									\
template<class T, class E>		\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprArray<E,D>						\
  >								\
>									\
NAME (const std::complex<T>& lhs,					\
      const XprArray<E,D>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprArray<E1,D>, XprArray<E2,D>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E1, class E2, int D>	\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprArray<E1,D>,						\
    XprArray<E2,D>						\
  >,D								\
>									\
NAME (const XprArray<E1,D>& lhs, 				\
      const XprArray<E2,D>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprArray<E1,D>,						\
    XprArray<E2,D>						\
  > 							 expr_type;	\
  return XprArray<expr_type,D>(expr_type(lhs, rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub)			// per se element wise
TVMET_IMPLEMENT_MACRO(mul)			// not defined for matrizes
TVMET_IMPLEMENT_MACRO(div)			// not defined for matrizes


#undef TVMET_IMPLEMENT_MACRO


/*
 * function(XprArray<E,D>, POD)
 * function(POD, XprArray<E,D>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class E, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprArray<E,D>,						\
    XprLiteral< POD >							\
  >	,D							\
>									\
NAME (const XprArray<E,D>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprArray<E,D>,						\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
    expr_type(lhs, XprLiteral< POD >(rhs)));				\
}									\
									\
template<class E, int D>			\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprArray<E,D>						\
  >	,D							\
>									\
NAME (POD lhs, const XprArray<E,D>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprArray<E,D>						\
  >							expr_type;	\
  return XprArray<expr_type,D>(				\
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
 * function(XprArray<E,D>, complex<T>)
 * function(complex<T>, XprArray<E,D>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E, class T>		\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprArray<E,D>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
NAME (const XprArray<E,D>& lhs, 				\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprArray<E,D>,						\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprArray<expr_type>(				\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class T, class E>		\
inline									\
XprArray<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprArray<E,D>						\
  >								\
>									\
NAME (const std::complex<T>& lhs, 					\
      const XprArray<E,D>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprArray<E,D>						\
  >							expr_type;	\
  return XprArray<expr_type>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)



} // namespace gpumatrix

#endif // TVMET_XPR_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
