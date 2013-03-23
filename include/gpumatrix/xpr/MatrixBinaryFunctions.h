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
 * $Id: MatrixBinaryFunctions.h,v 1.12 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_BINARY_FUNCTIONS_H
#define TVMET_XPR_MATRIX_BINARY_FUNCTIONS_H

namespace gpumatrix {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/

/*
 * binary_function(XprMatrix<E1>, XprMatrix<E2>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1,  class E2>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1>,						\
    XprMatrix<E2>						\
  >								\
>									\
NAME(const XprMatrix<E1>& lhs, 				\
     const XprMatrix<E2>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(atan2)
TVMET_DECLARE_MACRO(drem)
TVMET_DECLARE_MACRO(fmod)
TVMET_DECLARE_MACRO(hypot)
TVMET_DECLARE_MACRO(jn)
TVMET_DECLARE_MACRO(yn)
TVMET_DECLARE_MACRO(pow)
#if defined(TVMET_HAVE_COMPLEX)
  //TVMET_DECLARE_MACRO(polar)
#endif

#undef TVMET_DECLARE_MACRO


/*
 * binary_function(XprMatrix<E>, POD)
 */
#define TVMET_DECLARE_MACRO(NAME, TP)			\
template<class E>	\
XprMatrix<						\
  XprBinOp<						\
    Fcnl_##NAME<typename E::value_type, TP >,		\
    XprMatrix<E>,				\
    XprLiteral< TP >					\
  >					\
>							\
NAME(const XprMatrix<E>& lhs, 		\
     TP rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(atan2, int)
TVMET_DECLARE_MACRO(drem, int)
TVMET_DECLARE_MACRO(fmod, int)
TVMET_DECLARE_MACRO(hypot, int)
TVMET_DECLARE_MACRO(jn, int)
TVMET_DECLARE_MACRO(yn, int)
TVMET_DECLARE_MACRO(pow, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(atan2, long long int)
TVMET_DECLARE_MACRO(drem, long long int)
TVMET_DECLARE_MACRO(fmod, long long int)
TVMET_DECLARE_MACRO(hypot, long long int)
TVMET_DECLARE_MACRO(jn, long long int)
TVMET_DECLARE_MACRO(yn,long long int)
TVMET_DECLARE_MACRO(pow, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_DECLARE_MACRO(atan2, float)
TVMET_DECLARE_MACRO(drem, float)
TVMET_DECLARE_MACRO(fmod, float)
TVMET_DECLARE_MACRO(hypot, float)
TVMET_DECLARE_MACRO(jn, float)
TVMET_DECLARE_MACRO(yn, float)
TVMET_DECLARE_MACRO(pow, float)

TVMET_DECLARE_MACRO(atan2, double)
TVMET_DECLARE_MACRO(drem, double)
TVMET_DECLARE_MACRO(fmod, double)
TVMET_DECLARE_MACRO(hypot,double)
TVMET_DECLARE_MACRO(jn, double)
TVMET_DECLARE_MACRO(yn, double)
TVMET_DECLARE_MACRO(pow, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(atan2, long double)
TVMET_DECLARE_MACRO(drem, long double)
TVMET_DECLARE_MACRO(fmod, long double)
TVMET_DECLARE_MACRO(hypot, long double)
TVMET_DECLARE_MACRO(jn, long double)
TVMET_DECLARE_MACRO(yn, long double)
TVMET_DECLARE_MACRO(pow, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * binary_function(XprMatrix<E>, std::complex<>)
 */
#define TVMET_DECLARE_MACRO(NAME)				\
template<class E,  class T>	\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,	\
    XprMatrix<E>,					\
    XprLiteral< std::complex<T> >				\
  >							\
>								\
NAME(const XprMatrix<E>& lhs, 			\
     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(pow)

TVMET_DECLARE_MACRO(atan2)
TVMET_DECLARE_MACRO(drem)
TVMET_DECLARE_MACRO(fmod)
TVMET_DECLARE_MACRO(hypot)
TVMET_DECLARE_MACRO(jn)
TVMET_DECLARE_MACRO(yn)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*
 * binary_function(XprMatrix<E1>, XprMatrix<E2>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)							\
template<class E1,  class E2>			\
inline											\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,			\
    XprMatrix<E1>,								\
    XprMatrix<E2>								\
  >									\
>											\
NAME(const XprMatrix<E1>& lhs, const XprMatrix<E2>& rhs) {	\
  typedef XprBinOp<									\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,			\
    XprMatrix<E1>,								\
    XprMatrix<E2>								\
  >		    					expr_type;			\
  return XprMatrix<expr_type>(						\
    expr_type(lhs, rhs));								\
}

TVMET_IMPLEMENT_MACRO(atan2)
TVMET_IMPLEMENT_MACRO(drem)
TVMET_IMPLEMENT_MACRO(fmod)
TVMET_IMPLEMENT_MACRO(hypot)
TVMET_IMPLEMENT_MACRO(jn)
TVMET_IMPLEMENT_MACRO(yn)
TVMET_IMPLEMENT_MACRO(pow)
#if defined(TVMET_HAVE_COMPLEX)
  //TVMET_IMPLEMENT_MACRO(polar)
#endif

#undef TVMET_IMPLEMENT_MACRO


/*
 * binary_function(XprMatrix<E>, POD)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, TP)					\
template<class E>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprMatrix<E>,						\
    XprLiteral< TP >							\
  >								\
>									\
NAME(const XprMatrix<E>& lhs, TP rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprMatrix<E>,						\
    XprLiteral< TP >							\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< TP >(rhs)));				\
}

TVMET_IMPLEMENT_MACRO(atan2, int)
TVMET_IMPLEMENT_MACRO(drem, int)
TVMET_IMPLEMENT_MACRO(fmod, int)
TVMET_IMPLEMENT_MACRO(hypot, int)
TVMET_IMPLEMENT_MACRO(jn, int)
TVMET_IMPLEMENT_MACRO(yn, int)
TVMET_IMPLEMENT_MACRO(pow, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(atan2, long long int)
TVMET_IMPLEMENT_MACRO(drem, long long int)
TVMET_IMPLEMENT_MACRO(fmod, long long int)
TVMET_IMPLEMENT_MACRO(hypot, long long int)
TVMET_IMPLEMENT_MACRO(jn, long long int)
TVMET_IMPLEMENT_MACRO(yn,long long int)
TVMET_IMPLEMENT_MACRO(pow, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_IMPLEMENT_MACRO(atan2, float)
TVMET_IMPLEMENT_MACRO(drem, float)
TVMET_IMPLEMENT_MACRO(fmod, float)
TVMET_IMPLEMENT_MACRO(hypot, float)
TVMET_IMPLEMENT_MACRO(jn, float)
TVMET_IMPLEMENT_MACRO(yn, float)
TVMET_IMPLEMENT_MACRO(pow, float)

TVMET_IMPLEMENT_MACRO(atan2, double)
TVMET_IMPLEMENT_MACRO(drem, double)
TVMET_IMPLEMENT_MACRO(fmod, double)
TVMET_IMPLEMENT_MACRO(hypot,double)
TVMET_IMPLEMENT_MACRO(jn, double)
TVMET_IMPLEMENT_MACRO(yn, double)
TVMET_IMPLEMENT_MACRO(pow, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(atan2, long double)
TVMET_IMPLEMENT_MACRO(drem, long double)
TVMET_IMPLEMENT_MACRO(fmod, long double)
TVMET_IMPLEMENT_MACRO(hypot, long double)
TVMET_IMPLEMENT_MACRO(jn, long double)
TVMET_IMPLEMENT_MACRO(yn, long double)
TVMET_IMPLEMENT_MACRO(pow, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * binary_function(XprMatrix<E>, std::complex<>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E,  class T>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >								\
>									\
NAME(const XprMatrix<E>& lhs, const std::complex<T>& rhs) {	\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E>,						\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type>(				\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}

TVMET_IMPLEMENT_MACRO(pow)

TVMET_IMPLEMENT_MACRO(atan2)
TVMET_IMPLEMENT_MACRO(drem)
TVMET_IMPLEMENT_MACRO(fmod)
TVMET_IMPLEMENT_MACRO(hypot)
TVMET_IMPLEMENT_MACRO(jn)
TVMET_IMPLEMENT_MACRO(yn)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace gpumatrix

#endif // TVMET_XPR_MATRIX_BINARY_FUNCTIONS_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
