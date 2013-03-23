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
 * $Id: MatrixEval.h,v 1.18 2007-06-23 15:58:58 opetzold Exp $
 */

#ifndef TVMET_MATRIX_EVAL_H
#define TVMET_MATRIX_EVAL_H

namespace gpumatrix {


/**
 * \fn bool all_elements(const XprMatrix<E>& e)
 * \brief check on statements for all elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * all_elements(matrix > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
// template<class E>
// inline
// bool all_elements(const XprMatrix<E>& e) {
//   return meta::Matrix<, 0, 0>::all_elements(e);
// }


/**
 * \fn bool any_elements(const XprMatrix<E>& e)
 * \brief check on statements for any elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * any_elements(matrix > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
// template<class E>
// inline
// bool any_elements(const XprMatrix<E>& e) {
//   return meta::Matrix< 0, 0>::any_elements(e);
// }

#if 1>2 //trinary operator is not supported

/*
 * trinary evaluation functions with matrizes and xpr of
 *
 * XprMatrix<E1> ? Matrix<T2> : Matrix<T3>
 * XprMatrix<E1> ? Matrix<T2> : XprMatrix<E3>
 * XprMatrix<E1> ? XprMatrix<E2> : Matrix<T3>
 * XprMatrix<E1> ? XprMatrix<E2> : XprMatrix<E3>
 */

/**
 * \fn eval(const XprMatrix<E1>& e1, const Matrix<T2>& m2, const Matrix<T3>& m3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class T3>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    MatrixConstReference<T2>,
    MatrixConstReference<T3>
  >,
  
>
eval(const XprMatrix<E1>& e1,
     const Matrix<T2>& m2,
     const Matrix<T3>& m3) {
  typedef XprEval<
    XprMatrix<E1>,
    MatrixConstReference<T2>,
    MatrixConstReference<T3>
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e1, m2.as_expr(), m3.as_expr()));
}


/**
 * \fn eval(const XprMatrix<E1>& e1, const Matrix<T2>& m2, const XprMatrix<E3>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class E3>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    MatrixConstReference<T2>,
    XprMatrix<E3>
  >,
  
>
eval(const XprMatrix<E1>& e1,
     const Matrix<T2>& m2,
     const XprMatrix<E3>& e3) {
  typedef XprEval<
    XprMatrix<E1>,
    MatrixConstReference<T2>,
    XprMatrix<E3>
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e1, m2.as_expr(), e3));
}


/**
 * \fn eval(const XprMatrix<E1>& e1, const XprMatrix<E2>& e2, const Matrix<T3>& m3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class T3>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    MatrixConstReference<T3>
  >,
  
>
eval(const XprMatrix<E1>& e1,
    const  XprMatrix<E2>& e2,
     const Matrix<T3>& m3) {
  typedef XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    MatrixConstReference<T3>
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e1, e2, m3.as_expr()));
}


/**
 * \fn eval(const XprMatrix<E1>& e1, const XprMatrix<E2>& e2, const XprMatrix<E3>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class E3>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    XprMatrix<E3>
  >,
  
>
eval(const XprMatrix<E1>& e1,
     const XprMatrix<E2>& e2,
     const XprMatrix<E3>& e3) {
  typedef XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    XprMatrix<E3>
  > 							expr_type;
  return XprMatrix<expr_type>(expr_type(e1, e2, e3));
}


/*
 * trinary evaluation functions with matrizes, xpr of and POD
 *
 * XprMatrix<E> ? POD1 : POD2
 * XprMatrix<E1> ? POD : XprMatrix<E3>
 * XprMatrix<E1> ? XprMatrix<E2> : POD
 */
#define TVMET_IMPLEMENT_MACRO(POD)               			\
template<class E>			\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E>,               			      	\
    XprLiteral< POD >,               					\
    XprLiteral< POD >               					\
  >,                							\
  								\
>               							\
eval(const XprMatrix<E>& e, POD x2, POD x3) {      		\
  typedef XprEval<               					\
    XprMatrix<E>,               				\
    XprLiteral< POD >,                					\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprMatrix<expr_type>(				\
    expr_type(e, XprLiteral< POD >(x2), XprLiteral< POD >(x3))); 	\
}               							\
               								\
template<class E1, class E3> 	\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E1>,               				\
    XprLiteral< POD >,               					\
    XprMatrix<E3>               				\
  >,                							\
  								\
>               							\
eval(const XprMatrix<E1>& e1, POD x2, const XprMatrix<E3>& e3) { \
  typedef XprEval<               					\
    XprMatrix<E1>,               				\
    XprLiteral< POD >,                					\
    XprMatrix<E3>               				\
  > 							expr_type; 	\
  return XprMatrix<expr_type>(				\
    expr_type(e1, XprLiteral< POD >(x2), e3)); 				\
}               							\
               								\
template<class E1, class E2>	\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E1>,               				\
    XprMatrix<E2>,               				\
    XprLiteral< POD >               					\
  >,                							\
  								\
>               							\
eval(const XprMatrix<E1>& e1, const XprMatrix<E2>& e2, POD x3) { \
  typedef XprEval<               					\
    XprMatrix<E1>,               				\
    XprMatrix<E2>,               				\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprMatrix<expr_type>(				\
    expr_type(e1, e2, XprLiteral< POD >(x3))); 				\
}

TVMET_IMPLEMENT_MACRO(int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(long long int)
#endif

TVMET_IMPLEMENT_MACRO(float)
TVMET_IMPLEMENT_MACRO(double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(long double)
#endif

#undef TVMET_IMPLEMENT_MACRO


/*
 * trinary evaluation functions with matrizes, xpr of and complex<> types
 *
 * XprMatrix<E> e, std::complex<T> z2, std::complex<T> z3
 * XprMatrix<E1> e1, std::complex<T> z2, XprMatrix<E3> e3
 * XprMatrix<E1> e1, XprMatrix<E2> e2, std::complex<T> z3
 */
#if defined(TVMET_HAVE_COMPLEX)

/**
 * \fn eval(const XprMatrix<E>& e, const std::complex<T>& x2, const std::complex<T>& x3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E,  class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  >,
  
>
eval(const XprMatrix<E>& e, const std::complex<T>& x2, const std::complex<T>& x3) {
  typedef XprEval<
    XprMatrix<E>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e, XprLiteral< std::complex<T> >(x2), XprLiteral< std::complex<T> >(x3)));
}


/**
 * \fn eval(const XprMatrix<E1>& e1, const std::complex<T>& x2, const XprMatrix<E3>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E3,  class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    XprLiteral< std::complex<T> >,
    XprMatrix<E3>
  >,
  
>
eval(const XprMatrix<E1>& e1, const std::complex<T>& x2, const XprMatrix<E3>& e3) {
  typedef XprEval<
    XprMatrix<E1>,
    XprLiteral< std::complex<T> >,
    XprMatrix<E3>
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e1, XprLiteral< std::complex<T> >(x2), e3));
}


/**
 * \fn eval(const XprMatrix<E1>& e1, const XprMatrix<E2>& e2, const std::complex<T>& x3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2,  class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    XprLiteral< std::complex<T> >
  >,
  
>
eval(const XprMatrix<E1>& e1, const XprMatrix<E2>& e2, const std::complex<T>& x3) {
  typedef XprEval<
    XprMatrix<E1>,
    XprMatrix<E2>,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprMatrix<expr_type>(
    expr_type(e1, e2, XprLiteral< std::complex<T> >(x3)));
}
#endif // defined(TVMET_HAVE_COMPLEX)
#endif  // trinary operator is not supported

} // namespace gpumatrix

#endif // TVMET_MATRIX_EVAL_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
