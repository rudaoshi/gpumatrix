/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2007 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: MVProduct.h,v 1.21 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_MVPRODUCT_H
#define TVMET_XPR_MVPRODUCT_H

//#include <gpumatrix/meta/Gemv.h>
//#include <gpumatrix/loop/Gemv.h>

namespace gpumatrix {


/**
 * \class XprMVProduct MVProduct.h "gpumatrix/xpr/MVProduct.h"
 * \brief Expression for matrix-vector product
 *        using formula
 *        \f[
 *        M\,v
 *        \f]
 */
template<class E1,
	 class E2>
class XprMVProduct
  : public XprBinOpBase<E1,E2,XprMVProduct<E1,E2>>,public GpuMatrixBase< XprMVProduct<E1, E2> >
{
  XprMVProduct();
  XprMVProduct& operator=(const XprMVProduct&);
  
  using XprBinOpBase<E1,E2,XprMVProduct<E1,E2>>::m_lhs;
  using XprBinOpBase<E1,E2,XprMVProduct<E1,E2>>::m_rhs;

public:
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type							value_type;

  typedef typename XprResultType<XprMVProduct<E1,E2>>::result_type result_type;

public:
  ///** Complexity counter. */
  //enum {
  //  ops_lhs   = E1::ops,
  //  ops_rhs   = E2::ops,
  //  M         = Rows * Cols,
  //  N         = Rows * (Cols - 1),
  //  ops_plus  = M * NumericTraits<value_type>::ops_plus,
  //  ops_muls  = N * NumericTraits<value_type>::ops_muls,
  //  ops       = ops_plus + ops_muls,
  //  use_meta  = Rows*Cols < TVMET_COMPLEXITY_MV_TRIGGER ? true : false
  //};

public:
  /** Constructor. */
  explicit XprMVProduct(const E1& lhs, const E2& rhs)
    : XprBinOpBase<E1,E2,XprMVProduct<E1,E2>>(lhs,rhs)
  {
	  if (lhs.cols() != rhs.rows())
		  throw runtime_error("Dimension not Match for Matrix Vector Multiplication");
  }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMVProduct(const XprMVProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

//private:
//  /** Wrapper for meta gemm. */
//  static inline
//  value_type do_gemv(dispatch<true>, const E1& lhs, const E2& rhs, std::size_t j) {
//    return meta::gemv<Rows,
//                      0>::prod(lhs, rhs, j);
//  }
//
//  /** Wrapper for loop gemm. */
//  static inline
//  value_type do_gemv(dispatch<false>, const E1& lhs, const E2& rhs, std::size_t j) {
//    return loop::gemv<Rows>::prod(lhs, rhs, j);
//  }

public:
  ///** index operator, returns the expression by index. This is the vector
  //    style since a matrix*vector gives a vector. */
  //value_type operator()(std::size_t j) const {
  //  TVMET_RT_CONDITION(j < Rows , "XprMVProduct Bounce Violation")
  //  return do_gemv(dispatch<use_meta>(), m_lhs, m_rhs, j);
  //}

	std::size_t rows() const 
	{
		return m_lhs.rows();
	}

	std::size_t cols() const 
	{
		return 1;
	}

	std::size_t size() const 
	{
		return m_lhs.rows();
	}

		result_type eval() const
		{
			return impl::eval(*this);

		}

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMVProduct<"
       << std::endl;
    m_lhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "R=" << rows() << ", C=" << cols() << ",\n";
    m_rhs.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }


};


} // namespace gpumatrix

#endif // TVMET_XPR_MVPRODUCT_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
