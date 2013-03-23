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
 * $Id: MMtProduct.h,v 1.20 2007-06-23 15:58:59 opetzold Exp $
 */

#ifndef TVMET_XPR_MMTPRODUCT_H
#define TVMET_XPR_MMTPRODUCT_H

#include <gpumatrix/xpr/BinOpBase.h>

namespace gpumatrix {


/**
 * \class XprMMtProduct MMtProduct.h "gpumatrix/xpr/MMtProduct.h"
 * \brief Expression for matrix-matrix product.
 *        Using formula:
 *        \f[
 *        M_1\,M_2^T
 *        \f]
 * \note The number of cols of rhs matrix have to be equal to cols of rhs matrix.
 *       The result is a (Rows1 x Rows2) matrix.
 */
template<class E1, class E2>
class XprMMtProduct
  : public XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >, public GpuMatrixBase< XprMMtProduct<E1, E2> >
{
private:
  XprMMtProduct();
  XprMMtProduct& operator=(const XprMMtProduct&);
  
  using XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >::m_lhs;
  using XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >::m_rhs;

public:
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type							value_type;
  typedef typename XprResultType<XprMMtProduct<E1,E2>>::result_type result_type;

public:
			std::size_t rows() const 
		{
			return m_lhs.rows();
		}

		std::size_t cols() const 
		{
			return m_rhs.rows();
		}

		std::size_t size() const 
		{
			return m_lhs.rows()*m_rhs.rows();
		}

		result_type eval() const
		{
			return impl::eval(*this);

		}
public:
  /** Constructor. */
  explicit XprMMtProduct(const E1& lhs, const E2& rhs)
    : XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >(lhs,rhs)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMMtProduct(const XprMMtProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  ///** Wrapper for meta gemm. */
  //static inline
  //value_type do_gemmt(dispatch<true>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
  //  return meta::gemmt<Rows1,
  //                     Cols2,
  //                     0>::prod(lhs, rhs, i, j);
  //}

  ///** Wrapper for loop gemm. */
  //static inline
  //value_type do_gemmt(dispatch<false>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
  //  return loop::gemmt<Rows1>::prod(lhs, rhs, i, j);
  //}

public:


public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMMtProduct<"
       << std::endl;
    m_lhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "R1=" << m_lhs.rows() << ", C1=" << m_lhs.cols() << ",\n";
    m_rhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "C2=" << m_rhs.cols() << ",\n"
       << "\n"
       << IndentLevel(--l)
       << ">," << std::endl;
  }

};


} // namespace gpumatrix

#endif // TVMET_XPR_MMTPRODUCT_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
