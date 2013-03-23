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
 * $Id: MMProductTransposed.h,v 1.20 2007-06-23 15:58:59 opetzold Exp $
 */

#ifndef TVMET_XPR_MTMTPRODUCT_H
#define TVMET_XPR_MTMTPRODUCT_H

#include <gpumatrix/xpr/BinOpBase.h>

namespace gpumatrix {


/**
 * \class XprMMProductTransposed MMProductTransposed.h "gpumatrix/xpr/MMProductTransposed.h"
 * \brief Expression for transpose(matrix-matrix product).
 *        Using formula:
 *        \f[
 *        (M_1\,M_2)^T
 *        \f]
 * \note The Rows2 has to be  equal to Cols1.
 *       The result is a (Cols2 x Rows1) matrix.
 */
template<class E1,
	 class E2>
class XprMtMtProduct
  : public XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >, public GpuMatrixBase< XprMtMtProduct<E1, E2> >
{
private:
  XprMtMtProduct();
  XprMtMtProduct& operator=(const XprMtMtProduct&);
  
  using XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >::m_lhs;
  using XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >::m_rhs;

public:
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type							value_type;
  typedef typename XprResultType<XprMtMtProduct<E1,E2>>::result_type result_type;
public:
	std::size_t rows() const 
	{
		return m_lhs.cols();
	}

	std::size_t cols() const 
	{
		return m_rhs.rows();
	}

	std::size_t size() const 
	{
		return rows()*cols();
	}

	result_type eval() const
	{
		return impl::eval(*this);

	}
public:
  /** Constructor. */
  explicit XprMtMtProduct(const E1& lhs, const E2& rhs)
    : XprBinOpBase<E1,E2,XprMMtProduct<E1,E2> >(lhs,rhs) { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMtMtProduct(const XprMtMtProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  ///** Wrapper for meta gemm. */
  //static inline
  //value_type do_gemm(dispatch<true>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
  //  return meta::gemm<Rows1,
  //                    Cols2,
  //                    0>::prod(lhs, rhs, i, j);
  //}

  ///** Wrapper for loop gemm. */
  //static inline
  //value_type do_gemm(dispatch<false>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
  //  return loop::gemm<Rows1>::prod(lhs, rhs, i, j);
  //}

public:


public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMtMtProduct<"
       << std::endl;
    m_lhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "R1=" << m_lhs.rows() << ", C1=" << m_lhs.cols() << ",\n";
    m_rhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "C2=" << m_rhs.cols() << ",\n"
       << IndentLevel(l)
       << "\n"
       << IndentLevel(--l)
       << ">," << std::endl;
  }


};


} // namespace gpumatrix

#endif // TVMET_XPR_MMPRODUCT_TRANSPOSED_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
