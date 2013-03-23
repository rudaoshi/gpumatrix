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
 * $Id: MtVProduct.h,v 1.14 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_MTVPRODUCT_H
#define TVMET_XPR_MTVPRODUCT_H


namespace gpumatrix {


/**
 * \class XprMtVProduct MtVProduct.h "gpumatrix/xpr/MtVProduct.h"
 * \brief Expression for matrix-transposed vector product
 *        using formula
 *        \f[
 *        M^T\,v
 *        \f]
 */
template<class E1, 
	 class E2>
class XprMtVProduct
  : public XprBinOpBase<E1,E2,XprMtVProduct<E1,E2>>,public GpuMatrixBase< XprMtVProduct<E1, E2> >
{
  XprMtVProduct();
  XprMtVProduct& operator=(const XprMtVProduct&);
  
  using XprBinOpBase<E1,E2,XprMtVProduct<E1,E2>>::m_lhs;
  using XprBinOpBase<E1,E2,XprMtVProduct<E1,E2>>::m_rhs;

public:
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type							value_type;
   typedef typename XprResultType<XprMtVProduct<E1,E2>>::result_type result_type;
public:

public:
  /** Constructor. */
  explicit XprMtVProduct(const E1& lhs, const E2& rhs)
    : XprBinOpBase<E1,E2,XprMtVProduct<E1,E2>>(lhs,rhs)
  {
	  if (lhs.rows() != rhs.size())
		  throw runtime_error("Dimension not Match for Matrix Vector Multiplication");
  }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMtVProduct(const XprMtVProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  ///** Wrapper for meta gemm. */
  //static inline
  //value_type do_gemtv(dispatch<true>, const E1& lhs, const E2& rhs, std::size_t i) {
  //  return meta::gemtv<Rows, 0>::prod(lhs, rhs, i);
  //}

  ///** Wrapper for loop gemm. */
  //static inline
  //value_type do_gemtv(dispatch<false>, const E1& lhs, const E2& rhs, std::size_t i) {
  //  return loop::gemtv<Rows>::prod(lhs, rhs, i);
  //}

public:
		std::size_t rows() const 
	{
		return m_lhs.cols();
	}

	std::size_t cols() const 
	{
		return 1;
	}

	std::size_t size() const 
	{
		return m_lhs.cols();
	}

			result_type eval() const
		{
			return impl::eval(*this);

		}

public:


public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMtVProduct<"
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

#endif // TVMET_XPR_MTVPRODUCT_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
