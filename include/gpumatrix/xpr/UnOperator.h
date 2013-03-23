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
 * $Id: UnOperator.h,v 1.17 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_UNOPERATOR_H
#define TVMET_XPR_UNOPERATOR_H

#include <gpumatrix/xpr/UnOpBase.h>

namespace gpumatrix {


/**
 * \class XprUnOp UnOperator.h "gpumatrix/xpr/UnOperator.h"
 * \brief Unary operator working on one subexpression.
 *
 * Using the access operator() the unary operation will be evaluated.
 */
template<class UnOp, class E>
class XprUnOp
  : public XprUnOpBase<E, XprUnOp<UnOp,E>> ,public GpuMatrixBase< XprUnOp<UnOp, E> >
{
  XprUnOp();
  XprUnOp& operator=(const XprUnOp&);
  
  using XprUnOpBase<E, XprUnOp<UnOp,E>>::m_expr;

public:
  typedef typename E::value_type				value_type;
  typedef typename XprResultType<XprUnOp<UnOp,E>>::result_type result_type;
public:
  ///** Complexity counter. */
  //enum {
  //  ops_expr  = E::ops,
  //  ops       = 1 * ops_expr
  //};

public:
  /** Constructor for an expressions. */
  explicit XprUnOp(const E& e)
    : XprUnOpBase<E, XprUnOp<UnOp,E>>(e)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprUnOp(const XprUnOp& e)
    : m_expr(e.m_expr)
  { }
#endif

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprUnOp<"
       << std::endl;
    UnOp::print_xpr(os, l);
    m_expr.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

	std::size_t rows() const 
	{
		return m_expr.rows();
	}

	std::size_t cols() const 
	{
		return m_expr.cols();
	}

	std::size_t size() const
	{
		return rows()*cols();
	}

	result_type eval() const
	{
		return impl::eval(*this);
	}


};


} // namespace gpumatrix

#endif // TVMET_XPR_UNOPERATOR_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
