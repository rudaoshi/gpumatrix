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
 * $Id: BinOperator.h,v 1.19 2007-06-23 15:58:59 opetzold Exp $
 */

#ifndef TVMET_XPR_BINOPERATOR_H
#define TVMET_XPR_BINOPERATOR_H

#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/xpr/BinOpBase.h>
#include <stdexcept>

using namespace std;

namespace gpumatrix {

	namespace gpu
	{
		template <class E>
		typename XprResultType<E>::result_type eval(const E & expr);
	}

	template <class E1, class E2> 
	void check_dim(const E1 & expr1, const E2 & expr2)
	{
		if (expr1.rows() != expr2.rows() || expr1.cols() != expr2.cols() || expr1.size() != expr2.size())
		{
			throw runtime_error("Dimension do not match!");
		}
	}
	template <class POD, class E> 
	void check_dim(const XprLiteral<POD> & expr1, const E & expr2)
	{
	}
		template <class POD, class E> 
	void check_dim(const E & expr1,const XprLiteral<POD> & expr2)
	{
		
	}
	
	template <class E1, class E2> 
	std::size_t _row(const E1 & expr1, const E2 & expr2)
	{
		return expr1.rows();
	}
	template <class POD, class E> 
	std::size_t _row(const XprLiteral<POD> & expr1, const E & expr2)
	{
		return expr2.rows();
	}
	template <class POD, class E> 
	std::size_t _row(const E & expr1,const XprLiteral<POD> & expr2)
	{
		return expr1.rows();
	}
	
	template <class E1, class E2> 
	std::size_t _col(const E1 & expr1, const E2 & expr2)
	{
		return expr1.cols();
	}
	template <class POD, class E> 
	std::size_t _col(const XprLiteral<POD> & expr1, const E & expr2)
	{
		return expr2.cols();
	}
	template <class POD, class E> 
	std::size_t _col(const E & expr1,const XprLiteral<POD> & expr2)
	{
		return expr1.cols();
	}

	template <class E1, class E2> 
	std::size_t _size(const E1 & expr1, const E2 & expr2)
	{
		return expr1.size();
	}
	template <class POD, class E> 
	std::size_t _size(const XprLiteral<POD> & expr1, const E & expr2)
	{
		return expr2.size();
	}
	template <class POD, class E> 
	std::size_t _size(const E & expr1,const XprLiteral<POD> & expr2)
	{
		return expr1.size();
	}
	
	
/**
 * \class XprBinOp BinOperator.h "gpumatrix/xpr/BinOperator.h"
 * \brief Binary operators working on two sub expressions.
 *
 * On acessing using the index operator() the binary operation will be
 * evaluated at compile time.
 */
template<class BinOp, class E1, class E2>
class XprBinOp
  : public XprBinOpBase<E1,E2, XprBinOp<BinOp,E1,E2>>, public GpuMatrixBase< XprBinOp<BinOp, E1, E2> >
{
  XprBinOp();
  XprBinOp& operator=(const XprBinOp&);
  
  using XprBinOpBase<E1,E2, XprBinOp<BinOp,E1,E2>>::m_lhs;
  using XprBinOpBase<E1,E2, XprBinOp<BinOp,E1,E2>>::m_rhs;

public:
  
  typedef typename BinOp::value_type			value_type;
  typedef typename XprResultType<XprBinOp<BinOp,E1,E2>>::result_type result_type;
public:
  /** Complexity counter. */
  //enum {
  //  ops_lhs   = E1::ops,
  //  ops_rhs   = E2::ops,
  //  ops       = 2 * (ops_lhs + ops_rhs) // lhs op rhs
  //};

public:
  /** Constructor for two expressions. */
  explicit XprBinOp(const E1& lhs, const E2& rhs)
    : XprBinOpBase<E1,E2, XprBinOp<BinOp,E1,E2>>(lhs,rhs)
  {
	  check_dim(lhs,rhs);
  }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprBinOp(const XprBinOp& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

  std::size_t rows() const 
  {
	  return _row(m_lhs,m_rhs);
  }

  std::size_t cols() const 
  {
	  return _col(m_lhs,m_rhs);
  }

  std::size_t size() const 
  {
	  return _size(m_lhs,m_rhs);
  }

  result_type eval() const
  {
	  return impl::eval(*this);
  }

  ///** Index operator, evaluates the expression inside. */
  //value_type operator()(std::size_t i) const {
  //  return BinOp::apply_on(m_lhs(i), m_rhs(i));
  //}

  ///** Index operator for arrays/matrices */
  //value_type operator()(std::size_t i, std::size_t j) const {
  //  return BinOp::apply_on(m_lhs(i, j), m_rhs(i, j));
  //}

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprBinOp<"
       << std::endl;
    BinOp::print_xpr(os, l);
    m_lhs.print_xpr(os, l);
    m_rhs.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

//private:
//  const E1						m_lhs;
//  const E2						m_rhs;
};


} // namespace gpumatrix

#endif // TVMET_XPR_BINOPERATOR_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
