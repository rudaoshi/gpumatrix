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
 * $Id: Literal.h,v 1.13 2007-06-23 15:58:59 opetzold Exp $
 */

#ifndef TVMET_XPR_LITERAL_H
#define TVMET_XPR_LITERAL_H

namespace gpumatrix {


/**
 * \class XprLiteral Literal.h "gpumatrix/xpr/Literal.h"
 * \brief Specify literals like scalars into the expression.
 *        This expression is used for vectors and matrices - the
 *        decision is done by the access operator.
 */
template<class T>
class XprLiteral
  : public GpuMatrixBase< XprLiteral<T> >
{
  XprLiteral();
  XprLiteral& operator=(const XprLiteral&);

public:
  typedef T						value_type;

public:
  ///** Complexity counter. */
  //enum {
  //  ops       = 1
  //};
	
	std::size_t rows() const 
  {
	  return 1;
  }

  std::size_t cols() const 
  {
	  return 1;
  }

  std::size_t size() const 
  {
	  return 1;
  }

  T eval() const
  {
	  return m_data;
  }


public:
  /** Constructor by value for literals . */
  explicit XprLiteral(value_type value)
    : m_data(value)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprLiteral(const XprLiteral& e)
    : m_data(e.m_data)
  { }
#endif

  /** Index operator, gives the value for vectors. */
  value_type operator()(std::size_t) const { return m_data; }

  /** Index operator for arrays/matrices. */
  value_type operator()(std::size_t, std::size_t) const { return m_data; }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++) << "XprLiteral<T="
       << typeid(value_type).name()
       << ">," << std::endl;
  }

private:
  const value_type 					m_data;
};


} // namespace gpumatrix

#endif // TVMET_XPR_LITERAL_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
