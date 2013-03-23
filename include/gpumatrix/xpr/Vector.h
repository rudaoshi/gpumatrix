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
* $Id: Vector.h,v 1.28 2007-06-23 15:59:00 opetzold Exp $
*/

#ifndef TVMET_XPR_VECTOR_H
#define TVMET_XPR_VECTOR_H


#include <gpumatrix/impl/Interface.h>

namespace gpumatrix {


	/* forwards */
	template <class T> class Vector;

	/**
	* \class XprVector Vector.h "gpumatrix/xpr/Vector.h"
	* \brief Represents the expression for vectors at any node in the parse tree.
	*
	* Specifically, XprVector is the class that wraps the expression, and the
	* expression itself is represented by the template parameter E. The
	* class XprVector is known as an anonymizing expression wrapper because
	* it can hold any subexpression of arbitrary complexity, allowing
	* clients to work with any expression by holding on to it via the
	* wrapper, without having to know the name of the type object that
	* actually implements the expression.
	* \note leave the Ctors non-explicit to allow implicit type conversation.
	*/
	template<class E>
	class XprVector : public GpuMatrixBase< XprVector<E> >
	{
		XprVector();
		XprVector& operator=(const XprVector&);

	public:
		typedef typename E::value_type			value_type;
		typedef typename XprResultType<E>:: result_type result_type;
	public:


	public:
		///** Complexity counter */
		//enum {
		//  ops_assign = Size,
		//  ops        = E::ops,
		//  use_meta   = ops_assign < TVMET_COMPLEXITY_V_ASSIGN_TRIGGER ? true : false
		//};

	public:
		/** Constructor. */
		explicit XprVector(const E& e)
			: m_expr(e)
		{ }

		/** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
		XprVector(const XprVector& e)
			: m_expr(e.m_expr)
		{ }
#endif

		/** const index operator for vectors. */
		value_type operator()(std::size_t i) const {
			TVMET_RT_CONDITION(i < Size, "XprVector Bounce Violation")
				return m_expr(i);
		}

		/** const index operator for vectors. */
		value_type operator[](std::size_t i) const {
			return this->operator()(i);
		}

		std::size_t rows() const 
		  {
			  return m_expr.rows();
		  }

		  std::size_t cols() const 
		  {
			  return m_expr.cols();
		  }

		const E & expr() const
		{
			return m_expr;
		}

		std::size_t size() const 
		{
			return m_expr.size();
		}

		//private:
		//  /** Wrapper for meta assign. */
		//  template<class Dest, class Src, class Assign>
		//  static inline
		//  void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//    meta::Vector<Size, 0>::assign(dest, src, assign_fn);
		//  }
		//
		//  /** Wrapper for loop assign. */
		//  template<class Dest, class Src, class Assign>
		//  static inline
		//  void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//    loop::Vector<Size>::assign(dest, src, assign_fn);
		//  }

	public:
		/** assign this expression to Vector dest. */
		template<class Dest, class Assign>
		void assign_to(Dest& dest, const Assign& assign_fn) const {
			/* here is a way for caching, since each complex 'Node'
			is of type XprVector. */
			impl::do_assign(dest, m_expr, assign_fn);
		}


		result_type eval() const
		{
			return impl::eval(m_expr);
		}

		value_type squaredNorm()
		{
			return 	eval().squaredNorm();
		}

		operator Eigen::Matrix<value_type,Eigen::Dynamic,1> ()
		{
			return eval();
		}


	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "XprVector<"
				<< std::endl;
			m_expr.print_xpr(os, l);
			os << IndentLevel(l)
				<< "Sz=" << size() << std::endl;
			os << IndentLevel(--l) << ">"
				<< ((l != 0) ? "," : "") << std::endl;
		}

	private:
		const E						m_expr;
	};


} // namespace gpumatrix

#include <gpumatrix/Functional.h>

#include <gpumatrix/xpr/BinOperator.h>
#include <gpumatrix/xpr/UnOperator.h>
#include <gpumatrix/xpr/Literal.h>

#include <gpumatrix/xpr/VectorFunctions.h>
#include <gpumatrix/xpr/VectorBinaryFunctions.h>
#include <gpumatrix/xpr/VectorUnaryFunctions.h>
#include <gpumatrix/xpr/VectorOperators.h>


#endif // TVMET_XPR_VECTOR_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
