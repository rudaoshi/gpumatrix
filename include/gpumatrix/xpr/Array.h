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
* $Id: Matrix.h,v 1.26 2007-06-23 15:59:00 opetzold Exp $
*/

#ifndef TVMET_XPR_ARRAY_H
#define TVMET_XPR_ARRAY_H

//#include <gpumatrix/xpr/ResultType.h>


#include <gpumatrix/xpr/Simplify.h>

#include <gpumatrix/impl/Interface.h>

namespace gpumatrix {


	/* forwards */
	template <class T,int D> class Array;
	template <class T> class Matrix;
	template <class E> class RowWiseView;
	template <class E> class ColWiseView;
	/**
	* \class XprMatrix Matrix.h "gpumatrix/xpr/Matrix.h"
	* \brief Represents the expression for vectors at any node in the parse tree.
	*
	* Specifically, XprMatrix is the class that wraps the expression, and the
	* expression itself is represented by the template parameter E. The
	* class XprMatrix is known as an anonymizing expression wrapper because
	* it can hold any subexpression of arbitrary complexity, allowing
	* clients to work with any expression by holding on to it via the
	* wrapper, without having to know the name of the type object that
	* actually implements the expression.
	* \note leave the CCtors non-explicit to allow implicit type conversation.
	*/
	template<class E, int D>
	class XprArray
		: public GpuMatrixBase< XprArray<E, D> >
	{
		XprArray();
		XprArray& operator=(const XprArray&);

	public:
		///** Dimensions. */
		//enum {
		//  Rows = NRows,			/**< Number of rows. */
		//  Cols = NCols,			/**< Number of cols. */
		//  Size = Rows * Cols			/**< Complete Size of Matrix. */
		//};
		//std::size_t Rows;
		//std::size_t Cols;
		//std::size_t Size;

	public:
		/** Complexity counter. */
		//enum {
		//  ops_assign = Rows * Cols,
		//  ops        = E::ops,
		//  use_meta   = ops_assign < TVMET_COMPLEXITY_M_ASSIGN_TRIGGER ? true : false
		//};

	public:
		typedef typename E::value_type			value_type;
		typedef typename XprResultType<E>:: result_type result_type;

	public:
		/** Constructor. */
		explicit XprArray(const E& e)
			: m_expr(e)/*,Rows(e.rows()),Cols(e.cols()),Size(e.rows()*e.cols())*/
		{ }

		/** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
		XprArray(const XprArray& rhs)
			: m_expr(rhs.m_expr)/*,Rows(e.rows()),Cols(e.cols()),Size(e.rows()*e.cols())*/
		{ }
#endif

		/** access by index. */
		value_type operator()(std::size_t i, std::size_t j) const {
			TVMET_RT_CONDITION((i < Rows) && (j < Cols), "XprMatrix Bounce Violation")
				return m_expr(i, j);
		}

		const E & expr() const
		{
			return m_expr;
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
			return m_expr.size();
		}

		Matrix<value_type> matrix()
		{
			Matrix<value_type> result(rows(),cols());

			result.array() = eval();

			return result;
		}

		
		RowWiseView<XprArray<E,D>> rowwise() const
		{
			return RowWiseView<XprArray<E,D>>(*this);
		}

		ColWiseView<XprArray<E,D>> colwise() const
		{
			return ColWiseView<XprArray<E,D>>(*this);
		}

		///** Wrapper for meta assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//  meta::Matrix<Rows, 0, 0>::assign(dest, src, assign_fn);
		//}

		///** Wrapper for loop assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//  loop::Matrix<Rows>::assign(dest, src, assign_fn);
		//}
		XprArray<
		  XprUnOp<
			Fcnl_arrayinv<value_type>,
			XprArray<E,D>
		  >,D>  inverse() const
		{
			typedef XprUnOp<Fcnl_arrayinv<value_type>,XprArray<E,D>> op_type;
			return 	XprArray< op_type,D>(op_type(*this));
		}

		XprArray<
		  XprUnOp<
			Fcnl_exp<value_type>,
			XprArray<E,D>
		  >,D>  exp() const
		{
			typedef XprUnOp<Fcnl_exp<value_type>,XprArray<E,D>> op_type;
			return 	XprArray< op_type,D>(op_type(*this));
		}

		XprArray<
		  XprUnOp<
			Fcnl_log<value_type>,
			XprArray<E,D>
		  >,D>  log() const
		{
			typedef XprUnOp<Fcnl_log<value_type>,XprArray<E,D>> op_type;
			return 	XprArray< op_type,D>(op_type(*this));
		}
				XprArray<
		  XprUnOp<
			Fcnl_logistic<value_type>,
			XprArray<E,D>
		  >,D>  logistic() const
		{
			typedef XprUnOp<Fcnl_logistic<value_type>,XprArray<E,D>> op_type;
			return 	XprArray< op_type,D>(op_type(*this));
		}

		value_type squaredNorm() const
		{
			return eval().squaredNorm();
		}


		value_type sum() const
		{
			return eval().sum();
		}


	public:
		/** assign this expression to Matrix dest. */
		template<class Dest, class Assign>
		void assign_to(Dest& dest, const Assign& assign_fn) const {
			/* here is a way for caching, since each complex 'Node'
			is of type XprMatrix. */
			impl::do_assign(dest, m_expr, assign_fn);
		}

		result_type eval() const
		{
			return gpumatrix::impl::eval(m_expr);
		}


	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "XprArray<"
				<< std::endl;
			m_expr.print_xpr(os, l);
			os << IndentLevel(l)
				<< "R=" << rows() << ", C=" << cols() << std::endl;
			os << IndentLevel(--l) << ">"
				<< ((l != 0) ? "," : "") << std::endl;
		}

	private:
		const E						m_expr;
	};


	//template<class T> class MatrixConstReference;
	//
	//template <class T> 
	//MatrixConstReference<T> XprMatrix<MatrixConstReference<T>>::eval() const
	//{
	//	return m_expr;
	//}

} // namespace gpumatrix

#include <gpumatrix/Functional.h>

#include <gpumatrix/xpr/BinOperator.h>
#include <gpumatrix/xpr/UnOperator.h>
#include <gpumatrix/xpr/Literal.h>
//
#include <gpumatrix/xpr/Identity.h>

//
#include <gpumatrix/xpr/ArrayFunctions.h>
#include <gpumatrix/xpr/ArrayOperators.h>




#endif // TVMET_XPR_MATRIX_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
