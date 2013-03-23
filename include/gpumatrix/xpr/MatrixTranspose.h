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
* $Id: MatrixTranspose.h,v 1.15 2007-06-23 15:59:00 opetzold Exp $
*/

#ifndef TVMET_XPR_MATRIX_TRANSPOSE_H
#define TVMET_XPR_MATRIX_TRANSPOSE_H


#include <gpumatrix/xpr/UnOpBase.h>

namespace gpumatrix {


	/**
	* \class XprMatrixTranspose MatrixTranspose.h "gpumatrix/xpr/MatrixTranspose.h"
	* \brief Expression for transpose matrix
	*/
	template<class E>
	class XprMatrixTranspose
		: public XprUnOpBase<E,XprMatrixTranspose<E> >, public GpuMatrixBase< XprMatrixTranspose<E> >
	{
		XprMatrixTranspose();
		XprMatrixTranspose& operator=(const XprMatrixTranspose&);
		
		using XprUnOpBase<E,XprMatrixTranspose<E> >::m_expr;

	public:
		typedef typename E::value_type			value_type;

		typedef typename XprResultType<XprMatrixTranspose<E>>::result_type result_type;

	public:
		/** Constructor. */
		explicit XprMatrixTranspose(const E& e)
			: XprUnOpBase<E,XprMatrixTranspose<E> >(e)
		{ }

		/** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
		XprMatrixTranspose(const XprMatrixTranspose& e)
			: m_expr(e.m_expr)
		{ }
#endif

		/** index operator for arrays/matrices. This simple swap the index
		access for transpose. */
		value_type operator()(std::size_t i, std::size_t j) const { return m_expr(j, i); }
	public:

		std::size_t rows() const 
		{
			return m_expr.cols();
		}

		std::size_t cols() const
		{
			return m_expr.rows();
		}

		std::size_t size() const 
		{
			return m_expr.size();
		}

		result_type eval() const
		{
			return impl::eval(*this);
		}

		const E & expr() const
		{
			return m_expr;
		}

	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "XprMatrixTranspose<"
				<< std::endl;
			m_expr.print_xpr(os, l);
			os << IndentLevel(--l)
				<< ">," << std::endl;
		}

	};


} // namespace gpumatrix

#endif // TVMET_XPR_MATRIX_TRANSPOSE_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
