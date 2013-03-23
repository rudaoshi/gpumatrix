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
* $Id: MtMProduct.h,v 1.19 2007-06-23 15:59:00 opetzold Exp $
*/

#ifndef TVMET_XPR_MTMPRODUCT_H
#define TVMET_XPR_MTMPRODUCT_H

#include <gpumatrix/xpr/BinOpBase.h>

namespace gpumatrix {


	/**
	* \class XprMtMProduct MtMProduct.h "gpumatrix/xpr/MtMProduct.h"
	* \brief Expression for product of transposed(matrix)-matrix product.
	*        using formula
	*        \f[
	*        M_1^{T}\,M_2
	*        \f]
	* \note The number of rows of rhs matrix have to be equal rows of rhs matrix,
	*       since lhs matrix 1 is transposed.
	*       The result is a (Cols1 x Cols2) matrix.
	*/
	template<class E1,	 class E2>
	class XprMtMProduct
		: public XprBinOpBase<E1,E2,XprMtMProduct<E1,E2> >, public GpuMatrixBase< XprMtMProduct<E1, E2> >
	{
	private:
		XprMtMProduct();
		XprMtMProduct& operator=(const XprMtMProduct&);
		
		using XprBinOpBase<E1,E2,XprMtMProduct<E1,E2> >::m_lhs;
		using XprBinOpBase<E1,E2,XprMtMProduct<E1,E2> >::m_rhs;

	public:
		typedef typename PromoteTraits<
			typename E1::value_type,
			typename E2::value_type
		>::value_type							value_type;

		typedef typename XprResultType<XprMtMProduct<E1,E2>>::result_type result_type;

	public:
		std::size_t rows() const 
		{
			return m_lhs.cols();
		}

		std::size_t cols() const 
		{
			return m_rhs.cols();
		}

		std::size_t size() const 
		{
			return m_lhs.cols()*m_rhs.cols();
		}

		result_type eval() const
		{
			return impl::eval(*this);

		}
	public:
		/** Constructor. */
		explicit XprMtMProduct(const E1& lhs, const E2& rhs)
			: XprBinOpBase<E1,E2,XprMtMProduct<E1,E2> >(lhs,rhs)
		{ }

		/** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
		XprMtMProduct(const XprMtMProduct& e)
			: m_lhs(e.m_lhs), m_rhs(e.m_rhs) { }
#endif

	private:

	public:

	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "XprMtMProduct<"
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

#endif // TVMET_XPR_MTMPRODUCT_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
